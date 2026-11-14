
import streamlit as st
from pathlib import Path
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import faiss
import numpy as np

st.set_page_config(
    page_title="Zyro HR Assistant",
    page_icon="📚",
    layout="centered"
)

st.markdown("""
<style>
.stApp {
    background: #faf7f5;
    color: #30243a;
}
h1, h2, h3 {
    color: #593568;
}
.stChatMessage {
    background: #f0e9f4;
    border-radius: 14px;
}
.stButton button {
    background: #684078;
    color: white;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("📚 Zyro Dynamics HR Assistant")
st.caption(
    "Your AI-powered assistant for HR policies and workplace questions."
)

PDF_FOLDER = Path("hr_policies")


@st.cache_resource
def load_rag():
    documents = []

    if not PDF_FOLDER.exists():
        return None, None, None, None

    for pdf_path in sorted(PDF_FOLDER.glob("*.pdf")):
        reader = PdfReader(str(pdf_path))

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""

            if text.strip():
                documents.append({
                    "text": text,
                    "source": pdf_path.name,
                    "page": page_number
                })

    if not documents:
        return None, None, None, None

    chunks = []
    chunk_size = 700
    overlap = 100

    for doc in documents:
        text = doc["text"]
        start = 0

        while start < len(text):
            chunk = text[start:start + chunk_size]

            if chunk.strip():
                chunks.append({
                    "text": chunk,
                    "source": doc["source"],
                    "page": doc["page"]
                })

            start += chunk_size - overlap

    embedder = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    embeddings = embedder.encode(
        [item["text"] for item in chunks],
        normalize_embeddings=True
    )

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(
        np.array(embeddings, dtype="float32")
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "google/flan-t5-base"
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        "google/flan-t5-base"
    )

    model.eval()

    return chunks, embedder, index, (tokenizer, model)


with st.spinner("Loading HR policy documents and AI models..."):
    chunks, embedder, index, language_model = load_rag()

if chunks is None:
    st.error(
        "HR policy PDFs were not found. Please add the PDF files "
        "to the hr_policies folder in your GitHub repository."
    )
    st.stop()

st.success("HR policy assistant is ready!")


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


question = st.chat_input(
    "Ask a question about Zyro HR policies..."
)


if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    # Search with multiple query variations
    search_queries = [
        question,
        question + " annual leave entitlement days per year",
        "How many days of annual leave are allocated to employees annually?",
        "annual leave allowance yearly entitlement"
    ]

    all_scores = {}

    for search_query in search_queries:

        query_embedding = embedder.encode(
            [search_query],
            normalize_embeddings=True
        ).astype("float32")

        scores, indices = index.search(
            query_embedding, 4
        )

        for score, idx in zip(scores[0], indices[0]):

            if idx >= 0:
                all_scores[idx] = max(
                    all_scores.get(idx, 0),
                    float(score)
                )

    ranked_indices = sorted(
        all_scores,
        key=all_scores.get,
        reverse=True
    )

    if (
        not ranked_indices
        or all_scores[ranked_indices[0]] < 0.25
    ):

        answer = (
            "I can only answer questions related to the "
            "Zyro Dynamics HR policy documents. "
            "Please ask an HR-related question."
        )

        sources = []

    else:

        retrieved = [
            chunks[i] for i in ranked_indices[:6]
        ]

        context = "\n\n".join(
            item["text"] for item in retrieved
        )

        prompt = f"""
You are a helpful HR policy assistant.

Answer the employee's question directly using only the
policy context provided below.

Look carefully for the exact policy entitlement, number
of days, eligibility conditions, and applicable rules.

Do not confuse leave entitlement with leave expiry.
Do not answer with unrelated policy information.

If the exact answer is not stated in the context,
clearly say that the policy documents do not provide it.

Never invent numbers or policy details.

Policy context:
{context}

Employee question: {question}

Direct answer:
"""

        tokenizer, model = language_model

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        with torch.no_grad():

            output = model.generate(
                **inputs,
                max_new_tokens=120,
                num_beams=4,
                do_sample=False,
                repetition_penalty=1.2
            )

        answer = tokenizer.decode(
            output[0],
            skip_special_tokens=True
        )

        # Remove duplicate source citations
        sources = []
        seen_sources = set()

        for item in retrieved:

            source_key = (
                item["source"],
                item["page"]
            )

            if source_key not in seen_sources:
                sources.append(item)
                seen_sources.add(source_key)

    # Prepare final response
    response = answer

    if sources:
        response += "\n\n**Sources:**"

        for source in sources:
            response += (
                f"\n- {source['source']} "
                f"(Page {source['page']})"
            )

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })
