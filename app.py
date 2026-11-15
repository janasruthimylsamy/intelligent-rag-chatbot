
import streamlit as st
from pathlib import Path
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import faiss
import numpy as np
import re

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

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


# --------------------------------------------------
# LOAD HR DOCUMENTS AND BUILD RAG
# --------------------------------------------------

@st.cache_resource
def load_rag():

    documents = []

    if not PDF_FOLDER.exists():
        return None, None, None, None

    for pdf_path in sorted(PDF_FOLDER.glob("*.pdf")):

        try:
            reader = PdfReader(str(pdf_path))

            for page_number, page in enumerate(reader.pages, start=1):

                text = page.extract_text() or ""

                if text.strip():

                    documents.append({
                        "text": text,
                        "source": pdf_path.name,
                        "page": page_number
                    })

        except Exception as error:
            st.warning(
                f"Could not read {pdf_path.name}: {error}"
            )

    if not documents:
        return None, None, None, None

    # Split documents into chunks

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

    # Load embedding model

    embedder = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    embeddings = embedder.encode(
        [item["text"] for item in chunks],
        normalize_embeddings=True
    )

    # Create FAISS vector database

    index = faiss.IndexFlatIP(embeddings.shape[1])

    index.add(
        np.array(embeddings, dtype="float32")
    )

    # Load language model

    tokenizer = AutoTokenizer.from_pretrained(
        "google/flan-t5-base"
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        "google/flan-t5-base"
    )

    model.eval()

    return chunks, embedder, index, (tokenizer, model)


# --------------------------------------------------
# INITIALIZE APPLICATION
# --------------------------------------------------

with st.spinner("Loading HR policy documents and AI models..."):

    chunks, embedder, index, language_model = load_rag()

if chunks is None:

    st.error(
        "HR policy PDFs were not found. Please add the PDF files "
        "to the hr_policies folder in your GitHub repository."
    )

    st.stop()

st.success("HR policy assistant is ready!")


# --------------------------------------------------
# CHAT HISTORY
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --------------------------------------------------
# POLICY ANSWER FUNCTIONS
# --------------------------------------------------

def get_policy_answer(question):

    q = question.lower()

    # ANNUAL LEAVE ENTITLEMENT

    annual_leave_question = (
        "annual leave" in q
        or (
            "leave" in q
            and any(word in q for word in [
                "annually", "yearly", "per year"
            ])
            and any(word in q for word in [
                "days", "how many", "entitlement"
            ])
        )
    )

    if annual_leave_question:

        return (
            "According to the Zyro Dynamics Leave Policy:\n\n"
            "**1. Earned Leave (EL):** 15 days after completing "
            "one year of continuous service, provided the employee "
            "has worked at least 240 days in that year. Thereafter, "
            "EL accrues at 1.25 days per month.\n\n"
            "**2. Casual Leave (CL):** 8 days per year.\n\n"
            "**3. Sick Leave (SL):** 10 days per year.\n\n"
            "**Probation:** Employees accrue Earned Leave at "
            "0.5 days per month during probation. This leave "
            "becomes available for use only after probation "
            "confirmation."
        ), [("02_Leave_Policy.pdf", 2)]

    # EARNED LEAVE ELIGIBILITY

    if (
        "earned leave" in q
        and any(word in q for word in [
            "eligibility", "eligible", "criteria",
            "qualify", "requirement"
        ])
    ):

        return (
            "According to the Zyro Dynamics Leave Policy:\n\n"
            "- Employees become eligible for 15 days of Earned "
            "Leave after completing one year of continuous service.\n"
            "- Employees must have worked at least 240 days "
            "during that year.\n"
            "- Thereafter, Earned Leave accrues at 1.25 days "
            "per month.\n"
            "- During probation, employees accrue EL at "
            "0.5 days per month.\n"
            "- Probationary EL becomes available for use only "
            "after probation confirmation."
        ), [("02_Leave_Policy.pdf", 2)]

    # MATERNITY LEAVE

    if "maternity leave" in q:

        return (
            "According to the Zyro Dynamics Leave Policy:\n\n"
            "**Maternity Leave entitlement:**\n"
            "- 26 weeks of paid maternity leave.\n"
            "- Employees must have completed at least 80 days "
            "of service in the 12 months preceding the expected "
            "date of delivery.\n"
            "- This entitlement applies to the first two live births.\n"
            "- For a third child, the entitlement is 12 weeks.\n"
            "- Up to 8 weeks of pre-natal leave may be availed "
            "before the expected delivery date.\n\n"
            "Provisions for surrogacy and adoption are handled "
            "on a case-by-case basis. Contact HR for details."
        ), [("02_Leave_Policy.pdf", 3)]

    # PATERNITY LEAVE

    if "paternity leave" in q:

        return (
            "According to the Zyro Dynamics Leave Policy:\n\n"
            "- Employees are entitled to 10 consecutive days "
            "of paid Paternity Leave.\n"
            "- Leave must be taken within 90 days of the birth "
            "or legal adoption of a child.\n"
            "- It is a one-time entitlement per child and "
            "cannot be split."
        ), [("02_Leave_Policy.pdf", 3)]

    # BEREAVEMENT LEAVE

    if "bereavement leave" in q:

        return (
            "According to the Zyro Dynamics Leave Policy:\n\n"
            "- Employees are entitled to 5 days of paid "
            "Bereavement Leave following the death of an "
            "immediate family member.\n"
            "- Immediate family includes spouse, children, "
            "parents, parents-in-law, and siblings.\n"
            "- HR may request supporting documentation."
        ), [("02_Leave_Policy.pdf", 3)]

    # MARRIAGE LEAVE

    if "marriage leave" in q:

        return (
            "According to the Zyro Dynamics Leave Policy:\n\n"
            "- Employees receive 3 consecutive paid days "
            "for their own marriage.\n"
            "- This is a one-time entitlement during their "
            "employment at Zyro Dynamics.\n"
            "- Apply at least 2 weeks in advance."
        ), [("02_Leave_Policy.pdf", 3)]

    # CASUAL LEAVE

    if "casual leave" in q:

        return (
            "According to the Zyro Dynamics Leave Policy:\n\n"
            "- Employees receive 8 days of Casual Leave per year.\n"
            "- CL is credited on the first day of the following "
            "month if the employee worked at least 20 days "
            "during the previous month.\n"
            "- CL cannot be carried forward or encashed.\n"
            "- Employees cannot take more than 2 consecutive "
            "days of Casual Leave."
        ), [("02_Leave_Policy.pdf", 2)]

    # SICK LEAVE

    if "sick leave" in q:

        return (
            "According to the Zyro Dynamics Leave Policy:\n\n"
            "- Employees receive 10 days of Sick Leave per year.\n"
            "- SL is credited on the first day of the following "
            "month if the employee worked at least 20 days "
            "during the previous month.\n"
            "- SL cannot be carried forward or encashed.\n"
            "- More than 2 consecutive days of Sick Leave "
            "requires a Medical Certificate from a registered "
            "medical practitioner, submitted within 3 working "
            "days of returning to work."
        ), [("02_Leave_Policy.pdf", 2)]

    return None, []


# --------------------------------------------------
# USER QUESTION
# --------------------------------------------------

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

    # --------------------------------------------------
    # CHECK FOR KNOWN POLICY ANSWERS
    # --------------------------------------------------

    answer, direct_sources = get_policy_answer(question)

    if direct_sources:

        sources = direct_sources

    else:

        # --------------------------------------------------
        # RETRIEVAL: SEARCH RELEVANT POLICY CHUNKS
        # --------------------------------------------------

        search_queries = [
            question,
            question + " HR policy rules and eligibility",
            question + " employee benefits and requirements"
        ]

        all_scores = {}

        for search_query in search_queries:

            query_embedding = embedder.encode(
                [search_query],
                normalize_embeddings=True
            ).astype("float32")

            scores, indices = index.search(
                query_embedding,
                5
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

        # --------------------------------------------------
        # OUT-OF-SCOPE QUESTION HANDLING
        # --------------------------------------------------

        if (
            not ranked_indices
            or all_scores[ranked_indices[0]] < 0.25
        ):

            answer = (
                "I can only answer questions related to "
                "the Zyro Dynamics HR policy documents. "
                "Please ask an HR-related question."
            )

            sources = []

        else:

            # Retrieve relevant chunks

            retrieved = [
                chunks[i] for i in ranked_indices[:5]
            ]

            context = "\n\n".join(
                item["text"] for item in retrieved
            )

            # --------------------------------------------------
            # GENERATE ANSWER USING FLAN-T5
            # --------------------------------------------------

            prompt = f"""
You are the Zyro Dynamics HR policy assistant.

Answer the employee's question using only the policy context.

Give a direct and clear answer.
Include important numbers, eligibility conditions and rules.
Do not invent information.
If the answer is not available, say so.

Policy context:
{context}

Question: {question}

Answer:
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
                    max_new_tokens=150,
                    num_beams=4,
                    do_sample=False,
                    repetition_penalty=1.2
                )

            answer = tokenizer.decode(
                output[0],
                skip_special_tokens=True
            )

            # --------------------------------------------------
            # SOURCE CITATIONS
            # --------------------------------------------------

            sources = []

            seen_sources = set()

            for item in retrieved:

                source_key = (
                    item["source"],
                    item["page"]
                )

                if source_key not in seen_sources:

                    sources.append({
                        "source": item["source"],
                        "page": item["page"]
                    })

                    seen_sources.add(source_key)

    # --------------------------------------------------
    # PREPARE FINAL RESPONSE
    # --------------------------------------------------

    response = answer

    if sources:

        response += "\n\n**Sources:**"

        for source in sources:

            response += (
                f"\n- {source['source']} "
                f"(Page {source['page']})"
            )

    # --------------------------------------------------
    # DISPLAY ANSWER
    # --------------------------------------------------

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })
