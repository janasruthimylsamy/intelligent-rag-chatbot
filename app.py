
import streamlit as st
from pathlib import Path
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
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
# TEXT PROCESSING
# --------------------------------------------------

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_words(text):
    stop_words = {
        "the", "is", "are", "a", "an", "and", "or",
        "of", "to", "in", "at", "for", "what", "how",
        "many", "does", "do", "can", "i", "we", "you",
        "with", "on", "be", "it", "this", "that",
        "who", "which", "when", "where", "about"
    }

    words = set(re.findall(r"[a-zA-Z0-9]+", text.lower()))

    return words - stop_words


# --------------------------------------------------
# LOAD DOCUMENTS AND BUILD VECTOR DATABASE
# --------------------------------------------------

@st.cache_resource
def load_rag():

    documents = []

    if not PDF_FOLDER.exists():
        return None, None, None

    for pdf_path in sorted(PDF_FOLDER.glob("*.pdf")):

        try:
            reader = PdfReader(str(pdf_path))

            for page_number, page in enumerate(reader.pages, start=1):

                text = clean_text(page.extract_text() or "")

                if not text:
                    continue

                documents.append({
                    "text": text,
                    "source": pdf_path.name,
                    "page": page_number
                })

        except Exception as error:
            print(f"Error reading {pdf_path.name}: {error}")

    if not documents:
        return None, None, None

    # Create smaller chunks from each page

    chunks = []

    chunk_size = 500
    overlap = 80

    for doc in documents:

        text = doc["text"]
        start = 0

        while start < len(text):

            chunk_text = text[start:start + chunk_size]

            if chunk_text.strip():

                chunks.append({
                    "text": chunk_text,
                    "source": doc["source"],
                    "page": doc["page"]
                })

            start += chunk_size - overlap

    # Load the embedding model only once

    embedder = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    embeddings = embedder.encode(
        [chunk["text"] for chunk in chunks],
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=16
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    # Build FAISS index

    index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    index.add(embeddings)

    return chunks, embedder, index


# --------------------------------------------------
# RETRIEVAL
# --------------------------------------------------

def retrieve_documents(question, chunks, embedder, index):

    query_embedding = embedder.encode(
        [question],
        normalize_embeddings=True
    ).astype("float32")

    # Search a larger candidate pool

    candidate_count = min(15, len(chunks))

    scores, indices = index.search(
        query_embedding,
        candidate_count
    )

    question_words = get_words(question)

    ranked = []

    for similarity, idx in zip(scores[0], indices[0]):

        if idx < 0:
            continue

        chunk = chunks[idx]

        chunk_words = get_words(chunk["text"])

        if question_words:

            overlap = len(
                question_words.intersection(chunk_words)
            ) / len(question_words)

        else:
            overlap = 0

        # Combine semantic similarity and keyword matching

        final_score = (
            0.70 * float(similarity)
            + 0.30 * overlap
        )

        ranked.append({
            "chunk": chunk,
            "score": final_score
        })

    ranked.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    # Remove duplicate chunks from the same page

    selected = []
    seen = set()

    for item in ranked:

        chunk = item["chunk"]

        key = (
            chunk["source"],
            chunk["page"],
            chunk["text"][:100]
        )

        if key in seen:
            continue

        if item["score"] < 0.25:
            continue

        selected.append(item)
        seen.add(key)

        if len(selected) == 3:
            break

    return selected


# --------------------------------------------------
# SOURCE CITATIONS
# --------------------------------------------------

def format_sources(retrieved):

    sources = []
    seen = set()

    for item in retrieved:

        chunk = item["chunk"]

        key = (
            chunk["source"],
            chunk["page"]
        )

        if key not in seen:

            sources.append({
                "source": chunk["source"],
                "page": chunk["page"]
            })

            seen.add(key)

    return sources


# --------------------------------------------------
# PREDEFINED POLICY ANSWERS
# --------------------------------------------------

def get_policy_answer(question):

    q = question.lower()

    # WORK FROM HOME POLICY

    if any(term in q for term in [
        "work from home",
        "work-from-home",
        "wfh",
        "hybrid work",
        "hybrid wfh"
    ]):

        if "hybrid" in q and any(term in q for term in [
            "eligible", "eligibility", "who", "days"
        ]):

            return (
                "According to the Zyro Dynamics Work From Home Policy:\n\n"
                "**Hybrid WFH eligibility:** Employees at Level 3 (L3) "
                "and above.\n\n"
                "**Maximum allowance:** Up to 3 days per week.\n\n"
                "**Approval:** Fixed WFH days must be agreed with "
                "the reporting manager in writing."
            ), [("03_Work_From_Home_Policy.pdf", 2)]

        return (
            "According to the Zyro Dynamics Work From Home Policy, "
            "the following arrangements are specified:\n\n"
            "**1. Hybrid WFH**\n"
            "- Eligibility: L3 and above.\n"
            "- Maximum: 3 days per week.\n"
            "- Fixed days must be agreed with the reporting manager "
            "in writing.\n\n"
            "**2. Full Remote**\n"
            "- Eligibility: L5 and above, subject to case-by-case approval.\n"
            "- Maximum: 5 days per week.\n\n"
            "**3. Ad-hoc WFH**\n"
            "- Eligibility: L3 and above.\n"
            "- Maximum: 2 days per week.\n"
            "- For unplanned personal or minor health reasons.\n\n"
            "**4. Emergency WFH**\n"
            "- Available to all employees during declared emergencies, "
            "natural disasters, or health advisories.\n"
            "- Duration is as directed by HR."
        ), [("03_Work_From_Home_Policy.pdf", 2)]

    # ANNUAL LEAVE

    if "annual leave" in q or (
        "leave" in q
        and any(term in q for term in [
            "annually", "yearly", "per year"
        ])
        and any(term in q for term in [
            "days", "how many", "entitlement"
        ])
    ):

        return (
            "According to the Zyro Dynamics Leave Policy:\n\n"
            "**1. Earned Leave (EL):** 15 days after completing "
            "one year of continuous service, provided the employee "
            "has worked at least 240 days in that year. Thereafter, "
            "EL accrues at 1.25 days per month.\n\n"
            "**2. Casual Leave (CL):** 8 days per year.\n\n"
            "**3. Sick Leave (SL):** 10 days per year.\n\n"
            "**Probation:** Employees accrue Earned Leave at "
            "0.5 days per month during probation. This leave becomes "
            "available for use only after probation confirmation."
        ), [("02_Leave_Policy.pdf", 2)]

    # EARNED LEAVE ELIGIBILITY

    if (
        "earned leave" in q
        and any(term in q for term in [
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
            "- 26 weeks of paid maternity leave.\n"
            "- At least 80 days of service in the 12 months "
            "preceding the expected delivery date is required.\n"
            "- Applies to the first two live births.\n"
            "- For a third child, the entitlement is 12 weeks.\n"
            "- Up to 8 weeks of pre-natal leave may be taken."
        ), [("02_Leave_Policy.pdf", 3)]

    # PATERNITY LEAVE

    if "paternity leave" in q:

        return (
            "According to the Zyro Dynamics Leave Policy:\n\n"
            "- 10 consecutive days of paid Paternity Leave.\n"
            "- Must be taken within 90 days of birth or legal adoption.\n"
            "- One-time entitlement per child.\n"
            "- Cannot be split."
        ), [("02_Leave_Policy.pdf", 3)]

    # BEREAVEMENT LEAVE

    if "bereavement leave" in q:

        return (
            "According to the Zyro Dynamics Leave Policy:\n\n"
            "- 5 days of paid Bereavement Leave.\n"
            "- Covers immediate family members: spouse, children, "
            "parents, parents-in-law, and siblings."
        ), [("02_Leave_Policy.pdf", 3)]

    # MARRIAGE LEAVE

    if "marriage leave" in q:

        return (
            "According to the Zyro Dynamics Leave Policy:\n\n"
            "- 3 consecutive paid days for an employee's own marriage.\n"
            "- Available once during employment.\n"
            "- Apply at least 2 weeks in advance."
        ), [("02_Leave_Policy.pdf", 3)]

    # CASUAL LEAVE

    if "casual leave" in q:

        return (
            "According to the Zyro Dynamics Leave Policy:\n\n"
            "- 8 days of Casual Leave per year.\n"
            "- Credited on the first day of the following month "
            "if the employee worked at least 20 days in the "
            "previous month.\n"
            "- Cannot be carried forward or encashed.\n"
            "- Maximum of 2 consecutive days."
        ), [("02_Leave_Policy.pdf", 2)]

    # SICK LEAVE

    if "sick leave" in q:

        return (
            "According to the Zyro Dynamics Leave Policy:\n\n"
            "- 10 days of Sick Leave per year.\n"
            "- Credited on the first day of the following month "
            "if the employee worked at least 20 days in the "
            "previous month.\n"
            "- Cannot be carried forward or encashed.\n"
            "- More than 2 consecutive days requires a Medical "
            "Certificate from a registered medical practitioner."
        ), [("02_Leave_Policy.pdf", 2)]

    return None, []


# --------------------------------------------------
# INITIALIZE APP
# --------------------------------------------------

with st.spinner("Loading HR policy documents..."):

    chunks, embedder, index = load_rag()

if chunks is None:

    st.error(
        "HR policy PDFs were not found. Please check the "
        "hr_policies folder in your GitHub repository."
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
# CHAT INPUT
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

    with st.chat_message("assistant"):

        with st.spinner("Searching HR policy documents..."):

            # Check predefined answers first

            answer, direct_sources = get_policy_answer(question)

            if direct_sources:

                sources = [
                    {
                        "source": source,
                        "page": page
                    }
                    for source, page in direct_sources
                ]

            else:

                # Search the PDF knowledge base

                retrieved = retrieve_documents(
                    question,
                    chunks,
                    embedder,
                    index
                )

                if not retrieved:

                    answer = (
                        "I could not find a sufficiently relevant "
                        "answer in the Zyro Dynamics HR policy documents. "
                        "Please try rephrasing your question or contact HR."
                    )

                    sources = []

                else:

                    # Display relevant retrieved policy text

                    answer = (
                        "Here is the relevant information I found "
                        "in the Zyro Dynamics HR policy documents:\n\n"
                    )

                    for item in retrieved:

                        answer += (
                            "- "
                            + item["chunk"]["text"]
                            + "\n\n"
                        )

                    sources = format_sources(retrieved)

            # Display citations

            response = answer

            if sources:

                response += "\n\n**Sources:**"

                for source in sources:

                    response += (
                        f"\n- {source['source']} "
                        f"(Page {source['page']})"
                    )

            st.markdown(response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })
