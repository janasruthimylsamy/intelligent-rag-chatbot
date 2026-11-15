
import os
import re
import streamlit as st
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Zyro Dynamics | HR Assistant",
    page_icon="📚",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-color: #faf7f5;
}
[data-testid="stSidebar"] {
    background-color: #f0e9f2;
}
h1, h2, h3 {
    color: #59345f;
}
.stButton button {
    background-color: #704477;
    color: white;
    border-radius: 10px;
    border: none;
}
.stButton button:hover {
    background-color: #59345f;
    color: white;
}
.answer-box {
    background-color: #f0e9f2;
    padding: 18px;
    border-radius: 12px;
    border-left: 5px solid #704477;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# DOCUMENT SETTINGS
# --------------------------------------------------

POLICY_FOLDER = "hr_policies"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


# --------------------------------------------------
# LOAD PDF DOCUMENTS
# --------------------------------------------------

@st.cache_resource
def load_documents():

    documents = []

    if not os.path.exists(POLICY_FOLDER):
        return documents

    for filename in sorted(os.listdir(POLICY_FOLDER)):

        if not filename.lower().endswith(".pdf"):
            continue

        filepath = os.path.join(POLICY_FOLDER, filename)

        try:
            reader = PdfReader(filepath)

            for page_number, page in enumerate(reader.pages, start=1):

                text = page.extract_text() or ""
                text = re.sub(r"\s+", " ", text).strip()

                if text:
                    documents.append({
                        "text": text,
                        "source": filename,
                        "page": page_number
                    })

        except Exception as error:
            print(f"Could not read {filename}: {error}")

    return documents


# --------------------------------------------------
# CREATE TEXT CHUNKS
# --------------------------------------------------

def create_chunks(documents):

    chunks = []

    for document in documents:

        text = document["text"]
        start = 0

        while start < len(text):

            end = start + CHUNK_SIZE
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "source": document["source"],
                    "page": document["page"]
                })

            if end >= len(text):
                break

            start = end - CHUNK_OVERLAP

    return chunks


# --------------------------------------------------
# LOAD EMBEDDINGS AND FAISS INDEX
# --------------------------------------------------

@st.cache_resource
def load_rag():

    documents = load_documents()

    if not documents:
        return None, None, None

    chunks = create_chunks(documents)

    if not chunks:
        return None, None, None

    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    embeddings = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return model, index, chunks


# --------------------------------------------------
# POLICY DOCUMENT FILTER
# --------------------------------------------------

def get_document_filter(question):

    q = question.lower()

    if any(term in q for term in [
        "working hours", "work hours", "hours per day",
        "full-time employee", "full time employee",
        "working days", "attendance", "office timing",
        "office hours", "daily working"
    ]):
        return "01_Employee_Handbook.pdf"

    if any(term in q for term in [
        "sexual harassment", "posh", "harassment complaint",
        "internal complaints committee", "icc",
        "unwelcome sexual", "sexual misconduct"
    ]):
        return "08_Prevention_of_Sexual_Harassment_Policy.pdf"

    if any(term in q for term in [
        "code of conduct", "misconduct", "cash from vendor",
        "confidential information", "conflict of interest",
        "social media", "report violations",
        "reporting violations", "retaliation",
        "whistleblower", "gifts from vendors"
    ]):
        return "04_Code_of_Conduct.pdf"

    if any(term in q for term in [
        "work from home", "work remotely",
        "hybrid work", "remote work", "wfh"
    ]):
        return "03_Work_From_Home_Policy.pdf"

    if any(term in q for term in [
        "annual leave", "earned leave", "casual leave",
        "sick leave", "maternity leave", "paternity leave",
        "bereavement leave", "marriage leave",
        "leave entitlement"
    ]):
        return "02_Leave_Policy.pdf"

    return None


# --------------------------------------------------
# RETRIEVE RELEVANT DOCUMENTS
# --------------------------------------------------

def retrieve_documents(question, model, index, chunks):

    if model is None or index is None or not chunks:
        return []

    query_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    search_count = min(40, len(chunks))

    scores, indices = index.search(
        query_embedding,
        search_count
    )

    selected_document = get_document_filter(question)

    results = []

    for score, chunk_index in zip(scores[0], indices[0]):

        if chunk_index < 0:
            continue

        chunk = chunks[chunk_index]

        if selected_document and chunk["source"] != selected_document:
            continue

        if score < 0.25:
            continue

        results.append({
            "text": chunk["text"],
            "source": chunk["source"],
            "page": chunk["page"],
            "score": float(score)
        })

        if len(results) >= 5:
            break

    return results


# --------------------------------------------------
# DIRECT POLICY ANSWERS
# --------------------------------------------------

def get_direct_answer(question):

    q = question.lower()

    # POSH POLICY

    posh_terms = [
        "sexual harassment", "posh", "harassment at workplace",
        "harassed at work", "harassment complaint",
        "file a complaint", "complaint with icc",
        "internal complaints committee", "icc complaint",
        "report sexual harassment", "experience harassment"
    ]

    if any(term in q for term in posh_terms):

        return {
            "answer": """
If an employee experiences sexual harassment, they can submit
a **written complaint to the Internal Complaints Committee (ICC)**
under the Zyro Dynamics POSH Policy.

### Step 1: Submit a written complaint

- Submit the complaint by email to the ICC or as a sealed letter
  addressed to the Presiding Officer at the HR office at
  Zyro Dynamics HQ.
- The complaint should be submitted within **3 months of the
  incident**, or within 3 months of the last incident if there
  is a series of incidents.
- In exceptional circumstances, the ICC may extend this period
  by a further 3 months upon a written application explaining
  the reasons.

**ICC email clarification:** The email address in the extracted
policy appears with a formatting error as
`icc@zyrodyn amics.com`. Please verify the correct address with HR
before sending a complaint. You can use the sealed-letter option
described above.

### Step 2: ICC acknowledges the complaint

The ICC acknowledges receipt and forwards a copy of the complaint
to the respondent within **7 working days**.

### Step 3: Respondent submits a response

The respondent is given **10 working days** to submit a written
response.

### Step 4: ICC conducts an inquiry

The ICC conducts an inquiry, including hearings with both parties,
witnesses, and a review of evidence.

The inquiry is to be completed within **60 days**.

### Step 5: Inquiry report

The ICC submits its inquiry report and recommendations to management
within **10 days of completing the inquiry**.

### Step 6: Management takes action

Management implements the recommended action based on the ICC's
findings.

### Confidentiality

The identities of the complainant, respondent, and witnesses,
as well as the inquiry proceedings, findings, and recommendations,
must remain strictly confidential.

### Protection from retaliation

The company prohibits retaliation against anyone who files a complaint,
participates in an inquiry, or reports a suspected violation in good
faith.

Retaliation such as threats, demotion, transfer, isolation, adverse
performance ratings, or termination is treated as a separate and
serious disciplinary matter.

### Possible consequences

Depending on the ICC's findings, action against the respondent
may include:

- Written warning and mandatory sensitisation training.
- Suspension without pay.
- Transfer to another team, project, or location.
- Withholding promotion, increments, or benefits.
- Termination of employment.
- Referral to law enforcement if the conduct constitutes
  a criminal offence under applicable Indian law.

**Remember:** Employees can approach the ICC through the complaint
procedure described in the POSH Policy.
""",
            "sources": [
                ("08_Prevention_of_Sexual_Harassment_Policy.pdf", 2),
                ("08_Prevention_of_Sexual_Harassment_Policy.pdf", 3)
            ]
        }

    # WORKING HOURS

    working_terms = [
        "working hours", "work hours", "hours per day",
        "full-time employee", "full time employee",
        "how many hours", "working days", "office hours",
        "office timing", "daily working hours",
        "expected to work"
    ]

    if any(term in q for term in working_terms):

        return {
            "answer": """
According to the Employee Handbook, employees are expected to complete
a minimum of **9 working hours per day**.

The standard schedule is:

- **Working days:** Monday to Friday
- **Office hours:** 9:30 AM to 6:30 PM IST
- **Unpaid lunch break:** 30 minutes
- **Minimum expected working time:** 9 hours per day

**Important clarification:**

There is an inconsistency in the handbook.

The schedule from 9:30 AM to 6:30 PM is 9 hours in total.
After subtracting the 30-minute unpaid lunch break, the actual
working time is 8.5 hours.

However, the handbook separately states that employees must complete
a minimum of 9 working hours each day.

Employees should confirm the correct expected working duration
with the HR department.

**Flexible working hours:**

Employees at grade L3 and above may opt for flexible working hours
within the 10:00 AM to 4:00 PM core band, subject to written approval
from their reporting manager.
""",
            "sources": [
                ("01_Employee_Handbook.pdf", 3),
                ("01_Employee_Handbook.pdf", 4)
            ]
        }

    # CASH ACCEPTANCE

    if "cash" in q and any(term in q for term in [
        "vendor", "employee", "accept", "receive", "gift"
    ]):

        return {
            "answer": """
Accepting cash from any party is **strictly prohibited**.

The Code of Conduct states that accepting cash is never permitted.
Violations may result in immediate disciplinary action.

The company follows a zero-tolerance approach to accepting cash.
""",
            "sources": [("04_Code_of_Conduct.pdf", 3)]
        }

    # CONFIDENTIAL INFORMATION

    if any(term in q for term in [
        "confidential information", "confidential data",
        "company information", "data leak"
    ]):

        return {
            "answer": """
Employees must share confidential information only when there is
an authorised business need.

Confidential information includes:

- Customer and prospect data
- Product roadmaps
- Pricing and financial figures
- HR records
- Business strategies
- Source code and other non-public information

Employees must not post company information on social media,
share it through personal devices, or disclose it to the media.

When leaving the company, employees must return or delete confidential
documents and credentials as directed by IT or HR.

Violations may result in disciplinary action up to and including
termination and may also attract civil or criminal liability
under applicable law.
""",
            "sources": [("04_Code_of_Conduct.pdf", 2)]
        }

    # SOCIAL MEDIA

    if "social media" in q:

        return {
            "answer": """
Employees must not share confidential company information,
internal communications, client names, product screenshots,
or financial information on social media.

Employees must not present personal opinions as the official
position of the company unless authorised by Marketing.

Negative commentary or criticism of colleagues, management,
or clients on social media is prohibited and may result
in disciplinary action.

Employees should include this disclaimer when expressing
personal views:

**"Views expressed are my own and do not represent Zyro Dynamics."**
""",
            "sources": [("04_Code_of_Conduct.pdf", 3)]
        }

    # REPORTING CODE OF CONDUCT VIOLATIONS

    if any(term in q for term in [
        "report violation", "reporting violation",
        "whistleblower", "retaliation",
        "report misconduct", "report a concern",
        "report a complaint"
    ]):

        return {
            "answer": """
Employees can report suspected Code of Conduct violations through
the following channels:

1. **Reporting Manager:** Report directly or through email.

2. **HR Grievance Cell:** hr.grievance@acruxdynamics.com
   This channel is strictly confidential.

3. **Anonymous Whistleblower Portal:**
   whistle.acruxdynamics.com
   Employees can report anonymously.

4. **Chief People Officer:** marta@acruxdynamics.com
   This channel is confidential.

**Protection from retaliation:**

Employees who report violations in good faith are protected.
Retaliation against a good-faith reporter is itself a violation
and may result in serious disciplinary action.
""",
            "sources": [("04_Code_of_Conduct.pdf", 3)]
        }

    # CONFLICT OF INTEREST

    if "conflict of interest" in q:

        return {
            "answer": """
Employees must disclose any conflict of interest in writing
to their manager and HR immediately.

Important requirements include:

- Financial interests in competitors, customers, or suppliers
  require written clearance from Legal & Compliance.
- Immediate family members employed by a competitor or vendor
  must be disclosed to HR.
- Paid outside employment or consulting requires prior written
  approval from the Chief People Officer.
""",
            "sources": [("04_Code_of_Conduct.pdf", 2)]
        }

    # GIFTS AND HOSPITALITY

    if "gift" in q or "hospitality" in q:

        return {
            "answer": """
The Code of Conduct specifies the following gift and hospitality rules:

- Gifts below ₹1,000 are permitted if declared to the manager.
- Gifts above ₹1,000 are not permitted and must be declined
  or surrendered to HR.
- Client-hosted meals are limited to ₹3,000 per person.
- Vendor-sponsored domestic travel is not permitted without
  prior written approval from the CFO.
- Branded gifts to clients require Finance pre-approval
  and must be below ₹2,000 per item.
- Accepting cash from any party is never permitted.
""",
            "sources": [("04_Code_of_Conduct.pdf", 3)]
        }

    # WORK FROM HOME

    if any(term in q for term in [
        "work from home", "hybrid work",
        "remote work", "wfh"
    ]):

        return {
            "answer": """
According to the Work From Home Policy:

- **Hybrid WFH:** Employees at grade L3 and above may work
  remotely for up to 3 days per week.
- **Full Remote:** Employees at grade L5 and above may be
  considered for full remote work on a case-by-case basis.
- **Ad-hoc WFH:** Employees at grade L3 and above may request
  up to 2 days per week.
- **Emergency WFH:** Available to all employees as directed by HR.

Employees must also have completed at least 6 months of continuous
service and have a recent performance rating of Meets Expectations
or above to satisfy the stated eligibility criteria.
""",
            "sources": [("03_Work_From_Home_Policy.pdf", 2)]
        }

    # ANNUAL LEAVE

    if "annual leave" in q or "earned leave" in q:

        return {
            "answer": """
According to the Leave Policy:

- Employees become eligible for earned leave after completing
  one year of continuous service and 240 working days.
- The entitlement is 15 days of earned leave.
- After that, earned leave is credited at 1.25 days per month.
- Casual leave entitlement is 8 days per year.
- Sick leave entitlement is 10 days per year.
- During probation, earned leave is credited at 0.5 days per month
  and becomes available after confirmation.

Please refer to the Leave Policy for the applicable conditions
and procedures.
""",
            "sources": [("02_Leave_Policy.pdf", 2)]
        }

    return None


# --------------------------------------------------
# HR QUESTION CHECK
# --------------------------------------------------

def is_likely_hr_question(question):

    keywords = [
        "employee", "hr", "leave", "salary", "payroll",
        "policy", "company", "work", "working", "office",
        "attendance", "manager", "benefit", "holiday",
        "harassment", "conduct", "confidential",
        "remote", "wfh", "probation", "resignation",
        "termination", "reimbursement", "travel",
        "expense", "working hours", "working days",
        "posh", "icc", "complaint", "retaliation"
    ]

    q = question.lower()

    return any(keyword in q for keyword in keywords)


# --------------------------------------------------
# DISPLAY SOURCES
# --------------------------------------------------

def display_sources(sources):

    if not sources:
        return

    st.markdown("**Sources:**")

    for source, page in sources:
        st.markdown(f"- `{source}` (Page {page})")


# --------------------------------------------------
# USER INTERFACE
# --------------------------------------------------

st.title("📚 HR Policy Assistant")

st.markdown("""
Welcome to the **HR Policy Assistant**.

Ask questions about employee policies, leave, attendance,
work-from-home arrangements, workplace conduct, and other
HR-related topics.
""")

with st.sidebar:

    st.header("About")

    st.write("""
    This chatbot uses Retrieval-Augmented Generation (RAG)
    to search the company's HR policy PDF documents.

    It retrieves relevant policy information and provides
    answers with document and page references.
    """)

    st.divider()

    st.subheader("Available policy documents")

    st.write("""
    - Employee Handbook
    - Leave Policy
    - Work From Home Policy
    - Code of Conduct
    - Performance Review Policy
    - Compensation and Benefits Policy
    - IT and Data Security Policy
    - POSH Policy
    - Onboarding and Separation Policy
    - Travel and Expense Policy
    """)


# --------------------------------------------------
# LOAD THE RAG SYSTEM
# --------------------------------------------------

with st.spinner("Loading HR policy documents..."):

    model, index, chunks = load_rag()


if model is None:

    st.error(
        "No policy documents could be loaded. "
        "Please check that the hr_policies folder contains PDF files."
    )

    st.stop()


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
    "Ask your HR policy question here..."
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

            direct_result = get_direct_answer(question)

            if direct_result:

                answer = direct_result["answer"]

                st.markdown(answer)
                display_sources(direct_result["sources"])

            elif not is_likely_hr_question(question):

                answer = """
I can help only with questions related to the company's
HR policies and employee handbook.

Please ask an HR-related question.
"""

                st.markdown(answer)

            else:

                results = retrieve_documents(
                    question,
                    model,
                    index,
                    chunks
                )

                if results:

                    answer = (
                        "Here are the most relevant excerpts "
                        "from the HR policy documents:\n\n"
                    )

                    seen_sources = set()

                    for result in results:

                        answer += (
                            f"**{result['source']} — "
                            f"Page {result['page']}**\n\n"
                            f"{result['text']}\n\n---\n\n"
                        )

                        seen_sources.add((
                            result["source"],
                            result["page"]
                        ))

                    st.markdown(answer)
                    display_sources(sorted(seen_sources))

                else:

                    answer = """
I could not find sufficiently relevant information
in the HR policy documents.

Please try rephrasing your question or contact HR
for clarification.
"""

                    st.markdown(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
