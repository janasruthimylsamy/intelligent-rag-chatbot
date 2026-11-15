
import streamlit as st
from pathlib import Path
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re

# ==================================================
# PAGE CONFIGURATION
# ==================================================

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

[data-testid="stChatInput"] {
    border-color: #684078;
}
</style>
""", unsafe_allow_html=True)

st.title("📚 Zyro Dynamics HR Assistant")

st.caption(
    "Your intelligent assistant for HR policies "
    "and workplace questions."
)

PDF_FOLDER = Path("hr_policies")


# ==================================================
# TEXT CLEANING
# ==================================================

def clean_text(text):

    if not text:
        return ""

    text = text.replace("\x00", " ")
    text = text.replace("\ufeff", " ")
    text = text.replace("\u0007", " ")
    text = text.replace("\ufffd", " ")

    text = re.sub(r"[^\S\n]+", " ", text)

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    return "\n".join(lines)


def get_words(text):

    stop_words = {
        "the", "is", "are", "a", "an", "and", "or",
        "of", "to", "in", "at", "for", "what", "how",
        "many", "does", "do", "can", "i", "we", "you",
        "with", "on", "be", "it", "this", "that",
        "who", "which", "when", "where", "about",
        "under", "from", "by", "as", "their", "they",
        "them", "has", "have", "had", "was", "were",
        "being", "been", "will", "would", "should",
        "could", "employee", "employees", "zyro",
        "dynamics", "policy", "policies", "company",
        "main", "types", "regarding", "responsibilities",
        "actions", "considered", "following"
    }

    words = set(
        re.findall(r"[a-zA-Z0-9]+", text.lower())
    )

    return words - stop_words


# ==================================================
# DOCUMENT EXTRACTION
# ==================================================

@st.cache_resource
def load_documents():

    documents = []

    if not PDF_FOLDER.exists():
        return []

    for pdf_path in sorted(PDF_FOLDER.glob("*.pdf")):

        try:

            reader = PdfReader(str(pdf_path))

            for page_number, page in enumerate(
                reader.pages,
                start=1
            ):

                text = clean_text(
                    page.extract_text() or ""
                )

                if text.strip():

                    documents.append({
                        "text": text,
                        "source": pdf_path.name,
                        "page": page_number
                    })

        except Exception as error:

            print(
                f"Error reading {pdf_path.name}: {error}"
            )

    return documents


# ==================================================
# PARAGRAPH-AWARE CHUNKING
# ==================================================

def create_chunks(documents):

    chunks = []

    chunk_size = 1000
    overlap = 150

    for doc in documents:

        text = doc["text"]

        paragraphs = re.split(
            r"\n+",
            text
        )

        current_chunk = ""

        for paragraph in paragraphs:

            paragraph = paragraph.strip()

            if not paragraph:
                continue

            # Preserve complete paragraphs where possible.

            if len(paragraph) > chunk_size:

                if current_chunk:

                    chunks.append({
                        "text": current_chunk.strip(),
                        "source": doc["source"],
                        "page": doc["page"]
                    })

                    current_chunk = ""

                start = 0

                while start < len(paragraph):

                    piece = paragraph[
                        start:start + chunk_size
                    ]

                    if piece.strip():

                        chunks.append({
                            "text": piece.strip(),
                            "source": doc["source"],
                            "page": doc["page"]
                        })

                    start += chunk_size - overlap

                continue

            if (
                len(current_chunk) + len(paragraph)
                > chunk_size
                and current_chunk
            ):

                chunks.append({
                    "text": current_chunk.strip(),
                    "source": doc["source"],
                    "page": doc["page"]
                })

                previous_text = current_chunk[-overlap:]

                current_chunk = (
                    previous_text + " " + paragraph
                )

            else:

                current_chunk += (
                    "\n" + paragraph
                    if current_chunk
                    else paragraph
                )

        if current_chunk.strip():

            chunks.append({
                "text": current_chunk.strip(),
                "source": doc["source"],
                "page": doc["page"]
            })

    return chunks


# ==================================================
# BUILD VECTOR DATABASE
# ==================================================

@st.cache_resource
def load_rag():

    documents = load_documents()

    if not documents:
        return None, None, None

    chunks = create_chunks(documents)

    if not chunks:
        return None, None, None

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

    index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    index.add(embeddings)

    return chunks, embedder, index


# ==================================================
# DOCUMENT ROUTING
# ==================================================

def get_document_filter(question):

    q = question.lower()

    # Working hours and attendance

    working_terms = [
        "working hours",
        "work hours",
        "working time",
        "hours per day",
        "hours each day",
        "hours a day",
        "full-time hours",
        "full time hours",
        "office hours",
        "office timing",
        "office timings",
        "work schedule",
        "working schedule",
        "weekly working days",
        "working days",
        "attendance",
        "punctuality",
        "late coming",
        "late arrival",
        "flexible working hours"
    ]

    if any(term in q for term in working_terms):

        return "01_Employee_Handbook.pdf"

    # Code of Conduct

    code_terms = [
        "code of conduct",
        "conduct policy",
        "conduct violations",
        "conduct misconduct",
        "ethical behaviour",
        "ethical behavior",
        "social media",
        "social network",
        "conflict of interest",
        "conflicts of interest",
        "whistleblower",
        "confidentiality",
        "confidential company information",
        "report violations",
        "reporting violations",
        "retaliation"
    ]

    if any(term in q for term in code_terms):

        return "04_Code_of_Conduct.pdf"

    # Work From Home

    if any(term in q for term in [
        "work from home",
        "work-from-home",
        "wfh",
        "hybrid work",
        "remote work"
    ]):

        return "03_Work_From_Home_Policy.pdf"

    # Leave

    if any(term in q for term in [
        "annual leave",
        "earned leave",
        "casual leave",
        "sick leave",
        "maternity leave",
        "paternity leave",
        "bereavement leave",
        "marriage leave"
    ]):

        return "02_Leave_Policy.pdf"

    # POSH

    if any(term in q for term in [
        "sexual harassment",
        "posh",
        "harassment complaint"
    ]):

        return "08_Prevention_of_Sexual_Harassment_Policy.pdf"

    # Other employee handbook questions

    handbook_terms = [
        "employee handbook",
        "employment category",
        "employee grade",
        "grade structure",
        "probation",
        "company mission",
        "company vision",
        "office location",
        "hr contact",
        "helpdesk"
    ]

    if any(term in q for term in handbook_terms):

        return "01_Employee_Handbook.pdf"

    return None


# ==================================================
# QUERY EXPANSION
# ==================================================

def get_expanded_terms(question):

    q = question.lower()

    expanded = set()

    if any(term in q for term in [
        "misconduct",
        "unethical",
        "prohibited",
        "violations",
        "wrongdoing"
    ]):

        expanded.update({
            "fraud", "bribery", "corruption",
            "dishonesty", "harassment",
            "discrimination", "retaliation",
            "confidentiality", "conflict",
            "interest", "theft", "falsification",
            "misuse", "integrity"
        })

    if any(term in q for term in [
        "confidentiality",
        "confidential",
        "company information"
    ]):

        expanded.update({
            "data", "disclosure", "unauthorised",
            "unauthorized", "privacy",
            "credentials", "source code"
        })

    if any(term in q for term in [
        "conflict of interest",
        "conflicts of interest"
    ]):

        expanded.update({
            "employment", "consulting",
            "approval", "personal", "financial"
        })

    if any(term in q for term in [
        "working hours",
        "work hours",
        "working time",
        "full-time hours",
        "full time hours"
    ]):

        expanded.update({
            "nine", "9:30", "6:30",
            "lunch", "unpaid", "monday",
            "friday", "attendance",
            "daily", "hours"
        })

    return expanded


# ==================================================
# DOCUMENT RETRIEVAL
# ==================================================

def retrieve_documents(
    question,
    chunks,
    embedder,
    index
):

    query_embedding = embedder.encode(
        [question],
        normalize_embeddings=True
    ).astype("float32")

    candidate_count = min(
        40,
        len(chunks)
    )

    scores, indices = index.search(
        query_embedding,
        candidate_count
    )

    question_words = get_words(question)

    expanded_terms = get_expanded_terms(
        question
    )

    document_filter = get_document_filter(
        question
    )

    ranked = []

    for similarity, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx < 0:
            continue

        chunk = chunks[idx]

        if (
            document_filter
            and chunk["source"] != document_filter
        ):
            continue

        chunk_words = get_words(
            chunk["text"]
        )

        if question_words:

            overlap = (
                len(
                    question_words.intersection(
                        chunk_words
                    )
                )
                / len(question_words)
            )

        else:

            overlap = 0

        if expanded_terms:

            expansion_match = (
                len(
                    expanded_terms.intersection(
                        chunk_words
                    )
                )
                / len(expanded_terms)
            )

        else:

            expansion_match = 0

        final_score = (
            0.65 * float(similarity)
            + 0.25 * overlap
            + 0.10 * expansion_match
        )

        ranked.append({
            "chunk": chunk,
            "score": final_score,
            "similarity": float(similarity)
        })

    ranked.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    selected = []
    seen_text = set()
    page_counts = {}

    for item in ranked:

        chunk = item["chunk"]

        text_key = re.sub(
            r"\s+",
            " ",
            chunk["text"].lower()
        )

        if text_key in seen_text:
            continue

        if item["score"] < 0.28:
            continue

        page_key = (
            chunk["source"],
            chunk["page"]
        )

        if page_counts.get(page_key, 0) >= 2:
            continue

        selected.append(item)

        seen_text.add(text_key)

        page_counts[page_key] = (
            page_counts.get(page_key, 0) + 1
        )

        if len(selected) == 5:
            break

    return selected


# ==================================================
# SOURCE CITATIONS
# ==================================================

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


# ==================================================
# PREDEFINED POLICY ANSWERS
# ==================================================

def get_policy_answer(question):

    q = question.lower()

    # ----------------------------------------------
    # WORKING HOURS AND ATTENDANCE
    # ----------------------------------------------

    working_terms = [
        "working hours",
        "work hours",
        "working time",
        "hours per day",
        "hours a day",
        "hours each day",
        "full-time hours",
        "full time hours",
        "standard working hours",
        "office hours",
        "office timings",
        "working days",
        "weekly working days"
    ]

    if any(term in q for term in working_terms):

        return (
            "According to the Employee Handbook "
            "(Working Hours and Attendance, Page 3):\n\n"

            "**Standard working schedule:**\n"
            "- Working days: Monday to Friday (5 days per week).\n"
            "- Office timings: 9:30 AM to 6:30 PM IST.\n"
            "- Lunch break: 30 minutes, unpaid.\n"
            "- The handbook separately states that employees "
            "must complete a minimum of 9 working hours each day.\n\n"

            "**Important clarification:** The stated office "
            "timings span 9 hours, including the 30-minute "
            "unpaid lunch break. This gives 8.5 hours excluding "
            "lunch, which does not match the separate 9-hour "
            "minimum requirement stated in the handbook. "
            "Please confirm this discrepancy with HR.\n\n"

            "**Flexible working hours:** Employees at grade "
            "L3 and above may opt for flexible working hours "
            "within the core band of 10:00 AM to 4:00 PM IST, "
            "subject to written approval from their reporting manager."
        ), [
            ("01_Employee_Handbook.pdf", 3)
        ]

    # ----------------------------------------------
    # CODE OF CONDUCT — ACCEPTING CASH
    # ----------------------------------------------

    if (
        "cash" in q
        and any(term in q for term in [
            "accept", "accepting", "receive",
            "receiving", "vendor", "party", "payment"
        ])
    ):

        return (
            "According to the Zyro Dynamics Code of Conduct, "
            "accepting cash from any party is strictly prohibited.\n\n"
            "**Consequences:**\n"
            "- Accepting cash is never permitted.\n"
            "- Immediate disciplinary action applies.\n"
            "- The company follows a zero-tolerance rule."
        ), [("04_Code_of_Conduct.pdf", 3)]

    # ----------------------------------------------
    # CODE OF CONDUCT — CONFIDENTIALITY
    # ----------------------------------------------

    if (
        any(term in q for term in [
            "confidential", "confidentiality"
        ])
        and any(term in q for term in [
            "share", "sharing", "shares",
            "disclose", "disclosing", "post",
            "posting", "leak", "leaking",
            "send", "sending", "company information",
            "company data"
        ])
        and not any(term in q for term in [
            "sexual harassment", "posh"
        ])
    ):

        return (
            "According to the Zyro Dynamics Code of Conduct:\n\n"
            "- Confidential information must only be shared with "
            "people who have an authorised business need.\n"
            "- Company information must not be posted on social media, "
            "shared on personal devices, or disclosed to media.\n"
            "- On leaving the company, confidential documents and "
            "credentials must be returned or deleted as directed "
            "by IT and HR.\n"
            "- Violations may result in disciplinary action up to "
            "and including termination. Civil or criminal liability "
            "may also apply under applicable law."
        ), [("04_Code_of_Conduct.pdf", 2)]

    # ----------------------------------------------
    # CODE OF CONDUCT — SOCIAL MEDIA
    # ----------------------------------------------

    if (
        any(term in q for term in [
            "social media", "social network",
            "online post", "posting online"
        ])
        and not any(term in q for term in [
            "sexual harassment", "posh"
        ])
    ):

        return (
            "According to the Zyro Dynamics Code of Conduct:\n\n"
            "- Do not share confidential company information, "
            "internal communications, client names, product "
            "screenshots, or financial data on social media.\n"
            "- Do not present personal statements as the official "
            "position of Zyro Dynamics unless authorised.\n"
            "- Negative commentary or criticism of colleagues, "
            "management, or clients on social media is prohibited "
            "and may attract disciplinary action.\n"
            "- Include this disclaimer when sharing professional "
            "opinions publicly: 'Views expressed are my own and do "
            "not represent Zyro Dynamics.'"
        ), [("04_Code_of_Conduct.pdf", 3)]

    # ----------------------------------------------
    # CODE OF CONDUCT — REPORTING VIOLATIONS
    # ----------------------------------------------

    if (
        any(term in q for term in [
            "report", "reporting", "reported",
            "whistleblower", "whistle-blower",
            "retaliation", "retaliate"
        ])
        and any(term in q for term in [
            "violation", "violations", "misconduct",
            "concern", "concerns", "complaint",
            "complaints", "report", "reporting",
            "retaliation", "retaliate", "whistleblower",
            "whistle-blower", "protected", "protection"
        ])
    ):

        return (
            "According to the Zyro Dynamics Code of Conduct, "
            "employees can report suspected violations through:\n\n"
            "1. **Reporting Manager:** Direct communication or email.\n\n"
            "2. **HR Grievance Cell:** "
            "hr.grievance@acruxdynamics.com "
            "(Strictly confidential).\n\n"
            "3. **Anonymous Whistleblower Portal:** "
            "whistle.acruxdynamics.com (Fully anonymous).\n\n"
            "4. **Chief People Officer:** "
            "marta@acruxdynamics.com (Confidential).\n\n"
            "**Protection:** Employees reporting genuine concerns "
            "in good faith are protected from retaliation. "
            "Retaliation itself is a Code of Conduct violation "
            "and a serious disciplinary matter."
        ), [("04_Code_of_Conduct.pdf", 3)]

    # ----------------------------------------------
    # CODE OF CONDUCT — DISCIPLINARY ACTIONS
    # ----------------------------------------------

    if (
        "code of conduct" in q
        and any(term in q for term in [
            "disciplinary", "punishment", "penalty",
            "misconduct", "violation", "violations",
            "unethical", "wrongdoing"
        ])
    ):

        return (
            "According to the Zyro Dynamics Code of Conduct:\n\n"
            "- Accepting cash is prohibited and results in "
            "immediate disciplinary action.\n"
            "- Confidentiality violations may result in action "
            "up to and including termination.\n"
            "- Social media misconduct may attract disciplinary action.\n"
            "- Retaliation against good-faith reporters is a serious "
            "disciplinary matter.\n"
            "- Habitual tardiness without a valid reason is "
            "a disciplinary matter.\n\n"
            "The Code does not provide a complete list of "
            "disciplinary measures for every violation."
        ), [
            ("04_Code_of_Conduct.pdf", 1),
            ("04_Code_of_Conduct.pdf", 2),
            ("04_Code_of_Conduct.pdf", 3)
        ]

    # ----------------------------------------------
    # CODE OF CONDUCT — GENERAL RESPONSIBILITIES
    # ----------------------------------------------

    if (
        "code of conduct" in q
        and any(term in q for term in [
            "responsibilities", "responsibility",
            "expected behaviour", "expected behavior",
            "main rules", "main duties"
        ])
    ):

        return (
            "According to the Zyro Dynamics Code of Conduct, "
            "employees are expected to:\n\n"
            "- Treat colleagues, clients, vendors, and visitors "
            "with dignity and respect.\n"
            "- Maintain professional language and communication.\n"
            "- Resolve disagreements through respectful dialogue.\n"
            "- Avoid offensive or discriminatory language.\n"
            "- Take ownership of work and commitments.\n"
            "- Be punctual for work and meetings.\n"
            "- Protect confidential company information.\n"
            "- Disclose actual or potential conflicts of interest.\n"
            "- Follow the company's social media rules."
        ), [
            ("04_Code_of_Conduct.pdf", 1),
            ("04_Code_of_Conduct.pdf", 2),
            ("04_Code_of_Conduct.pdf", 3)
        ]

    # ----------------------------------------------
    # CODE OF CONDUCT — CONFLICTS OF INTEREST
    # ----------------------------------------------

    if any(term in q for term in [
        "conflict of interest",
        "conflicts of interest"
    ]):

        return (
            "According to the Zyro Dynamics Code of Conduct:\n\n"
            "- Disclose actual or potential conflicts in writing "
            "to the reporting manager and HR immediately.\n"
            "- Financial interests in competitors, customers, or "
            "suppliers require written clearance from Legal and Compliance.\n"
            "- Disclose immediate family employment with a competitor "
            "or vendor to HR.\n"
            "- Paid outside employment or consulting requires prior "
            "written approval from the Chief People Officer."
        ), [("04_Code_of_Conduct.pdf", 2)]

    # ----------------------------------------------
    # CODE OF CONDUCT — GIFTS AND HOSPITALITY
    # ----------------------------------------------

    if any(term in q for term in [
        "gift", "gifts", "hospitality", "bribe"
    ]):

        return (
            "According to the Zyro Dynamics Code of Conduct:\n\n"
            "- Gifts under Rs. 1,000 may be accepted but must be "
            "declared to the reporting manager.\n"
            "- Gifts over Rs. 1,000 must be declined or surrendered "
            "to HR.\n"
            "- Client-hosted meals are permitted up to Rs. 3,000 "
            "per person.\n"
            "- Vendor-sponsored domestic travel is not permitted "
            "without prior written approval from the CFO.\n"
            "- Branded client gifts require Finance pre-approval "
            "and must cost under Rs. 2,000 per item.\n"
            "- Accepting cash is never permitted."
        ), [("04_Code_of_Conduct.pdf", 3)]

    # ----------------------------------------------
    # WORK FROM HOME
    # ----------------------------------------------

    if any(term in q for term in [
        "work from home", "work-from-home",
        "wfh", "hybrid work", "remote work"
    ]):

        if (
            "hybrid" in q
            and any(term in q for term in [
                "eligible", "eligibility", "who", "days"
            ])
        ):

            return (
                "According to the Zyro Dynamics Work From Home Policy:\n\n"
                "**Eligibility:** Employees at Level 3 (L3) and above.\n\n"
                "**Maximum allowance:** Up to 3 days per week.\n\n"
                "**Approval:** Fixed WFH days must be agreed with "
                "the reporting manager in writing."
            ), [("03_Work_From_Home_Policy.pdf", 2)]

        return (
            "According to the Zyro Dynamics Work From Home Policy:\n\n"
            "**Hybrid WFH:** L3 and above, up to 3 days per week.\n\n"
            "**Full Remote:** L5 and above, case-by-case approval, "
            "up to 5 days per week.\n\n"
            "**Ad-hoc WFH:** L3 and above, up to 2 days per week.\n\n"
            "**Emergency WFH:** Available to all employees as "
            "directed by HR during declared emergencies."
        ), [("03_Work_From_Home_Policy.pdf", 2)]

    # ----------------------------------------------
    # ANNUAL LEAVE
    # ----------------------------------------------

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
            "**Earned Leave:** 15 days after one year of continuous "
            "service, subject to at least 240 days worked in that year. "
            "Thereafter, EL accrues at 1.25 days per month.\n\n"
            "**Casual Leave:** 8 days per year.\n\n"
            "**Sick Leave:** 10 days per year.\n\n"
            "**Probation:** Earned Leave accrues at 0.5 days per month "
            "and becomes available after probation confirmation."
        ), [("02_Leave_Policy.pdf", 2)]

    # ----------------------------------------------
    # EARNED LEAVE
    # ----------------------------------------------

    if (
        "earned leave" in q
        and any(term in q for term in [
            "eligibility", "eligible", "criteria",
            "qualify", "requirement"
        ])
    ):

        return (
            "According to the Zyro Dynamics Leave Policy:\n\n"
            "- 15 days after one year of continuous service.\n"
            "- At least 240 days must have been worked during that year.\n"
            "- Thereafter, EL accrues at 1.25 days per month.\n"
            "- During probation, EL accrues at 0.5 days per month.\n"
            "- Probationary EL can be used after confirmation."
        ), [("02_Leave_Policy.pdf", 2)]

    # ----------------------------------------------
    # MATERNITY LEAVE
    # ----------------------------------------------

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

    # ----------------------------------------------
    # PATERNITY LEAVE
    # ----------------------------------------------

    if "paternity leave" in q:

        return (
            "According to the Zyro Dynamics Leave Policy:\n\n"
            "- 10 consecutive days of paid Paternity Leave.\n"
            "- Must be taken within 90 days of birth or legal adoption.\n"
            "- One-time entitlement per child.\n"
            "- Cannot be split."
        ), [("02_Leave_Policy.pdf", 3)]

    # ----------------------------------------------
    # BEREAVEMENT LEAVE
    # ----------------------------------------------

    if "bereavement leave" in q:

        return (
            "According to the Zyro Dynamics Leave Policy:\n\n"
            "- 5 days of paid Bereavement Leave.\n"
            "- Covers spouse, children, parents, parents-in-law, "
            "and siblings."
        ), [("02_Leave_Policy.pdf", 3)]

    # ----------------------------------------------
    # MARRIAGE LEAVE
    # ----------------------------------------------

    if "marriage leave" in q:

        return (
            "According to the Zyro Dynamics Leave Policy:\n\n"
            "- 3 consecutive paid days for an employee's own marriage.\n"
            "- Available once during employment.\n"
            "- Apply at least 2 weeks in advance."
        ), [("02_Leave_Policy.pdf", 3)]

    # ----------------------------------------------
    # CASUAL LEAVE
    # ----------------------------------------------

    if "casual leave" in q:

        return (
            "According to the Zyro Dynamics Leave Policy:\n\n"
            "- 8 days of Casual Leave per year.\n"
            "- Credited monthly subject to the policy conditions.\n"
            "- Cannot be carried forward or encashed.\n"
            "- Maximum of 2 consecutive days."
        ), [("02_Leave_Policy.pdf", 2)]

    # ----------------------------------------------
    # SICK LEAVE
    # ----------------------------------------------

    if "sick leave" in q:

        return (
            "According to the Zyro Dynamics Leave Policy:\n\n"
            "- 10 days of Sick Leave per year.\n"
            "- Credited monthly subject to the policy conditions.\n"
            "- Cannot be carried forward or encashed.\n"
            "- More than 2 consecutive days requires a Medical "
            "Certificate from a registered medical practitioner."
        ), [("02_Leave_Policy.pdf", 2)]

    return None, []


# ==================================================
# SIMPLE OUT-OF-SCOPE CHECK
# ==================================================

def is_likely_hr_question(question):

    q = question.lower()

    hr_terms = [
        "employee", "employees", "staff",
        "hr", "human resource", "workplace",
        "work", "working", "job", "office",
        "leave", "holiday", "vacation",
        "salary", "payroll", "compensation",
        "benefit", "bonus", "appraisal",
        "performance", "review", "conduct",
        "misconduct", "behaviour", "behavior",
        "ethics", "gift", "vendor", "conflict",
        "confidential", "data security",
        "password", "device", "access",
        "harassment", "posh", "onboarding",
        "joining", "resignation", "exit",
        "separation", "termination",
        "notice period", "travel", "expense",
        "reimbursement", "attendance",
        "punctuality", "probation", "promotion",
        "manager", "grievance", "complaint",
        "policy", "policies", "handbook",
        "company", "wfh", "remote",
        "hybrid", "working hours",
        "working days", "full-time",
        "full time", "grade", "training",
        "maternity", "paternity", "sick leave",
        "casual leave", "earned leave",
        "annual leave", "working time",
        "whistleblower", "retaliation"
    ]

    return any(term in q for term in hr_terms)


# ==================================================
# INITIALIZE APP
# ==================================================

with st.spinner("Loading HR policy documents..."):

    chunks, embedder, index = load_rag()

if chunks is None:

    st.error(
        "HR policy PDFs were not found. Please check "
        "the hr_policies folder in your repository."
    )

    st.stop()

st.success("HR policy assistant is ready!")


# ==================================================
# CHAT HISTORY
# ==================================================

if "messages" not in st.session_state:

    st.session_state.messages = []

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ==================================================
# CHAT INPUT
# ==================================================

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

            answer, direct_sources = get_policy_answer(
                question
            )

            if direct_sources:

                sources = [
                    {
                        "source": source,
                        "page": page
                    }
                    for source, page in direct_sources
                ]

            elif not is_likely_hr_question(question):

                answer = (
                    "I'm sorry, but I can only answer questions "
                    "related to Zyro Dynamics HR policies, "
                    "employee handbook, and workplace procedures. "
                    "Please ask an HR-related question."
                )

                sources = []

            else:

                retrieved = retrieve_documents(
                    question,
                    chunks,
                    embedder,
                    index
                )

                if not retrieved:

                    answer = (
                        "I could not find sufficiently relevant "
                        "information in the Zyro Dynamics HR policy "
                        "documents. Please try rephrasing your question "
                        "or contact HR."
                    )

                    sources = []

                else:

                    answer = (
                        "Here are the most relevant excerpts "
                        "from the HR policy documents:\n\n"
                    )

                    for item in retrieved:

                        chunk = item["chunk"]

                        answer += (
                            f"**{chunk['source']} — "
                            f"Page {chunk['page']}**\n\n"
                            f"{chunk['text']}\n\n"
                            "---\n\n"
                        )

                    sources = format_sources(retrieved)

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
