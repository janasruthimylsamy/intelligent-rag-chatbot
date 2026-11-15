
import os
import re
import streamlit as st
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

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
</style>
""", unsafe_allow_html=True)


# ==================================================
# 2. DOCUMENT SETTINGS
# ==================================================

POLICY_FOLDER = "hr_policies"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

POSH_FILE = "08_Prevention_of_Sexual_Harassment_Policy.pdf"


# ==================================================
# 3. LOAD POLICY PDF DOCUMENTS
# ==================================================

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


# ==================================================
# 4. CREATE TEXT CHUNKS
# ==================================================

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


# ==================================================
# 5. CREATE EMBEDDINGS AND FAISS INDEX
# ==================================================

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

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return model, index, chunks


# ==================================================
# 6. IDENTIFY THE RELEVANT POLICY DOCUMENT
# ==================================================

def get_document_filter(question):

    q = question.lower()

    document_rules = {

        "01_Employee_Handbook.pdf": [
            "working hours", "work hours", "hours per day",
            "full-time employee", "full time employee",
            "working days", "attendance", "office timing",
            "office hours", "daily working", "employee handbook",
            "dress code", "attendance policy",
            "standard working schedule", "standard work schedule",
            "working schedule", "standard schedule",
            "including lunch break", "lunch break"
        ],

        POSH_FILE: [
            "sexual harassment", "posh", "harassment complaint",
            "internal complaints committee", "icc",
            "unwelcome sexual", "sexual misconduct",
            "sexually coloured", "sexually explicit"
        ],

        "04_Code_of_Conduct.pdf": [
            "code of conduct", "misconduct",
            "cash from vendor", "confidential information",
            "conflict of interest", "social media",
            "report violations", "reporting violations",
            "whistleblower", "gifts from vendors"
        ],

        "03_Work_From_Home_Policy.pdf": [
            "work from home", "work remotely",
            "hybrid work", "remote work", "wfh"
        ],

        "02_Leave_Policy.pdf": [
            "annual leave", "earned leave", "casual leave",
            "sick leave", "maternity leave", "paternity leave",
            "bereavement leave", "marriage leave",
            "leave entitlement", "leave policy"
        ],

        "05_Performance_Review_Policy.pdf": [
            "performance review", "performance appraisal",
            "performance rating", "promotion criteria",
            "review cycle", "360 degree feedback"
        ],

        "06_Compensation_and_Benefits_Policy.pdf": [
            "compensation", "salary structure",
            "employee benefits", "bonus", "increment",
            "pay structure", "benefits policy",
            "insurance", "provident fund", "gratuity",
            "wellness allowance", "employee assistance",
            "esop", "stock options"
        ],

        "07_IT_and_Data_Security_Policy.pdf": [
            "data security", "password policy",
            "information security", "cybersecurity",
            "it policy", "company laptop", "data breach",
            "security incident", "phishing", "malware",
            "unauthorised access", "unauthorized access"
        ],

        "09_Onboarding_and_Separation_Policy.pdf": [
            "onboarding", "joining formalities",
            "resignation", "notice period",
            "exit process", "separation policy",
            "offboarding", "leaving the company",
            "last working day", "exit formalities"
        ],

        "10_Travel_and_Expense_Policy.pdf": [
            "travel policy", "travel expenses",
            "reimbursement", "business travel",
            "hotel expenses", "travel allowance",
            "eligible expenses", "travel claim"
        ]
    }

    for filename, terms in document_rules.items():

        if any(term in q for term in terms):
            return filename

    return None


# ==================================================
# 7. RETRIEVE RELEVANT POLICY EXCERPTS
# ==================================================

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

    search_count = len(chunks)

    scores, indices = index.search(
        query_embedding,
        search_count
    )

    selected_document = get_document_filter(question)

    stop_words = {
        "what", "when", "where", "which", "who", "why",
        "how", "does", "do", "did", "is", "are", "was",
        "were", "the", "a", "an", "and", "or", "to",
        "of", "in", "on", "for", "from", "with", "by",
        "at", "as", "be", "can", "could", "would",
        "should", "will", "shall", "has", "have", "had",
        "i", "we", "they", "he", "she", "it", "my",
        "our", "their", "me", "us", "this", "that",
        "these", "those", "please", "tell", "explain",
        "company", "employee", "employees", "policy",
        "policies", "information", "regarding", "about",
        "according", "companys", "company's", "code"
    }

    question_words = set(
        re.findall(r"[a-z]{3,}", question.lower())
    )

    meaningful_terms = question_words - stop_words

    results = []

    for score, chunk_index in zip(scores[0], indices[0]):

        if chunk_index < 0:
            continue

        chunk = chunks[chunk_index]

        if (
            selected_document
            and chunk["source"] != selected_document
        ):
            continue

        # Reject weak semantic matches.
        if score < 0.30:
            continue

        chunk_words = set(
            re.findall(r"[a-z]{3,}", chunk["text"].lower())
        )

        # Require a meaningful topic-word match.
        matching_terms = meaningful_terms.intersection(chunk_words)

        if not matching_terms:
            continue

        results.append({
            "text": chunk["text"],
            "source": chunk["source"],
            "page": chunk["page"],
            "score": float(score)
        })

    return results[:5]


# ==================================================
# 8. HELPER FOR DIRECT ANSWERS
# ==================================================

def make_answer(answer, page_numbers, filename):

    return {
        "answer": answer.strip(),
        "sources": [
            (filename, page)
            for page in page_numbers
        ]
    }


# ==================================================
# 9. DIRECT POLICY ANSWERS
# ==================================================

def get_direct_answer(question):

    q = re.sub(r"\s+", " ", question.lower().strip())

    handbook = "01_Employee_Handbook.pdf"
    leave = "02_Leave_Policy.pdf"
    wfh = "03_Work_From_Home_Policy.pdf"
    conduct = "04_Code_of_Conduct.pdf"
    performance = "05_Performance_Review_Policy.pdf"
    benefits = "06_Compensation_and_Benefits_Policy.pdf"
    security = "07_IT_and_Data_Security_Policy.pdf"
    separation = "09_Onboarding_and_Separation_Policy.pdf"
    travel = "10_Travel_and_Expense_Policy.pdf"

    # --------------------------------------------------
    # POSH: TYPES OF SEXUAL HARASSMENT
    # --------------------------------------------------

    behaviour_terms = [
        "types of behaviour", "types of behavior",
        "type of behaviour", "type of behavior",
        "what behaviours", "what behaviors",
        "what behaviour", "what behavior",
        "types of conduct", "examples of sexual harassment",
        "what constitutes sexual harassment",
        "what is considered sexual harassment",
        "meaning of sexual harassment",
        "define sexual harassment",
        "what counts as sexual harassment",
        "forms of sexual harassment"
    ]

    if (
        any(term in q for term in behaviour_terms)
        or (
            "sexual harassment" in q
            and any(term in q for term in [
                "behaviour", "behavior", "examples",
                "constitute", "considered", "meaning",
                "types", "definition", "include"
            ])
        )
    ):

        return make_answer("""
According to the Zyro Dynamics POSH Policy, sexual harassment includes the following unwelcome behaviours:

### Types of sexual harassment

1. **Unwelcome physical contact or advances:** Unwanted physical touching or advances of a sexual nature.

2. **Requests or demands for sexual favours:** Asking or pressuring someone to provide sexual favours.

3. **Sexually coloured remarks, gestures, or jokes:** Making unwelcome comments, gestures, or jokes of a sexual nature.

4. **Showing or sharing pornography:** Displaying or sharing pornographic or sexually explicit content.

5. **Other unwelcome conduct:** Any other unwelcome physical, verbal, or non-verbal conduct of a sexual nature.

### Related situations covered by the policy

- Implied or explicit promises of preferential treatment.
- Threats of adverse employment consequences.
- Creating a hostile work environment.
- Humiliating treatment that may affect a person's health or safety.

These examples are based on the company's POSH Policy.
""", [2], POSH_FILE)

    # --------------------------------------------------
    # POSH: POLICY COVERAGE
    # --------------------------------------------------

    coverage_terms = [
        "who is covered", "who are covered",
        "who does the policy cover", "who is covered under",
        "who comes under", "who does the posh policy apply to",
        "who does this policy apply to", "who is included",
        "who are included", "scope of the posh policy",
        "posh policy scope", "coverage of the posh policy",
        "people covered under", "persons covered under",
        "employees covered under", "third party personnel",
        "third-party personnel"
    ]

    if any(term in q for term in coverage_terms):

        return make_answer("""
According to the Zyro Dynamics POSH Policy, the policy covers:

### People covered

- Full-time employees.
- Fixed-term employees.
- Probationary employees.
- Contractual employees.
- Interns.
- Trainees.
- Third-party personnel.

### Where does the policy apply?

- Company offices in India and abroad.
- Client sites.
- Coworking spaces.
- Social, business, and professional events.
- Business travel.
- Digital communication channels connected to the workplace.

The policy is not limited to permanent or full-time employees.
""", [2], POSH_FILE)

    # --------------------------------------------------
    # POSH: ICC MEMBERSHIP
    # --------------------------------------------------

    membership_terms = [
        "who are the members", "who is on the committee",
        "who are on the committee", "committee members",
        "icc members", "members of the icc",
        "members of internal complaints committee",
        "internal complaints committee members",
        "who is in the icc", "who are in the icc",
        "composition of the icc", "icc composition",
        "presiding officer", "external member",
        "internal member"
    ]

    if any(term in q for term in membership_terms):

        return make_answer("""
According to the Zyro Dynamics POSH Policy, the Internal Complaints Committee (ICC) consists of four members:

1. **Marta** – Presiding Officer and Chief People Officer.
2. **Pooja Ramachandran** – Internal Member, Legal & Compliance.
3. **Aryan Kapoor** – Internal Member, Engineering Manager (L6).
4. **Adv. Preethi Srinivas** – External Member from an external women's rights organisation.

The ICC handles workplace sexual-harassment complaints and conducts inquiries under the POSH Policy.
""", [2], POSH_FILE)

    # --------------------------------------------------
    # POSH: CONFIDENTIALITY
    # --------------------------------------------------

    confidentiality_terms = [
        "confidentiality", "confidential",
        "identity confidential", "identities confidential",
        "kept confidential", "remain confidential",
        "privacy of complainant", "identity of complainant",
        "identity of the complainant", "complainant identity",
        "confidentiality of complaint",
        "confidentiality of proceedings",
        "who can know about the complaint",
        "will my identity be revealed",
        "is the complaint confidential"
    ]

    if (
        any(term in q for term in confidentiality_terms)
        and any(term in q for term in [
            "harassment", "posh", "icc", "complainant",
            "complaint", "identity", "inquiry", "witness"
        ])
    ):

        return make_answer("""
**Yes.** The Zyro Dynamics POSH Policy requires strict confidentiality throughout the complaint and inquiry process.

The following information must remain confidential:

- Identity of the complainant.
- Identity of the respondent.
- Identities of witnesses.
- Inquiry proceedings.
- Inquiry findings and recommendations.

Disclosure is permitted only to people directly involved in the matter or those who need the information to implement the recommendations.

A breach of confidentiality is itself a disciplinary offence.

The policy also prohibits retaliation against anyone who files a complaint, participates in an inquiry, or reports a concern in good faith.
""", [3], POSH_FILE)

    # --------------------------------------------------
    # POSH: DISCIPLINARY ACTION
    # --------------------------------------------------

    disciplinary_terms = [
        "disciplinary action", "disciplinary measures",
        "disciplinary consequences", "possible consequences",
        "what action can be taken", "what action will be taken",
        "action against the respondent", "action against an employee",
        "action against the accused", "found guilty",
        "found responsible", "if the employee is guilty",
        "if found guilty", "punishment for sexual harassment",
        "penalty for sexual harassment",
        "consequences of sexual harassment",
        "sanctions for sexual harassment",
        "what happens to the accused",
        "termination for sexual harassment"
    ]

    if (
        any(term in q for term in disciplinary_terms)
        and any(term in q for term in [
            "harassment", "posh", "icc", "respondent",
            "employee", "accused", "sexual", "guilty"
        ])
    ):

        return make_answer("""
According to the Zyro Dynamics POSH Policy, disciplinary action may be taken depending on the findings and recommendations of the Internal Complaints Committee (ICC).

Possible consequences include:

1. Written warning with sensitisation and inclusion training.
2. Suspension without pay.
3. Transfer to another team, project, or location.
4. Withholding promotion, salary increments, or other employment benefits.
5. Termination of employment.
6. Referral to law enforcement if the conduct constitutes a criminal offence under Indian law.

The appropriate action depends on the ICC inquiry findings and recommendations.
""", [3], POSH_FILE)

    # --------------------------------------------------
    # POSH: PROTECTION FROM RETALIATION
    # --------------------------------------------------

    retaliation_terms = [
        "protection from retaliation",
        "protected from retaliation", "retaliation against",
        "retaliation protection", "retaliatory action",
        "retaliatory actions", "punished for reporting",
        "punished for complaining", "revenge for reporting",
        "protection for reporting", "protection does the company provide",
        "reporting sexual harassment"
    ]

    if (
        any(term in q for term in retaliation_terms)
        and any(term in q for term in [
            "harassment", "posh", "icc", "complaint",
            "report", "employee", "retaliation"
        ])
    ):

        return make_answer("""
The Zyro Dynamics POSH Policy strictly prohibits retaliation against anyone who:

- Files a sexual-harassment complaint.
- Participates in an ICC inquiry.
- Reports a suspected violation in good faith.

Examples of prohibited retaliation include:

- Adverse performance ratings.
- Demotion or transfer.
- Isolation.
- Threats.
- Termination.

Retaliation is treated as a separate and serious disciplinary matter under the policy.
""", [2, 3], POSH_FILE)

    # --------------------------------------------------
    # POSH: COMPLAINT PROCEDURE
    # --------------------------------------------------

    complaint_intent = any(term in q for term in [
        "file a complaint", "submit a complaint",
        "make a complaint", "complaint procedure",
        "complaint process", "how to complain",
        "how can i complain", "how do i report",
        "how to report", "report sexual harassment",
        "report harassment", "experience sexual harassment",
        "experiences sexual harassment",
        "what should i do if", "what should an employee do if"
    ])

    if (
        complaint_intent
        and any(term in q for term in [
            "harassment", "posh", "icc", "sexual",
            "workplace", "complaint", "report"
        ])
    ):

        return make_answer("""
If an employee experiences sexual harassment, they can submit a written complaint to the Internal Complaints Committee (ICC).

### Step 1: Submit a written complaint

- Submit the complaint by email to the ICC or as a sealed letter addressed to the Presiding Officer at the HR office at Zyro Dynamics HQ.
- Submit it within 3 months of the incident, or within 3 months of the last incident in a series.
- In exceptional circumstances, the ICC may extend this period by a further 3 months upon a written application explaining the reasons.

**ICC email clarification:** The address in the extracted policy appears as `icc@zyrodyn amics.com`. Verify the correct address with HR before sending. The sealed-letter option is also available.

### Step 2: Acknowledgement

The ICC acknowledges receipt and forwards a copy to the respondent within 7 working days.

### Step 3: Respondent's reply

The respondent gets 10 working days to submit a written response.

### Step 4: Inquiry

The ICC conducts hearings with both parties, considers witnesses, and reviews evidence. The inquiry is to be completed within 60 days.

### Step 5: Inquiry report

The ICC submits its report and recommendations to management within 10 days of completing the inquiry.

### Step 6: Management action

Management implements the recommended action based on the ICC's findings.

The policy also requires confidentiality and prohibits retaliation against people who report complaints or participate in inquiries in good faith.
""", [2, 3], POSH_FILE)

    # --------------------------------------------------
    # ATTENDANCE-MARKING PROCEDURE
    # --------------------------------------------------

    attendance_terms = [
        "attendance-marking", "attendance marking",
        "mark attendance", "marking attendance",
        "attendance procedure", "attendance process",
        "how is attendance", "how do employees record attendance",
        "record attendance", "attendance recorded",
        "mark entry and exit", "biometric attendance",
        "attendance system", "attendance tracking"
    ]

    if any(term in q for term in attendance_terms):

        return make_answer("""
According to the Employee Handbook, employees must record their attendance every working day.

### Attendance methods

- **Office employees:** Attendance is recorded through biometric terminals at office locations.
- **Remote employees:** Attendance is recorded through the AcruxHR mobile application.

### Employee responsibilities

- Mark entry when starting work.
- Mark exit when finishing work.
- Record both entry and exit every working day without exception.

### Missing attendance

Three or more consecutive days of absent or unmarked attendance without a corresponding approved leave entry will trigger an escalation to the HR Business Partner.

### Attendance portal

The handbook identifies the AcruxHR portal for attendance and employee self-service.
""", [4], handbook)

    # --------------------------------------------------
    # WORKING HOURS AND STANDARD WORK SCHEDULE
    # --------------------------------------------------

    working_terms = [
        "working hours", "work hours", "hours per day",
        "full-time employee", "full time employee",
        "how many hours", "working days", "office hours",
        "office timing", "daily working hours",
        "expected to work", "standard working schedule",
        "standard work schedule", "including lunch break",
        "working schedule", "standard schedule",
        "work schedule", "lunch break"
    ]

    if any(term in q for term in working_terms):

        return make_answer("""
According to the Employee Handbook, employees are expected to complete a minimum of 9 working hours per day.

### Standard schedule

- Working days: Monday to Friday.
- Office hours: 9:30 AM to 6:30 PM IST.
- Unpaid lunch break: 30 minutes.
- Stated minimum working time: 9 hours per day.

**Important clarification:** There is an inconsistency in the handbook. The 9:30 AM to 6:30 PM schedule is 9 hours in total. After subtracting the 30-minute unpaid lunch break, the actual working time is 8.5 hours. However, the handbook separately states a minimum of 9 working hours.

Employees should confirm the correct expected working duration with HR.

### Flexible working hours

Employees at grade L3 and above may opt for flexible hours within the 10:00 AM to 4:00 PM core band, subject to written manager approval.
""", [3, 4], handbook)

    # --------------------------------------------------
    # SICK LEAVE
    # --------------------------------------------------

    sick_leave_terms = [
        "sick leave", "sick leave entitlement",
        "sick leave days", "medical leave",
        "leave when sick", "leave for illness"
    ]

    if any(term in q for term in sick_leave_terms):

        return make_answer("""
According to the Zyro Dynamics Leave Policy, employees are entitled to **10 days of Sick Leave (SL) per year**.

### Sick Leave rules

- Annual entitlement: 10 days.
- Sick Leave is credited on the first day of the following month, provided the employee worked for at least 20 days during the previous month.
- Sick Leave cannot be carried forward to the next financial year.
- Unused Sick Leave cannot be encashed.
- Any remaining balance lapses at the end of the financial year on 31 March.
- Sick Leave exceeding 2 consecutive days requires a Medical Certificate from a registered medical practitioner.
- The certificate must be submitted within 3 working days of returning to work.

For unplanned sickness or emergencies, employees should follow the leave application procedure in the Leave Policy.
""", [2], leave)

    # --------------------------------------------------
    # ANNUAL / EARNED LEAVE
    # --------------------------------------------------

    if (
        "annual leave" in q
        or "earned leave" in q
        or "casual leave" in q
    ):

        return make_answer("""
According to the Zyro Dynamics Leave Policy:

- Earned Leave: 15 days after one year of service and 240 working days; thereafter, 1.25 days per month.
- Casual Leave: 8 days per year.
- Sick Leave: 10 days per year.
- During probation, Earned Leave is credited at 0.5 days per month and becomes available after confirmation.

Casual Leave and Sick Leave cannot be carried forward or encashed. Please refer to the Leave Policy for the applicable conditions.
""", [2], leave)

    # --------------------------------------------------
    # BUSINESS TRAVEL EXPENSES
    # --------------------------------------------------

    travel_terms = [
        "travel expenses", "business travel expenses",
        "eligible expenses", "expenses can employees claim",
        "claim during official travel", "travel reimbursement",
        "expenses during business travel",
        "official business travel", "travel claim",
        "reimbursable travel", "travel allowance"
    ]

    if any(term in q for term in travel_terms):

        return make_answer("""
According to the Zyro Dynamics Travel and Expense Policy, eligible official business travel expenses include:

### Eligible expenses

1. **Flights:** Economy or business class, as applicable to the employee's grade and policy limits.
2. **Accommodation:** Business hotels within the approved per-night rate limit.
3. **Meals:** Within the applicable daily allowance limit and supported by valid receipts.
4. **Ground transport:** Cab or taxi travel to airports and client locations.
5. **International roaming and calls:** Documented business-related calls.
6. **Laundry:** Eligible for trips exceeding 4 nights.

### Expenses that are not eligible

- Flight upgrades beyond grade entitlement.
- Personal travel extensions.
- Luxury suites.
- Alcoholic beverages.
- Meals for personal guests.
- Personal trips, sightseeing, and leisure travel.
- Personal calls, streaming, and entertainment.
- Laundry for trips of 4 nights or fewer.

### Claim submission requirements

- Submit the expense report through the ZyroHR portal within 7 working days of returning.
- Individual expenses above ₹500 require an original receipt, GST invoice, or valid digital invoice.
- Expenses without supporting documents will not be reimbursed.
- Reports submitted after 30 calendar days will not be processed.

All official travel must be pre-approved. The policy's specific grade-based limits and daily allowance amounts should be checked before booking.
""", [1, 2, 3], travel)

    # --------------------------------------------------
    # DATA SECURITY INCIDENT REPORTING
    # --------------------------------------------------

    incident_terms = [
        "security incident", "data security incident",
        "suspected security", "suspect a data",
        "data breach", "phishing", "malware",
        "lost or stolen device", "unauthorised access",
        "unauthorized access", "security breach",
        "suspected data leak", "report a security incident",
        "reporting security incident"
    ]

    if any(term in q for term in incident_terms):

        return make_answer("""
According to the Zyro Dynamics IT and Data Security Policy, any actual or suspected security incident must be reported without delay.

### Where to report

Report incidents to the IT Security team. The extracted policy displays the email as `security@zyrodyn amics.com`. Because the extracted address contains a space, verify the correct email address with IT before sending.

### Reporting timelines

| Incident type | Report to | Timeline |
|---|---|---|
| Phishing or suspicious email | IT Security | Immediately |
| Malware or suspicious device behaviour | IT Helpdesk and IT Security | Within 1 hour |
| Data breach or suspected leak | CISO, Legal, and HR | Within 2 hours |
| Lost or stolen device | IT and HR | Within 4 hours |
| Suspected unauthorised access | IT Security and reporting manager | Immediately |

Examples of reportable incidents include data loss, malware infection, phishing, unauthorised access, and device loss.

**Important:** Employees must not delay reporting a suspected incident. The policy states that employees who fail to report a known or suspected incident may be held accountable for resulting harm.
""", [3], security)

    # --------------------------------------------------
    # PERFORMANCE REVIEW FREQUENCY
    # --------------------------------------------------

    performance_terms = [
        "performance review", "performance appraisal",
        "review frequency", "how frequently",
        "how often are reviews", "review cycles",
        "performance evaluation", "mid-year review",
        "annual performance review", "360 degree feedback"
    ]

    if any(term in q for term in performance_terms):

        return make_answer("""
According to the Zyro Dynamics Performance Review Policy, the company follows these review cycles:

| Review type | Frequency / Timing |
|---|---|
| Probation Review | At Month 3 and Month 6 |
| Mid-Year Review (MYR) | Annually, in September |
| Annual Performance Review (APR) | Annually, in March |
| 360-Degree Feedback | Annually, in February |
| PIP Review | Monthly during a Performance Improvement Plan |

### Annual review process

- 1–20 February: 360-degree feedback.
- 1–10 March: Employee self-assessment.
- 11–20 March: Manager assessment.
- 21–25 March: Calibration meeting.
- 26–31 March: Final ratings confirmed by HR.
- 1–10 April: One-on-one feedback conversation.
- 15 April: Increment and promotion letters issued.

Therefore, employees have an annual performance review, a September mid-year review, and other reviews where applicable, such as probation and PIP reviews.
""", [2, 3], performance)

    # --------------------------------------------------
    # EMPLOYEE RESIGNATION AND EXIT FORMALITIES
    # --------------------------------------------------

    separation_terms = [
        "leaving the company", "leaving the organisation",
        "leaving the organization", "resign",
        "resignation", "exit formalities",
        "exit process", "separation process",
        "offboarding", "last working day",
        "when leaving", "notice period",
        "employee leaving", "return company assets"
    ]

    if any(term in q for term in separation_terms):

        return make_answer("""
According to the Zyro Dynamics Onboarding and Separation Policy, an employee leaving the company must complete the following steps.

### 1. Submit resignation

- Give written notice to the reporting manager and HR.
- Submit the resignation through the ZyroHR portal.
- Obtain written confirmation from HR.
- Follow the notice period applicable to the employee's grade.

### Notice periods

| Grade | Notice period |
|---|---|
| L1 to L3 | 30 days |
| L4 to L6 | 60 days |
| L7 to L9 | 90 days |
| L10 (C-Suite) | 90 days, as per individual employment contract |

### 2. Complete handover

Complete knowledge transfer and handover documentation with the manager one week before the last working day.

### 3. Return company assets

Return the laptop, phone, access cards, and all other company assets to IT on the last working day.

### 4. Complete clearances

- IT revokes system access and credentials on the last working day.
- Finance clearance for advances, loans, and expenses must be obtained 3 days before the last working day.
- Attend the HR exit interview during the final week of employment.

### 5. Receive exit documents

- Experience letter: HR issues it within 7 working days of the last working day.
- Relieving letter: HR issues it after Full and Final settlement is processed.

Employees must also return or delete confidential documents and credentials as directed by IT and HR.
""", [2, 3], separation)

    # --------------------------------------------------
    # EMPLOYEE BENEFITS
    # --------------------------------------------------

    benefits_terms = [
        "employee benefits", "benefits provided",
        "what benefits", "company benefits",
        "insurance benefits", "health insurance",
        "medical insurance", "wellness allowance",
        "employee assistance programme",
        "employee assistance program",
        "provident fund", "gratuity",
        "stock options", "esop"
    ]

    if any(term in q for term in benefits_terms):

        return make_answer("""
According to the Zyro Dynamics Compensation and Benefits Policy, employee benefits include the following:

### 1. Health and insurance benefits

- **Group Medical Insurance:** Coverage up to ₹5,00,000 per year for the employee, spouse, and up to two dependent children. The company pays the premiums.
- **Personal Accident Insurance:** Coverage equal to 5 times the employee's annual CTC.
- **Term Life Insurance:** Coverage equal to 3 times annual CTC for permanent employees.

### 2. Retirement benefits

- **Provident Fund (PF):** Both the employee and company contribute 12% of the employee's basic salary each month.
- **Gratuity:** Payable to employees who complete 5 or more years of continuous service, according to applicable law.

### 3. Wellness and lifestyle benefits

- **Annual Wellness Allowance:** ₹12,000 per year, available at ₹1,000 per month for eligible wellness expenses such as gym memberships, fitness apps, health check-ups, and mental health consultations.
- **Employee Assistance Programme (EAP):** Up to 6 free and confidential counselling sessions per year.

### 4. Employee Stock Options

- ESOPs are offered to employees at grade L5 and above.
- Vesting schedule: 4 years.
- Cliff period: 1 year.

Eligibility and applicable conditions depend on the relevant benefit provisions and employment terms.
""", [3], benefits)

    # --------------------------------------------------
    # CASH ACCEPTANCE
    # --------------------------------------------------

    if "cash" in q and any(term in q for term in [
        "vendor", "employee", "accept", "receive", "gift"
    ]):

        return make_answer("""
Accepting cash from any party is strictly prohibited.

The Code of Conduct states that accepting cash is never permitted. Violations may result in immediate disciplinary action.

The company follows a zero-tolerance approach to accepting cash.
""", [3], conduct)

    # --------------------------------------------------
    # CONFIDENTIAL INFORMATION
    # --------------------------------------------------

    if any(term in q for term in [
        "confidential information", "confidential data",
        "company information", "data leak"
    ]):

        return make_answer("""
Employees must share confidential information only when there is an authorised business need.

Confidential information includes:

- Customer and prospect data.
- Product roadmaps.
- Pricing and financial figures.
- HR records.
- Business strategies.
- Source code and other non-public information.

Employees must not post company information on social media, share it through personal devices, or disclose it to the media.

When leaving the company, employees must return or delete confidential documents and credentials as directed by IT or HR.

Violations may result in disciplinary action up to and including termination and may also attract civil or criminal liability under applicable law.
""", [2], conduct)

    # --------------------------------------------------
    # SOCIAL MEDIA
    # --------------------------------------------------

    if "social media" in q:

        return make_answer("""
Employees must not share confidential company information, internal communications, client names, product screenshots, or financial information on social media.

Employees must not present personal opinions as the official position of the company unless authorised by Marketing.

Negative commentary or criticism of colleagues, management, or clients on social media is prohibited and may result in disciplinary action.

Employees should include this disclaimer when expressing personal views:

**"Views expressed are my own and do not represent Zyro Dynamics."**
""", [3], conduct)

    # --------------------------------------------------
    # REPORTING CODE OF CONDUCT VIOLATIONS
    # --------------------------------------------------

    if any(term in q for term in [
        "report violation", "reporting violation",
        "whistleblower", "report misconduct",
        "report a concern", "report code of conduct"
    ]):

        return make_answer("""
Employees can report suspected Code of Conduct violations through:

1. **Reporting Manager:** Report directly or through email.
2. **HR Grievance Cell:** hr.grievance@acruxdynamics.com.
3. **Anonymous Whistleblower Portal:** whistle.acruxdynamics.com.
4. **Chief People Officer:** marta@acruxdynamics.com.

Employees who report violations in good faith are protected. Retaliation against a good-faith reporter is itself a violation and may result in serious disciplinary action.
""", [3], conduct)

    # --------------------------------------------------
    # CONFLICT OF INTEREST
    # --------------------------------------------------

    if "conflict of interest" in q:

        return make_answer("""
Employees must disclose any conflict of interest in writing to their manager and HR immediately.

Important requirements:

- Financial interests in competitors, customers, or suppliers require written clearance from Legal & Compliance.
- Immediate family members employed by a competitor or vendor must be disclosed to HR.
- Paid outside employment or consulting requires prior written approval from the Chief People Officer.
""", [2], conduct)

    # --------------------------------------------------
    # GIFTS AND HOSPITALITY
    # --------------------------------------------------

    if "gift" in q or "hospitality" in q:

        return make_answer("""
The Code of Conduct specifies these gift and hospitality rules:

- Gifts below ₹1,000 are permitted if declared to the manager.
- Gifts above ₹1,000 are not permitted and must be declined or surrendered to HR.
- Client-hosted meals are limited to ₹3,000 per person.
- Vendor-sponsored domestic travel is not permitted without prior written approval from the CFO.
- Branded gifts to clients require Finance pre-approval and must be below ₹2,000 per item.
- Accepting cash from any party is never permitted.
""", [3], conduct)

    # --------------------------------------------------
    # WORK FROM HOME
    # --------------------------------------------------

    if any(term in q for term in [
        "work from home", "hybrid work",
        "remote work", "wfh"
    ]):

        return make_answer("""
According to the Work From Home Policy:

- **Hybrid WFH:** Employees at grade L3 and above may work remotely for up to 3 days per week.
- **Full Remote:** Employees at grade L5 and above may be considered for full remote work on a case-by-case basis.
- **Ad-hoc WFH:** Employees at grade L3 and above may request up to 2 days per week.
- **Emergency WFH:** Available to all employees as directed by HR.

Employees must also have completed at least 6 months of continuous service and have a recent performance rating of Meets Expectations or above to satisfy the stated eligibility criteria.
""", [2], wfh)

    return None


# ==================================================
# 10. CHECK WHETHER QUESTION IS HR-RELATED
# ==================================================

def is_likely_hr_question(question):

    keywords = [
        "employee", "hr", "leave", "salary", "payroll",
        "policy", "company", "work", "working", "office",
        "attendance", "manager", "benefit", "holiday",
        "harassment", "conduct", "confidential",
        "remote", "wfh", "probation", "resignation",
        "termination", "reimbursement", "travel",
        "expense", "working hours", "working days",
        "posh", "icc", "complaint", "retaliation",
        "onboarding", "separation", "performance",
        "appraisal", "security", "password",
        "compensation", "bonus", "allowance",
        "sick", "medical", "insurance", "gratuity",
        "provident", "phishing", "malware",
        "resign", "exit", "review", "benefits",
        "business trip", "business travel",
        "dress code", "dress", "attire",
        "uniform", "appearance", "grooming"
    ]

    q = question.lower()

    return any(keyword in q for keyword in keywords)


# ==================================================
# 11. DISPLAY SOURCES
# ==================================================

def display_sources(sources):

    if not sources:
        return

    st.markdown("**Sources:**")

    seen = set()

    for source, page in sources:

        if (source, page) in seen:
            continue

        seen.add((source, page))

        st.markdown(f"- `{source}` (Page {page})")


# ==================================================
# 12. DISPLAY RETRIEVED EXCERPTS
# ==================================================

def format_retrieved_results(results):

    answer = (
        "I found relevant information in the HR policy documents. "
        "Here are the most relevant excerpts:\n\n"
    )

    seen_sources = set()

    for result in results:

        answer += (
            f"### {result['source']} — Page {result['page']}\n\n"
            f"{result['text']}\n\n"
            "---\n\n"
        )

        seen_sources.add((
            result["source"],
            result["page"]
        ))

    return answer, sorted(seen_sources)


# ==================================================
# 13. USER INTERFACE
# ==================================================

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

    It provides predefined answers for common policy questions
    and retrieves relevant policy excerpts for other questions.
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

    st.divider()

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ==================================================
# 14. LOAD THE RAG SYSTEM
# ==================================================

with st.spinner("Loading HR policy documents..."):

    model, index, chunks = load_rag()


if model is None:

    st.error(
        "No policy documents could be loaded. "
        "Please check that the hr_policies folder contains PDF files."
    )

    st.stop()


# ==================================================
# 15. CHAT HISTORY
# ==================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message.get("sources"):
            display_sources(message["sources"])


# ==================================================
# 16. CHAT INPUT AND ANSWER PROCESSING
# ==================================================

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

            sources = []

            # First: predefined policy answers.

            if direct_result:

                answer = direct_result["answer"]
                sources = direct_result["sources"]

            # Second: reject unrelated questions.

            elif not is_likely_hr_question(question):

                answer = """
I can help only with questions related to the company's
HR policies and employee handbook.

Please ask an HR-related question.
"""

            # Third: retrieve relevant policy content.

            else:

                results = retrieve_documents(
                    question,
                    model,
                    index,
                    chunks
                )

                if results:

                    answer, sources = format_retrieved_results(
                        results
                    )

                else:

                    answer = """
I couldn't find a relevant policy or sufficient information
about this topic in the available HR documents.

I don't want to guess or provide incorrect policy information.

Please contact the HR team for clarification:

**Email:** hr.helpdesk@acruxdynamics.com
"""

            st.markdown(answer)

            if sources:
                display_sources(sources)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
