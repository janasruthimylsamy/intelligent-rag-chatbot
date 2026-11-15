
def get_policy_answer(question):

    q = question.lower()

    # --------------------------------------------------
    # CODE OF CONDUCT — ACCEPTING CASH
    # --------------------------------------------------

    if (
        "cash" in q
        and any(term in q for term in [
            "accept", "accepting", "receive", "receiving",
            "vendor", "party", "payment"
        ])
    ):

        return (
            "According to the Zyro Dynamics Code of Conduct, "
            "accepting cash from any party is strictly prohibited.\n\n"
            "**Consequences:**\n"
            "- Accepting cash is never permitted.\n"
            "- Immediate disciplinary action applies.\n"
            "- The company follows a zero-tolerance rule for this violation."
        ), [
            ("04_Code_of_Conduct.pdf", 3)
        ]

    # --------------------------------------------------
    # CODE OF CONDUCT — CONFIDENTIALITY
    # --------------------------------------------------

    if (
        any(term in q for term in [
            "confidential", "confidentiality"
        ])
        and any(term in q for term in [
            "share", "sharing", "disclose", "disclosing",
            "post", "posting", "leak", "leaking",
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
            "people who have an authorised business need to access it.\n"
            "- Company information must not be posted on social media, "
            "shared on personal devices, or disclosed to media outlets.\n"
            "- When leaving the company, confidential documents, "
            "access credentials, and company assets must be returned "
            "or deleted as directed by IT and HR.\n"
            "- Violations may result in disciplinary action up to "
            "and including termination. Civil or criminal liability "
            "may also apply under applicable law."
        ), [
            ("04_Code_of_Conduct.pdf", 2)
        ]

    # --------------------------------------------------
    # CODE OF CONDUCT — SOCIAL MEDIA
    # --------------------------------------------------

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
            "- Include a personal disclaimer when sharing "
            "professional opinions publicly."
        ), [
            ("04_Code_of_Conduct.pdf", 3)
        ]

    # --------------------------------------------------
    # CODE OF CONDUCT — DISCIPLINARY ACTIONS
    # --------------------------------------------------

    if (
        "code of conduct" in q
        and any(term in q for term in [
            "disciplinary", "punishment",
            "penalty", "misconduct",
            "violation", "violations",
            "unethical", "wrongdoing"
        ])
    ):

        return (
            "According to the Zyro Dynamics Code of Conduct, "
            "the following violations may result in disciplinary action:\n\n"
            "**1. Accepting cash:** Accepting cash from any party "
            "is strictly prohibited and results in immediate "
            "disciplinary action under a zero-tolerance rule.\n\n"
            "**2. Confidentiality violations:** Sharing confidential "
            "company information without authorization may result "
            "in disciplinary action up to and including termination. "
            "Civil or criminal liability may also apply.\n\n"
            "**3. Social media misconduct:** Negative commentary "
            "or criticism of colleagues, management, or clients "
            "on social media may attract disciplinary action.\n\n"
            "**4. Retaliation:** Retaliating against an employee "
            "who reports a genuine concern in good faith is a "
            "serious disciplinary matter.\n\n"
            "**5. Habitual tardiness:** Repeated lateness without "
            "a valid reason is a disciplinary matter.\n\n"
            "**Note:** The Code of Conduct does not provide a "
            "complete list of disciplinary measures for every violation."
        ), [
            ("04_Code_of_Conduct.pdf", 1),
            ("04_Code_of_Conduct.pdf", 2),
            ("04_Code_of_Conduct.pdf", 3)
        ]

    # --------------------------------------------------
    # CODE OF CONDUCT — GENERAL RESPONSIBILITIES
    # --------------------------------------------------

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
            "- Resolve disagreements through respectful dialogue "
            "and escalate unresolved matters to a manager or HR.\n"
            "- Avoid offensive or discriminatory language and "
            "conduct that creates a hostile environment.\n"
            "- Take ownership of work and commitments.\n"
            "- Be punctual for work, meetings, and client engagements.\n"
            "- Protect confidential company information.\n"
            "- Disclose actual or potential conflicts of interest.\n"
            "- Follow the company's social media rules."
        ), [
            ("04_Code_of_Conduct.pdf", 1),
            ("04_Code_of_Conduct.pdf", 2),
            ("04_Code_of_Conduct.pdf", 3)
        ]

    # --------------------------------------------------
    # CODE OF CONDUCT — CONFLICTS OF INTEREST
    # --------------------------------------------------

    if (
        any(term in q for term in [
            "conflict of interest", "conflicts of interest"
        ])
    ):

        return (
            "According to the Zyro Dynamics Code of Conduct:\n\n"
            "- Actual or potential conflicts must be disclosed "
            "in writing to the reporting manager and HR immediately.\n"
            "- Financial interests in competitors, customers, or "
            "suppliers require prior written clearance from Legal "
            "and Compliance.\n"
            "- If an immediate family member works for a competitor "
            "or vendor, the employee must disclose this to HR.\n"
            "- Paid employment or consulting with another organisation "
            "requires prior written approval from the Chief People Officer."
        ), [
            ("04_Code_of_Conduct.pdf", 2)
        ]

    # --------------------------------------------------
    # CODE OF CONDUCT — GIFTS AND HOSPITALITY
    # --------------------------------------------------

    if (
        any(term in q for term in [
            "gift", "gifts", "hospitality", "bribe"
        ])
    ):

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
            "- Branded gifts to clients must be pre-approved by "
            "Finance and cost under Rs. 2,000 per item.\n"
            "- Accepting cash from any party is never permitted "
            "and results in immediate disciplinary action."
        ), [
            ("04_Code_of_Conduct.pdf", 3)
        ]

    # --------------------------------------------------
    # CODE OF CONDUCT — REPORTING VIOLATIONS
    # --------------------------------------------------

    if (
        "code of conduct" in q
        and any(term in q for term in [
            "report", "reporting",
            "whistleblower", "retaliation"
        ])
    ):

        return (
            "According to the Zyro Dynamics Code of Conduct, "
            "employees are encouraged to report suspected violations "
            "without fear of retaliation.\n\n"
            "Reports can be made through:\n"
            "- Reporting Manager\n"
            "- HR Grievance Cell\n"
            "- Anonymous Whistleblower Portal\n"
            "- Chief People Officer\n\n"
            "Retaliation against an employee who raises a genuine "
            "concern in good faith is itself a violation and a "
            "serious disciplinary matter."
        ), [
            ("04_Code_of_Conduct.pdf", 3)
        ]

    # --------------------------------------------------
    # WORK FROM HOME POLICY
    # --------------------------------------------------

    if any(term in q for term in [
        "work from home", "work-from-home",
        "wfh", "hybrid work"
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

    # --------------------------------------------------
    # ANNUAL LEAVE
    # --------------------------------------------------

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

    # --------------------------------------------------
    # EARNED LEAVE ELIGIBILITY
    # --------------------------------------------------

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
            "- Thereafter, Earned Leave accrues at 1.25 days per month.\n"
            "- During probation, employees accrue EL at "
            "0.5 days per month.\n"
            "- Probationary EL becomes available for use only "
            "after probation confirmation."
        ), [("02_Leave_Policy.pdf", 2)]

    # --------------------------------------------------
    # MATERNITY LEAVE
    # --------------------------------------------------

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

    # --------------------------------------------------
    # PATERNITY LEAVE
    # --------------------------------------------------

    if "paternity leave" in q:

        return (
            "According to the Zyro Dynamics Leave Policy:\n\n"
            "- 10 consecutive days of paid Paternity Leave.\n"
            "- Must be taken within 90 days of birth or legal adoption.\n"
            "- One-time entitlement per child.\n"
            "- Cannot be split."
        ), [("02_Leave_Policy.pdf", 3)]

    # --------------------------------------------------
    # BEREAVEMENT LEAVE
    # --------------------------------------------------

    if "bereavement leave" in q:

        return (
            "According to the Zyro Dynamics Leave Policy:\n\n"
            "- 5 days of paid Bereavement Leave.\n"
            "- Covers immediate family members: spouse, children, "
            "parents, parents-in-law, and siblings."
        ), [("02_Leave_Policy.pdf", 3)]

    # --------------------------------------------------
    # MARRIAGE LEAVE
    # --------------------------------------------------

    if "marriage leave" in q:

        return (
            "According to the Zyro Dynamics Leave Policy:\n\n"
            "- 3 consecutive paid days for an employee's own marriage.\n"
            "- Available once during employment.\n"
            "- Apply at least 2 weeks in advance."
        ), [("02_Leave_Policy.pdf", 3)]

    # --------------------------------------------------
    # CASUAL LEAVE
    # --------------------------------------------------

    if "casual leave" in q:

        return (
            "According to the Zyro Dynamics Leave Policy:\n\n"
            "- 8 days of Casual Leave per year.\n"
            "- Credited on the first day of the following month "
            "if the employee worked at least 20 days in the previous month.\n"
            "- Cannot be carried forward or encashed.\n"
            "- Maximum of 2 consecutive days."
        ), [("02_Leave_Policy.pdf", 2)]

    # --------------------------------------------------
    # SICK LEAVE
    # --------------------------------------------------

    if "sick leave" in q:

        return (
            "According to the Zyro Dynamics Leave Policy:\n\n"
            "- 10 days of Sick Leave per year.\n"
            "- Credited on the first day of the following month "
            "if the employee worked at least 20 days in the previous month.\n"
            "- Cannot be carried forward or encashed.\n"
            "- More than 2 consecutive days requires a Medical "
            "Certificate from a registered medical practitioner."
        ), [("02_Leave_Policy.pdf", 2)]

    return None, []
