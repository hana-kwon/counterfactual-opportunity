# ============================================================
# START DATE   : 2026-10-27
# CODE RUN DATE: 2026-10-31
# TIME TAKEN   : (library, imported)
# DESCRIPTION  : Prompt templates for the grant competition (YouWiN!). Applicant profiles are built from
#                pre-assignment application fields only. Two information sets: X = applicant and business
#                facts; XR = X plus the financing facts on the form (current loans for existing firms;
#                stated obstacle to starting for new firms). Two tasks per applicant and information set:
#                F (outside financing without the grant) and LA (employment in three years without and
#                with the grant). One call per prompt; the system and user text are hashed with
#                SHA-256 over system + "\n----\n" + user.
# INPUT FILES  : none (imported); profiles come from the replication data (full path) or the
#                synthetic generator (demo path)
# OUTPUT FILES : none
# ============================================================
import hashlib

import numpy as np
import pandas as pd

SECTOR = {11: "market trader", 12: "street vendor", 13: "shop", 14: "street kiosk", 15: "other retail trade",
          16: "market stall", 17: "street kiosk or street sales", 18: "shop or restaurant",
          19: "food preparation or restaurant", 20: "hair and beauty care", 21: "other personal services",
          30: "tailoring, dressmaking or shoe making", 31: "furniture", 32: "crafts", 33: "other manufacturing",
          40: "repair of electronics", 41: "repair of vehicles", 42: "other repair services",
          50: "IT and computer services", 51: "accounting services", 52: "legal services", 53: "medical services",
          54: "other professional services", 60: "taxi driver", 61: "other transportation services",
          70: "carpentry", 71: "bricklaying", 72: "electrical work", 73: "plumbing", 74: "other construction work",
          80: "agricultural production", 90: "other industries"}
# Codes from the baseline questionnaire (BaselineDataSheet_Dec62011.pdf, "Marital Status").
MARITAL = {1: "married or living with a partner", 2: "divorced or separated", 3: "widowed", 4: "never married"}
SYSTEM = ("You are an analyst who evaluates applicants to a business support program in Nigeria. "
          "Use only the information provided. Answer with a single JSON object and nothing else.")
CONTEXT = ("Applicant to the YouWiN! business plan competition, Nigeria, application made in late 2011. "
           "The competition can award this applicant a grant averaging about US$50,000, paid in installments "
           "during the following year.")
F_Q = ("Question: What is the probability (0-100) that this applicant obtains at least US$10,000 of outside "
       "financing (a loan or an equity investment) within the next 12 months if it does not receive this grant?\n\n"
       'Return exactly: {"p_outside_financing": number}')
LA_Q = ("Questions:\n"
        "Q1. Expected total employment in the applicant's firm three years from now, counting the owner, if the "
        "applicant does not receive the grant (0 if no firm is operating).\n"
        "Q2. Expected total employment in the applicant's firm three years from now, counting the owner, if the "
        "applicant receives the grant.\n"
        "Q3. Does the profile state the facts needed to judge how much the grant would change employment? "
        'Answer "sufficient" or "insufficient".\n\n'
        'Return exactly: {"emp_without": number, "emp_with": number, "information": "sufficient" or "insufficient"}')


def txt(v, n=40):
    s = "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v).strip()
    return s[:n]


def yn(v):
    return None if pd.isna(v) else ("yes" if v == 1 else "no")


def profile(r, with_r):
    edu = ("Masters or PhD" if r.postgraduate == 1 else "university degree" if r.university == 1 else
           "polytechnic or college of education" if r.polytechnic == 1 else "secondary or primary school")
    L = [f"- Owner: {'female' if r.female == 1 else 'male'}, age {int(r.age)}, {MARITAL.get(r.marital_status, 'marital status not stated')}, "
         f"{int(r.number_of_children)} children, education: {edu}",
         f"- State of residence: {r.state_residence}",
         f"- Application: {'existing business' if r.existing == 1 else 'new business'}; sector: {SECTOR.get(r.business_sector, 'not stated')}",
         f"- Training taken: business accounting {yn(r.business_accounting_training)}, marketing {yn(r.marketing_training)}, "
         f"computer skills {yn(r.computer_skills)}, vocational {yn(r.vocational_training)}",
         f"- Currently employed outside the firm: {yn(r.otherwise_employed)}; has lived or worked abroad: {yn(r.lived_worked_abroad)}",
         f"- Household owns: {int(r.have_computers)} computer(s), {int(r.have_cars_and_other_vehicles)} car(s); internet at home: {yn(r.have_internet)}"]
    if pd.notna(r.years_of_experience):
        L.append(f"- Years of work experience: {int(r.years_of_experience)}")
    if r.existing == 1:
        L.append(f"- Business as stated on the application: years in business \"{txt(r.numyearsinbusiness)}\"; "
                 f"employees \"{txt(r.number_employees)}\"; past turnover \"{txt(r.turnover)}\"")
        if pd.notna(r.business_registered):
            L.append(f"- Registered with the Corporate Affairs Commission: {yn(r.business_registered)}")
    if with_r:
        # ★ IMPORTANT: the financing facts on the form (the "R" condition). Loans were asked mainly of
        # existing businesses; new-business applicants were asked what had prevented them from starting.
        loans = [(n, getattr(r, c)) for n, c in [("a bank", "loan_from_bank"), ("a microfinance institution",
                 "loan_from_microfinance_organisat"), ("a cooperative", "loan_from_cooperative")]]
        if any(pd.notna(v) for _, v in loans):
            L.append("- Current loans: " + "; ".join(f"from {n}: {yn(v) or 'not stated'}" for n, v in loans))
        else:
            L.append("- Current loans: not stated")
        obst = [n for n, c in [("lack of funds", "lack_funding"), ("lack of skills", "lack_skills"),
                               ("lack of business experience", "lack_experience"), ("lack of interest", "lack_interest")]
                if getattr(r, c) == 1]
        L.append("- What had prevented the applicant from starting or expanding a business: "
                 + (", ".join(obst) if obst else "none of the listed reasons"))
    return CONTEXT + "\n\nApplicant profile\n" + "\n".join(L)




def prompt_records(df):
    """All four prompts per applicant, in the order they were sent: X then XR, F then LA."""
    out = []
    for r in df.itertuples(index=False):
        for cond in ("X", "XR"):
            prof = profile(r, cond == "XR")
            for task, q in (("F", F_Q), ("LA", LA_Q)):
                user = prof + "\n\n" + q
                out.append({"key": f"{int(r.uid)}|{cond}|{task}", "system": SYSTEM, "user": user,
                            "prompt_sha256": hashlib.sha256((SYSTEM + "\n----\n" + user).encode()).hexdigest()})
    return out
