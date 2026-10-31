# ============================================================
# START DATE   : 2026-10-29
# CODE RUN DATE: 2026-10-31
# TIME TAKEN   : (library, imported)
# DESCRIPTION  : Prompt templates for the lending experiment (Bryan, Karlan, and Osman 2024). Profiles
#                are built from baseline answers only, with income and size fields shown in fixed bands.
#                Seven conditions = (goal, include alternative financing R, include intended loan use B,
#                include the 50 psychological statements P): P0, P_R, P_B, P_RB, P_RBP (goal: profit
#                gain), S_RBP (goal: repayment), L_RBP (goal: profit level under the large offer).
#                The lender's name and the study are never named in the prompt. One call per prompt;
#                SHA-256 over system + "\n----\n" + user.
# INPUT FILES  : none (imported); the 50 statements come from the baseline file's variable labels
#                (two labels are cut at 80 characters; the full wording is in the AER online appendix)
# OUTPUT FILES : none
# ============================================================
import hashlib
import re

import numpy as np
import pandas as pd

SYSTEM = ("You are an analyst who advises a microfinance lender. Use only the information provided. "
          "Answer with a single JSON object and nothing else.")
INTRO = ("Borrower of a nonprofit microfinance lender in Egypt, 2016-2017. The borrower runs a small business and "
         "has repaid at least three earlier loans from this lender.")
GOALS = {
    "P": ("The lender wants to know for which borrowers the large offer, compared with the normal offer, would "
          "raise the borrower's business profit the most."),
    "S": ("The lender wants to know for which borrowers the large offer, compared with the normal offer, would "
          "lower the chance of late repayment the most, or raise it the least."),
    # Optional (spec rev2 13.3): profit level rather than gain, same information and answer format.
    "L": "The lender wants to know which borrowers would earn the highest business profit if given the large offer.",
}
QUESTION = (
    "The lender can offer this borrower one of two loans:\n"
    "- Normal offer: a loan about twice the size of the borrower's previous loan from this lender.\n"
    "- Large offer: a loan about four times the size of the previous loan.\n"
    "Annual interest is 14-17 percent. The borrower chooses a term of 6 to 24 months (12 months is most common). "
    "Late installments are charged a penalty. A borrower who declines an offer takes no loan from this lender in "
    "this cycle, so that borrower has no late installments.\n\n"
    "Goal: {goal}\n\n"
    "Answer each question for the normal offer and for the large offer. Decimals are allowed. Months are counted "
    "from the date the lender makes the offer.\n"
    "Q1. Probability (0-100) that the borrower accepts the offer.\n"
    "Q2. If the borrower accepts, probability (0-100) that at least one monthly installment is paid late during "
    "the loan.\n"
    "Q3. Expected average monthly profit in Egyptian pounds about 20 months after the offer, summed over all "
    "businesses the owner runs at that time (profit = what the businesses earn after all expenses except payments "
    "to the owners; 0 if the owner runs no business then), whether or not the borrower accepts the offer.\n"
    "Q4. The same quantity as Q3, about 30 months after the offer.\n"
    "Q5. Does the profile state the facts needed to judge how the large offer would change this borrower's profit "
    'and repayment? Answer "sufficient" or "insufficient".\n\n'
    'Return exactly: {{"normal": {{"accept": number, "late_if_accepted": number, "profit_20m": number, '
    '"profit_30m": number}}, "large": {{"accept": number, "late_if_accepted": number, "profit_20m": number, '
    '"profit_30m": number}}, "information": "sufficient" or "insufficient"}}')
# ★ IMPORTANT: the six conditions of spec 4.4 = (goal, include R, include B, include P).
CONDITIONS = {"P0": ("P", 0, 0, 0), "P_R": ("P", 1, 0, 0), "P_B": ("P", 0, 1, 0), "P_RB": ("P", 1, 1, 0),
              "P_RBP": ("P", 1, 1, 1), "S_RBP": ("S", 1, 1, 1)}

NS = "not stated"
EDU = [((1, 2), "no formal schooling or a literacy certificate only"),
       ((3, 6), "some or completed elementary or preparatory school"),
       ((7, 10), "some or completed high school or vocational school"),
       ((11, 16), "some or completed post-secondary, college or advanced degree")]
SECTORS = [("b_sector_services", "services"), ("b_sector_retail", "retail sales"),
           ("b_sector_manufacturing", "manufacturing"), ("b_sector_agriculture", "agriculture"),
           ("b_sector_wholesale", "wholesale")]
SOURCES = [("b_other_loans_bank", "a bank"), ("b_other_loans_mfi", "another microfinance institution"),
           ("b_other_loans_informal", "an informal lender"), ("b_other_loans_familyfriends", "family or friends"),
           ("b_other_loans_rosca", "a rotating savings group (ROSCA)")]
LACKCREDIT = {1: "a major constraint", 3: "an average constraint", 2: "a minor constraint", 4: "not a constraint"}
USES = [("b_fin_loanuse_machinequip", "machinery or equipment"), ("b_fin_loanuse_inputs", "inputs or merchandise"),
        ("b_fin_loanuse_marketing", "marketing"), ("b_fin_loanuse_rentlandbuild", "renting land or buildings"),
        ("b_fin_loanuse_buylandbuild", "buying land or buildings"), ("b_fin_loanuse_newemployees", "hiring new employees"),
        ("b_fin_loanuse_paymentemployees", "payments or training for employees"),
        ("b_fin_loanuse_newbranch", "opening a new branch"), ("b_fin_loanuse_debt", "paying off other debt"),
        ("b_fin_loanuse_personal", "personal or family use"), ("b_fin_loanuse_productdvpt", "product development"),
        ("b_fin_loanuse_renovation", "renovating the business premises"), ("b_fin_loanuse_other1", "other (details not shared)")]
PRIMARY = {1: "machinery or equipment", 2: "inputs or merchandise", 3: "marketing", 4: "renting land or buildings",
           5: "buying land or buildings", 6: "hiring new employees", 7: "payments or training for employees",
           8: "opening a new branch", 9: "paying off other debt", 10: "personal or family use",
           11: "product development", 12: "renovating the business premises", 13: "other (details not shared)"}
AGREE = {1: "strongly agree", 2: "agree", 3: "neither agree nor disagree", 4: "disagree", 5: "strongly disagree"}


def band(x, cuts, labels):
    """cuts are the lower bounds of each band after the first; labels has len(cuts)+1 entries."""
    if x is None or pd.isna(x):
        return NS
    return labels[int(np.searchsorted(cuts, x, side="right"))]


def yn(v):
    return NS if v is None or pd.isna(v) or v not in (0, 1) else ("yes" if v == 1 else "no")


def x0_lines(r):
    sec = [n for c, n in SECTORS if r.get(c) == 1]
    edu = next((t for (a, b), t in EDU if pd.notna(r.get("b_education")) and a <= r["b_education"] <= b), NS)
    reg = yn(r.get("b_registration"))
    prof = r.get("b_monthly_profits")
    prof_b = NS if pd.isna(prof) else ("zero or a loss" if prof <= 0 else
                                       band(prof, [1000, 2500, 5000, 10000, 20000],
                                            ["1-999", "1,000-2,499", "2,500-4,999", "5,000-9,999", "10,000-19,999", "20,000 or more"]))
    return [
        f"- Owner age: {band(r.get('b_age'), [30, 40, 50, 60], ['under 30', '30-39', '40-49', '50-59', '60 or older'])}",
        f"- Owner gender: {'female' if r.get('b_gender') == 1 else 'male' if r.get('b_gender') == 0 else NS}",
        f"- Owner education: {edu}",
        f"- Business activity: {', '.join(sec) if sec else NS}",
        f"- Years in business: {band(r.get('b_experience'), [3, 6, 11, 21], ['under 3', '3-5', '6-10', '11-20', '21 or more'])}",
        f"- Business has staff: {yn(r.get('b_staff_yesno'))}",
        f"- Business is registered: {reg}",
        f"- Previous loan from this lender (Egyptian pounds): "
        f"{band(r.get('b_size_lastloan'), [3000, 5000, 7500, 10000, 15000], ['under 3,000', '3,000-4,999', '5,000-7,499', '7,500-9,999', '10,000-14,999', '15,000 or more'])}",
        f"- Monthly profit of the main business before the offer (Egyptian pounds): {prof_b}",
        f"- Monthly revenue of the main business before the offer (Egyptian pounds): "
        f"{band(r.get('b_monthly_revenue'), [2500, 5000, 10000, 25000, 50000, 100000], ['under 2,500', '2,500-4,999', '5,000-9,999', '10,000-24,999', '25,000-49,999', '50,000-99,999', '100,000 or more'])}",
    ]


def r_lines(r):
    has = yn(r.get("b_loans"))
    line = f"- Has other loans now: {has}"
    if has == "yes":
        src = [n for c, n in SOURCES if r.get(c) == 1]
        line += f" (from: {', '.join(src) if src else 'source not stated'})"
    return [line,
            f"- Lack of credit as a constraint on current operations: {LACKCREDIT.get(r.get('b_fin_lackcredit'), NS)}",
            f"- Lists lack of credit among the constraints that will limit the business over the long term: "
            f"{yn(r.get('b_growconstr_credit'))}"]


def b_lines(r):
    vals = [r.get(c) for c, _ in USES]
    if all(pd.isna(v) for v in vals):
        uses = NS
    else:
        u = [n for (c, n) in USES if r.get(c) == 1]
        uses = ", ".join(u) if u else "none of the listed uses"
    return [f"- Planned uses of a larger loan (owner's answer, several allowed): {uses}",
            f"- Primary planned use: {PRIMARY.get(r.get('b_fin_primaryloanuse'), NS)}"]


def p_lines(r, statements):
    return [f"- {s}: {AGREE.get(r.get(f'b_psy{k}'), 'no answer')}" for k, s in enumerate(statements, start=1)]


def user_prompt(r, cond, statements):
    goal, wr, wb, wp = {**CONDITIONS, **OPTIONAL_CONDITIONS}[cond]
    parts = [INTRO, "", "Borrower profile"] + x0_lines(r)
    if wr:
        parts += ["", "Other financing"] + r_lines(r)
    if wb:
        parts += ["", "Planned use of a larger loan"] + b_lines(r)
    if wp:
        parts += ["", "How much the owner agrees with each statement"] + p_lines(r, statements)
    return "\n".join(parts) + "\n\n" + QUESTION.format(goal=GOALS[goal])


OPTIONAL_CONDITIONS = {"L_RBP": ("L", 1, 1, 1)}


def statements_from_labels(lab):
    out = []
    for k in range(1, 51):
        m = re.match(rf"^{k}\.\s*(.+)$", (lab.get(f"b_psy{k}") or "").strip())
        if not m:
            raise SystemExit(f"b_psy{k}: label does not start with '{k}.'")
        out.append(m.group(1).strip())
    return out


def norm(s):
    return re.sub(r"[^a-z]", "", s.lower().replace("’", "'"))


def full_wording_from_appendix(statements, appendix_text):
    """Replace labels cut at 80 characters by the appendix wording (Survey Questions, Baseline list)."""
    block = appendix_text.split("Psychometric Questions")[-1].split("Baseline", 1)[1].split("Follow Up 1")[0]
    lines = [ln.strip() for ln in block.split("\n") if ln.strip() and not ln.strip().isdigit()]
    items = []
    for ln in lines:                          # wrapped items continue on an indented line
        if items and (ln[0].islower() or items[-1].endswith("the")):
            items[-1] += " " + ln
        else:
            items.append(ln)
    if len(items) != 50:
        raise ValueError(f"parsed {len(items)} appendix items, expected 50")
    final, truncated = [], []
    for k, (a, s) in enumerate(zip(items, statements), start=1):
        if norm(a) == norm(s):
            final.append(s)
        elif norm(a).startswith(norm(s)):
            final.append(a.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'"))
            truncated.append(k)
        else:
            raise ValueError(f"statement {k} differs from the appendix")
    return final, truncated


def prompt_records(baseline, statements, conds):
    out = []
    for cond in conds:
        for r in baseline.to_dict("records"):
            user = user_prompt(r, cond, statements)
            out.append({"key": f"{int(r['client_key'])}|{cond}", "cond": cond, "system": SYSTEM, "user": user,
                        "prompt_sha256": hashlib.sha256((SYSTEM + "\n----\n" + user).encode()).hexdigest()})
    return out
