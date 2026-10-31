# ============================================================
# START DATE   : 2026-10-27
# CODE RUN DATE: 2026-10-31
# TIME TAKEN   : < 1 min
# DESCRIPTION  : Full path only. Links the YouWiN! baseline and third follow-up by uid, rebuilds the
#                outcomes and randomization strata exactly as McKenzie's MainAnalysis_AERreplication.do,
#                checks the link by reproducing the AER Table 4 employment effects, rebuilds every
#                applicant prompt and checks it against the SHA-256 stored with the cached model answer,
#                and writes the analysis frame used by steps 05 and 06.
# INPUT FILES  : data/raw/**/BaselineandFirstFollowup.dta, data/raw/**/ThirdFollowup.dta
#                (openICPSR project 113101, version V1; see data/README.md)
#                data/model_outputs/youwin_answers_claude-sonnet-5-5.csv
# OUTPUT FILES : <run>/work/grant_frame.pkl, <run>/work/grant_prompts.jsonl (row level, not shared);
#                <run>/03_youwin_build/{aer_table4_check.csv, prompt_hash_check.json}
# ============================================================
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, args, find_raw, step_dir, write_json  # noqa: E402
import prompts_grant  # noqa: E402

# Baseline application fields shown in the prompt; they are also the T-learner's features.
FEATURES = ["female", "age", "marital_status", "number_of_children", "postgraduate", "university",
            "polytechnic", "state_residence", "existing", "business_sector",
            "business_accounting_training", "marketing_training", "computer_skills",
            "vocational_training", "otherwise_employed", "lived_worked_abroad", "have_computers",
            "have_cars_and_other_vehicles", "have_internet", "years_of_experience",
            "numyearsinbusiness", "number_employees", "turnover", "business_registered",
            "loan_from_bank", "loan_from_microfinance_organisat", "loan_from_cooperative",
            "lack_funding", "lack_skills", "lack_experience", "lack_interest"]
# do-file 2954-2961: Nigeria CPI by third follow-up interview month, base Nov 2009 = 100
CPI = {10: 164.13, 11: 163.10, 12: 164.40, 1: 165.77}


def t_outcome(t):
    # ★ CORE LOGIC: line-by-line port of the do-file (lines 290-296, 1022-1025, 1039-1042, 1183-1186).
    op = pd.Series(np.where(t.t_sc == 1, 1.0, 0.0), index=t.index)
    op[t.t_sc.isna()] = np.nan
    op[op.isna() & (t.t_operate == 1)] = 1
    op[op.isna() & (t.t_operate == 2)] = 0
    op[(t.t_a3 == 1) & op.isna()] = 1
    op[(t.t_a3 == 2) & op.isna()] = 0
    wage = t.t_ef3_1.copy()
    wage[t.t_ef3_1.isin([998, 999])] = np.nan
    wage[op == 0] = 0
    cas = t.t_ef3_2.copy()
    cas[t.t_ef3_2 == 999] = 0
    cas[op == 0] = 0
    tot = op + wage + cas
    tot[op == 0] = 0
    m = tot.isna() & (op == 1) & (t.t_ef3_5 < 998)
    tot[m] = t.t_ef3_5[m] + 1
    m = tot.isna() & (t.t_a4 < 998)
    tot[m] = t.t_a4[m]
    return op, tot


def ihs(x):
    return np.log(x + np.sqrt(x ** 2 + 1))


def profit_outcome(d):
    """★ KEY FUNCTION: third follow-up real profits as the do-file builds them, inverse hyperbolic sine."""
    m3b = pd.to_numeric(d["t_resp_10"].astype(str).str[3:5], errors="coerce")       # do-file 90-91
    prof = pd.to_numeric(d["t_bf9"], errors="coerce").mask(lambda s: s.isin([998, 9998]))
    sales = pd.to_numeric(d["t_bf5"], errors="coerce").mask(lambda s: s == 998)
    cpi = m3b.map(CPI)
    cpi = cpi.where(cpi.notna() | ~(m3b >= 2), 166.90)
    cpi = cpi.where(cpi.notna() | ~(sales.notna() | prof.notna()), 166.90)
    # ★ IMPORTANT: real naira first (2970-2972), then zeros for closed firms (2983, 2996-2998).
    prof = prof * (140.01 / cpi)
    closed = d["t_operatefirm"] == 0
    prof = prof.where(~(prof.isna() & closed), 0.0)
    return ihs(prof)                                                                  # 2984


def itt(d, y):
    """areg y assigntreat, a(strata) robust."""
    d = d.dropna(subset=[y, "assigntreat", "strata"])
    yd = d[y] - d.groupby("strata")[y].transform("mean")
    xd = d.assigntreat - d.groupby("strata").assigntreat.transform("mean")
    b = (xd * yd).sum() / (xd ** 2).sum()
    e = yd - b * xd
    n, k = len(d), 1 + d.strata.nunique()
    se = np.sqrt((xd ** 2 * e ** 2).sum() / (xd ** 2).sum() ** 2 * n / (n - k))
    return b, se, len(d), d.loc[d.assigntreat == 0, y].mean()


def first_number(v):
    """Free-text application fields ('about 3 years', '5-6') carry a number; keep it, else missing."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    m = re.search(r"\d+(?:\.\d+)?", str(v).replace(",", ""))
    return float(m.group()) if m else np.nan


def feature_matrix(d):
    """Same variables the prompt showed. A text field with no number at all (state of residence) enters
    as a category code. Median imputation over the experimental sample."""
    X = pd.DataFrame(index=d.index)
    for c in FEATURES:
        v = d[c]
        if v.dtype == object:
            col = v.map(first_number)
            if col.isna().mean() > 0.99:
                col = pd.Series(pd.factorize(v)[0], index=d.index).astype(float)
            X[c] = col
        else:
            X[c] = pd.to_numeric(v, errors="coerce")
    med = X.median(numeric_only=True)
    return X.fillna(med).fillna(-1.0)


def main():
    a = args("Build the YouWiN! analysis frame from the replication package")
    out = step_dir(a, "03_youwin_build")
    b = pd.read_stata(find_raw("BaselineandFirstFollowup.dta"), convert_categoricals=False)
    t = pd.read_stata(find_raw("ThirdFollowup.dta"), convert_categoricals=False)
    d = b.merge(t, on="uid", how="left", validate="one_to_one")
    d["t_operatefirm"], d["t_totalemp1"] = t_outcome(d)
    # strata = group(region existing female), defined on the experimental sample (do-file 310-312)
    d["strata"] = d.groupby(["region", "existing", "female"]).ngroup()
    d.loc[d.experimentalsample != 1, "strata"] = np.nan

    rows = []
    for ex, lab, target_b, target_c in [(0, "new firm applicants", 5.2, 3.7), (1, "existing firm applicants", 4.4, 5.6)]:
        s = d[(d.experimentalsample == 1) & (d.existing == ex)]
        bb, se, n, cm = itt(s, "t_totalemp1")
        rows.append({"group": lab, "n": n, "itt": round(bb, 3), "se": round(se, 3), "control_mean": round(cm, 3),
                     "aer_table4_itt": target_b, "aer_table4_control_mean": target_c})
    pd.DataFrame(rows).to_csv(out / "aer_table4_check.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))

    exp = d[d.experimentalsample == 1].copy()
    # The ordinary-winner experiment: 1,841 applicants (same set as group <= 2 in the do-file).
    assert len(exp) == 1841 and set(exp.uid) == set(d[d["group"] <= 2].uid)

    # Prompts: rebuild and compare with the hashes stored with the cached answers.
    prompts = prompts_grant.prompt_records(exp)
    with open(a.work / "grant_prompts.jsonl", "w") as fh:
        for p in prompts:
            fh.write(json.dumps(p) + "\n")
    ans = pd.read_csv(DATA / "model_outputs" / "youwin_answers_claude-sonnet-5-5.csv", dtype={"answer": str})
    stored = dict(zip(ans.uid.astype(str) + "|" + ans.condition + "|" + ans.task, ans.prompt_sha256))
    match = sum(stored.get(p["key"]) == p["prompt_sha256"] for p in prompts)
    write_json(out / "prompt_hash_check.json", {"prompts_rebuilt": len(prompts), "answers": len(stored),
                                                "hash_matches": match})
    print(f"prompt hashes: {match} of {len(prompts)} rebuilt prompts match the cached answers")

    exp["uid"] = exp.uid.astype(int)
    exp["t_IHSprofits"] = profit_outcome(exp)
    exp = exp.sort_values("uid").reset_index(drop=True)
    X = feature_matrix(exp)
    frame = exp[["uid", "strata", "assigntreat", "existing", "t_totalemp1", "t_IHSprofits", "totalmarks"]].copy()
    for c in FEATURES:
        frame["x_" + c] = X[c].to_numpy(float)
    frame.to_pickle(a.work / "grant_frame.pkl")
    print("grant frame", frame.shape, "responders", int(frame.t_totalemp1.notna().sum()))


if __name__ == "__main__":
    main()
