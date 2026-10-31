# ============================================================
# START DATE   : 2026-10-29
# CODE RUN DATE: 2026-10-31
# TIME TAKEN   : ~1 min (reads the 180 MB merged file)
# DESCRIPTION  : Full path only. Builds the lending-experiment design rows from the Bryan, Karlan, and
#                Osman (2024) clean data: profit rows (follow-up rounds 1 and 2, monthly profit summed over
#                all businesses) and one lender row per client (perfect repayment record), the
#                randomization block (loan officer x batch), and the original study's stored machine-
#                learning effect proxy (benchmark). Rebuilds the 7 x 1,004 prompts and checks them against
#                the SHA-256 stored with the cached model answers.
# INPUT FILES  : data/raw/**/16-ABA-All-Merged.dta, 17-ABA-penalty-clean.dta, 20-ABA-ML-group.dta,
#                10-ABA-baseline-clean.dta (openICPSR project 192297, version V1; see data/README.md);
#                optional data/raw/**/aer_online_appendix.pdf (full wording of two statements);
#                data/model_outputs/egypt_answers_claude-sonnet-5-5.csv
# OUTPUT FILES : <run>/work/lending_{prof,cl,bench}.pkl, <run>/work/lending_prompts.jsonl (row level);
#                <run>/04_egypt_build/{design_check.json, prompt_hash_check.json}
# ============================================================
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, RAW, args, find_raw, step_dir, write_json  # noqa: E402
import prompts_lending as PL  # noqa: E402

MERGED_COLS = ["client_key", "round", "random_cohort", "loid_baseline", "treatment", "e_profits_all",
               "mis_perfect_repayment", "perfect_repayment"]


def load_design():
    """★ KEY FUNCTION: profit rows and one lender row per client, as in the locked analysis."""
    with pd.io.stata.StataReader(find_raw("16-ABA-All-Merged.dta")) as r:
        m = r.read(convert_categoricals=False, columns=MERGED_COLS)
    m = m[m["round"] != 0].copy()
    # Randomization block = loan officer x batch (BKO p. 2832); random_cohort is constant within client.
    coh = m.dropna(subset=["random_cohort"]).groupby("client_key").random_cohort.first()
    m["block"] = pd.factorize(pd.Series(list(zip(m.loid_baseline, m.client_key.map(coh))), index=m.index))[0]
    prof = m[m.e_profits_all.notna()][["client_key", "loid_baseline", "block", "treatment", "round",
                                       "e_profits_all"]].reset_index(drop=True)
    # ★ IMPORTANT: filled-in panel rows carry mean-imputed lender outcomes (01_Cleaning.do 898-900); every
    # client has a non-imputed row and the value is constant within client (the Table 3 value).
    cl = (m[m.mis_perfect_repayment == 0].sort_values(["client_key", "round"])
          .groupby("client_key", as_index=False).first())
    cl = cl[["client_key", "loid_baseline", "block", "treatment", "perfect_repayment"]].reset_index(drop=True)
    assert len(cl) == 1004 and len(prof) == 1922
    return prof, cl


def load_ml_benchmark():
    """The original study's stored random-forest effect proxy RF_HTE_S, by client (higher = larger gain)."""
    with pd.io.stata.StataReader(find_raw("20-ABA-ML-group.dta")) as r:
        g = r.read(convert_categoricals=False, columns=["client_key", "RF_HTE_S"])
    s = g.dropna().drop_duplicates("client_key").set_index("client_key").RF_HTE_S.astype(float)
    return s.rename("ML").reset_index()


def main():
    a = args("Build the lending-experiment design rows from the replication package")
    out = step_dir(a, "04_egypt_build")
    prof, cl = load_design()
    bench = load_ml_benchmark()
    prof.to_pickle(a.work / "lending_prof.pkl")
    cl.to_pickle(a.work / "lending_cl.pkl")
    bench.to_pickle(a.work / "lending_bench.pkl")
    write_json(out / "design_check.json", {
        "clients": len(cl), "profit_rows": len(prof), "clients_with_profit": int(prof.client_key.nunique()),
        "larger_offer_clients": int(cl.treatment.sum()), "loan_officers": int(cl.loid_baseline.nunique()),
        "randomization_blocks": int(cl.block.nunique()), "ml_benchmark_clients": len(bench)})

    # Prompts: rebuild from baseline answers and compare with the stored hashes.
    with pd.io.stata.StataReader(find_raw("10-ABA-baseline-clean.dta")) as rd:
        b = rd.read(convert_categoricals=False)
        lab = rd.variable_labels()
    b = b.sort_values("client_key").reset_index(drop=True)
    statements = PL.statements_from_labels(lab)
    appendix = sorted(Path(dp) / f for dp, _, fs in os.walk(RAW, followlinks=True) for f in fs
                      if f.lower().startswith("aer_online_appendix") and f.lower().endswith(".pdf"))
    full_wording = False
    if appendix and shutil.which("pdftotext"):
        txt = subprocess.run(["pdftotext", "-layout", str(appendix[0]), "-"], capture_output=True, text=True,
                             check=True).stdout
        statements, truncated = PL.full_wording_from_appendix(statements, txt)
        full_wording = True
    prompts = PL.prompt_records(b, statements, list(PL.CONDITIONS) + list(PL.OPTIONAL_CONDITIONS))
    with open(a.work / "lending_prompts.jsonl", "w") as fh:
        for p in prompts:
            fh.write(json.dumps(p) + "\n")
    ans = pd.read_csv(DATA / "model_outputs" / "egypt_answers_claude-sonnet-5-5.csv", dtype={"answer": str})
    stored = dict(zip(ans.client_key.astype(str) + "|" + ans.condition, ans.prompt_sha256))
    by_cond = {}
    for p in prompts:
        c = by_cond.setdefault(p["cond"], [0, 0])
        c[0] += 1
        c[1] += stored.get(p["key"]) == p["prompt_sha256"]
    write_json(out / "prompt_hash_check.json", {
        "full_statement_wording_from_appendix": full_wording,
        "note": ("Conditions with the 50 statements (P_RBP, S_RBP, L_RBP) match only when the AER online "
                 "appendix PDF is placed in data/raw/, because two variable labels are cut at 80 characters."),
        "by_condition": {k: {"prompts": v[0], "hash_matches": v[1]} for k, v in by_cond.items()}})
    print("design:", len(cl), "clients,", len(prof), "profit rows; prompt hash matches:",
          {k: v[1] for k, v in by_cond.items()})


if __name__ == "__main__":
    main()
