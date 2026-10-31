# ============================================================
# START DATE   : 2026-10-31
# CODE RUN DATE: 2026-10-31
# TIME TAKEN   : < 1 min
# DESCRIPTION  : Full mode: collects the numbers this run produced for every proposal number in
#                config/proposal_numbers.json and checks (a) that they round to the printed value and
#                (b) how far they are from the full-precision value of the authors' original run.
#                Demo mode: one table of estimate vs known truth for the synthetic experiments.
#                Both modes: the hypothetical-firm cells quoted in the proposal (Figure 1).
# INPUT FILES  : outputs of steps 01, 05, 06, 07 (and 02, 08 in demo mode); config/proposal_numbers.json
# OUTPUT FILES : <run>/10_compare_reference/{comparison.csv, summary.json}
# ============================================================
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CONFIG, args, step_dir, write_json  # noqa: E402


def produced_full(run):
    pc = pd.read_csv(run / "05_grant_policy_value" / "policy_contrasts.csv").set_index("rule")
    it = pd.read_csv(run / "06_grant_heterogeneity" / "interactions.csv")
    db = pd.read_csv(run / "06_grant_heterogeneity" / "difference_bootstrap.csv").set_index("existing")
    pv = pd.read_csv(run / "07_lending_evaluation" / "policy_values.csv")
    cf = pd.read_csv(run / "07_lending_evaluation" / "confirmatory.csv").set_index("contrast")
    ov = pd.read_csv(run / "07_lending_evaluation" / "overlap.csv")
    sm = json.loads((run / "07_lending_evaluation" / "summary.json").read_text())
    m = pc.loc["Claude A, XR"]
    ai = it[(it.existing == 1) & (it.score == "A_XR") & (it.outcome == "t_totalemp1")].iloc[0]
    jd = it[(it.existing == 1) & (it.score == "totalmarks") & (it.outcome == "t_totalemp1")].iloc[0]
    pr = pv[(pv.rule == "P_RBP:PI") & (pv.versus == "random") & (pv.outcome == "profit")].iloc[0]
    ml = pv[(pv.rule == "BKO:ML") & (pv.versus == "random")].iloc[0]
    h1 = cf.loc["H1_add_alternative_financing"]
    o1 = ov[(ov.a == "P_RBP:PI") & (ov.b == "S_RBP:PI")].iloc[0]
    o2 = ov[(ov.a == "S_RBP:RHO") & (ov.b == "P_RBP:RHO")].iloc[0]
    return {"grant_main_rule_value": m.value, "grant_lottery_value": m.value_reference,
            "grant_main_rule_gain": m["diff"], "grant_main_rule_gain_lo": m.ci_lo, "grant_main_rule_gain_hi": m.ci_hi,
            "grant_existing_ai_interaction": ai.inter_per_sd, "grant_existing_ai_interaction_lo": ai.lo,
            "grant_existing_ai_interaction_hi": ai.hi, "grant_existing_judges_interaction": jd.inter_per_sd,
            "grant_existing_difference": db.loc[1, "diff"], "grant_existing_difference_lo": db.loc[1, "diff_lo"],
            "grant_existing_difference_hi": db.loc[1, "diff_hi"],
            "lending_policy_gain": pr.estimate, "lending_policy_gain_lo": pr.lo95, "lending_policy_gain_hi": pr.hi95,
            "lending_ml_gain": ml.estimate, "lending_H1": h1.estimate, "lending_H1_lo": h1.lo_adj,
            "lending_H1_hi": h1.hi_adj, "lending_H1_mde": h1.mde_adj,
            "lending_repay_itt_pp": 100 * sm["itt_perfect_repayment"]["estimate"],
            "lending_overlap_profit": o1.overlap, "lending_overlap_repay": o2.overlap, "lending_answers": sm["answers"]}


def main():
    a = args("Compare with proposal numbers (full) or known truth (demo)")
    out = step_dir(a, "10_compare_reference")
    run = a.run_dir
    acc = pd.read_csv(run / "01_hypothetical_firms" / "information_accuracy.csv").set_index(["model", "info"])
    fig1 = {"Sonnet both facts correct (100%)": (acc.loc[("Sonnet 5.5", "XRB"), "pct_correct"], 100),
            "Haiku both facts correct (97%)": (acc.loc[("Haiku 5.5", "XRB"), "pct_correct"], 97),
            "Qwen both facts correct (42%)": (acc.loc[("Qwen2.5-3B", "XRB"), "pct_correct"], 42),
            "Sonnet withholds, effect only (89%)": (acc.loc[("Sonnet 5.5", "XB"), "pct_abstain"], 89),
            "Haiku withholds, effect only (24%)": (acc.loc[("Haiku 5.5", "XB"), "pct_abstain"], 24)}
    rows = [dict(item=k, produced=v, printed=p, rounds_to_printed=round(v) == p) for k, (v, p) in fig1.items()]
    summary = {"hypothetical_firms_all_match": all(r["rounds_to_printed"] for r in rows)}
    if a.mode == "full":
        got = produced_full(run)
        for it in json.loads((CONFIG / "proposal_numbers.json").read_text())["items"]:
            v = float(got[it["id"]])
            rows.append(dict(item=it["id"], description=it["text"], produced=v, printed=it["printed"],
                             rounds_to_printed=round(v, it["decimals"]) == round(it["printed"], it["decimals"]),
                             original_full_precision=it["original"], abs_diff_vs_original=abs(v - it["original"])))
        d = pd.DataFrame(rows)
        summary.update(all_round_to_printed=bool(d.rounds_to_printed.all()),
                       max_abs_diff_vs_original=float(d.abs_diff_vs_original.max()))
    else:
        pc = pd.read_csv(run / "05_grant_policy_value" / "policy_contrasts.csv")
        it = pd.read_csv(run / "06_grant_heterogeneity" / "interactions.csv")
        pv = pd.read_csv(run / "07_lending_evaluation" / "policy_values.csv")
        for r in pc.itertuples():
            rows.append(dict(item=f"grant policy gain: {r.rule}", produced=r.diff, truth=r.true_gain,
                             lo=r.ci_lo, hi=r.ci_hi, covers=r.ci_covers_truth))
        for r in it.itertuples():
            rows.append(dict(item=f"grant interaction: existing={r.existing} {r.score} {r.outcome}",
                             produced=r.inter_per_sd, truth=r.true_slope, lo=r.lo, hi=r.hi, covers=r.ci_covers_truth))
        for r in pv.itertuples():
            rows.append(dict(item=f"lending policy gain: {r.outcome} {r.rule} vs {r.versus}", produced=r.estimate,
                             truth=r.true_gain, lo=r.lo95, hi=r.hi95, covers=r.ci_covers_truth))
        d = pd.DataFrame(rows)
        sim = run / "08_simulation_check" / "summary.json"
        if sim.exists():
            summary["simulation_check"] = json.loads(sim.read_text())
    d.to_csv(out / "comparison.csv", index=False)
    write_json(out / "summary.json", summary)
    pd.set_option("display.width", 220)
    print(d.to_string(index=False))
    print(summary)


if __name__ == "__main__":
    main()
