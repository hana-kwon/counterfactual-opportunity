# ============================================================
# START DATE   : 2026-10-29
# CODE RUN DATE: 2026-10-31
# TIME TAKEN   : ~2 min
# DESCRIPTION  : Lending experiment: scores the AI forecasts against the randomized outcomes.
#                (1) The four pre-specified comparisons: change in the larger-offer x score interaction
#                    when alternative financing is added (H1) and when only the stated objective changes
#                    (H2_profit, H2_repay, H2_level); SE clustered by randomization block, Bonferroni
#                    intervals for four tests (98.75 percent).
#                (2) Policy value: give the larger offer to the higher-scoring half of each loan officer's
#                    clients (tie-aware) instead of at random; Horvitz-Thompson with assignment
#                    probability 0.5, block-clustered SE (supporting, not adjusted).
#                (3) Average effects of the larger offer and mean model forecasts (descriptive).
#                (4) Expected overlap of selected clients when only the objective sentence changes.
#                Same code for the replication data (full) and the synthetic lending experiment (demo);
#                in demo mode the true policy gains are reported too.
# INPUT FILES  : <run>/work/lending_{prof,cl,bench}.pkl; answers: data/model_outputs/egypt_answers_claude-sonnet-5-5.csv
#                (full) or <run>/work/synthetic_lending_answers.csv (demo); demo truth <run>/work/synthetic_lending_truth.pkl
# OUTPUT FILES : <run>/07_lending_evaluation/{confirmatory.csv, policy_values.csv, forecasts_by_condition.csv,
#                overlap.csv, summary.json}
# ============================================================
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, args, params, step_dir, write_json  # noqa: E402
import evallib as E  # noqa: E402

# A contrast is a list of (weight, condition, score type).
PRIMARY = {
    "H1_add_alternative_financing": dict(outcome="profit", terms=[(1, "P_R", "PI"), (-1, "P0", "PI")]),
    "H2_profit_vs_repayment_objective": dict(outcome="profit", terms=[(1, "P_RBP", "PI"), (-1, "S_RBP", "PI")]),
    "H2_repayment_vs_profit_objective": dict(outcome="repay", terms=[(1, "S_RBP", "RHO"), (-1, "P_RBP", "RHO")]),
    "H2_gain_vs_level_objective": dict(outcome="profit", terms=[(1, "P_RBP", "PI"), (-1, "L_RBP", "PI")]),
}
POLICIES = {"profit": [("P_RBP", "PI"), ("P0", "PI"), ("L_RBP", "LEVEL"), ("BKO", "ML")],
            "repay": [("S_RBP", "RHO"), ("P_RBP", "PI")]}
PAIRS = {"profit": [(("P_RBP", "PI"), ("L_RBP", "LEVEL"))]}
OVERLAPS = [(("P_RBP", "PI"), ("S_RBP", "PI")), (("S_RBP", "RHO"), ("P_RBP", "RHO")),
            (("P_RBP", "PI"), ("S_RBP", "RHO")), (("P_RBP", "PI"), ("L_RBP", "LEVEL"))]
CONDS = ["P0", "P_R", "P_B", "P_RB", "P_RBP", "S_RBP", "L_RBP"]


def main():
    a = args("Lending experiment evaluation")
    out = step_dir(a, "07_lending_evaluation")
    P = params()["lending"]
    prof = pd.read_pickle(a.work / "lending_prof.pkl")
    cl = pd.read_pickle(a.work / "lending_cl.pkl")
    bench = pd.read_pickle(a.work / "lending_bench.pkl")
    path = DATA / "model_outputs" / "egypt_answers_claude-sonnet-5-5.csv" if a.mode == "full" \
        else a.work / "synthetic_lending_answers.csv"
    scores, flat = E.lending_scores_from_answers(pd.read_csv(path, dtype={"answer": str}))
    scores[("BKO", "ML")] = bench.set_index("client_key").ML.astype(float)

    # (1) pre-specified comparisons
    conf = []
    for k, spec in PRIMARY.items():
        r = E.contrast(prof, cl, scores, spec, n_tests=P["n_primary_tests"])
        conf.append(dict(contrast=k, **{x: r[x] for x in ("outcome", "estimate", "se", "lo_adj", "hi_adj", "lo95",
                                                          "hi95", "mde_adj", "n_common_clients", "n_rows",
                                                          "n_clusters")}))
    conf = pd.DataFrame(conf)
    conf["distinguishes"] = (conf.lo_adj > 0) | (conf.hi_adj < 0)

    # (2) policy values (HT), every client gets p from the score's half-per-officer selection
    truth = pd.read_pickle(a.work / "synthetic_lending_truth.pkl").set_index("client_key") if a.mode == "demo" else None
    pol = []
    for ycol in ("profit", "repay"):
        if ycol == "profit":
            y = prof.groupby("client_key").e_profits_all.mean()      # client mean over observed rounds
            base = cl.set_index("client_key").loc[y.index]
        else:
            base = cl.set_index("client_key")
            y = base.perfect_repayment
        lo = base.loid_baseline.to_numpy()
        t, blk, yv = base.treatment.to_numpy(float), base.block.to_numpy(), y.to_numpy(float)

        def probs(key):
            sc = scores[key].reindex(base.index)
            p = np.full(len(base), 0.5)                               # a missing score gives p = 0.5
            ok = sc.notna().to_numpy()
            p[ok] = E.selection_probs(sc[ok].to_numpy(), lo[ok])[0]
            return p, int((~ok).sum())

        def true_gain(dp):
            if truth is None:
                return np.nan
            tau = truth.loc[base.index, "tau_profit" if ycol == "profit" else "tau_repay"].to_numpy()
            return float(np.mean(dp * tau))
        for k in POLICIES[ycol]:
            p, miss = probs(k)
            r = E.policy_contrast_ht(p - 0.5, yv, t, blk)
            pol.append(dict(outcome=ycol, rule=f"{k[0]}:{k[1]}", versus="random", missing_scores=miss,
                            true_gain=true_gain(p - 0.5), **r))
        for ka, kb in PAIRS.get(ycol, []):
            pa, pb = probs(ka)[0], probs(kb)[0]
            r = E.policy_contrast_ht(pa - pb, yv, t, blk)
            pol.append(dict(outcome=ycol, rule=f"{ka[0]}:{ka[1]}", versus=f"{kb[0]}:{kb[1]}", missing_scores=0,
                            true_gain=true_gain(pa - pb), **r))
    pol = pd.DataFrame(pol)
    if a.mode == "demo":
        pol["ci_covers_truth"] = (pol.lo95 <= pol.true_gain) & (pol.true_gain <= pol.hi95)

    # (3) average effects and forecasts
    itt_r1 = E.itt(prof[prof["round"] == 1], "e_profits_all")
    itt_r2 = E.itt(prof[prof["round"] == 2], "e_profits_all")
    itt_rep = E.itt(cl, "perfect_repayment")
    fc = flat.groupby("cond").agg(
        answers=("PI", "size"), insufficient_share=("information", lambda s: float((s == "insufficient").mean())),
        distinct_profit_gain_values=("PI", "nunique"), mean_PI_20=("PI_20", "mean"), mean_PI_30=("PI_30", "mean"),
        mean_accept_normal=("normal_accept", "mean"), mean_accept_large=("large_accept", "mean"),
        mean_late_normal=("normal_late", "mean"), mean_late_large=("large_late", "mean"),
        mean_RHO_pp=("RHO", lambda s: 100 * s.mean())).reindex([c for c in CONDS if c in set(flat.cond)])
    fc["share_lower_accept_large"] = flat.assign(x=flat.large_accept < flat.normal_accept).groupby("cond").x.mean()
    fc["share_higher_late_large"] = flat.assign(x=flat.large_late > flat.normal_late).groupby("cond").x.mean()

    # (4) expected overlap of selections
    base = cl.set_index("client_key")
    ov = []
    for ka, kb in OVERLAPS:
        dd = pd.concat([scores[ka].rename("a"), scores[kb].rename("b")], axis=1).dropna()
        dd = dd[dd.index.isin(base.index)]
        o = E.overlap_stats(dd.a.to_numpy(), dd.b.to_numpy(), base.loc[dd.index].loid_baseline.to_numpy(),
                            reps=P["overlap_reps"], seed=P["overlap_seed"])
        ov.append(dict(a=f"{ka[0]}:{ka[1]}", b=f"{kb[0]}:{kb[1]}", **o))
    ov = pd.DataFrame(ov)

    conf.to_csv(out / "confirmatory.csv", index=False)
    pol.to_csv(out / "policy_values.csv", index=False)
    fc.reset_index().to_csv(out / "forecasts_by_condition.csv", index=False)
    ov.to_csv(out / "overlap.csv", index=False)
    write_json(out / "summary.json", {
        "answers": len(flat), "clients": int(cl.client_key.nunique()),
        "clients_by_arm": {str(k): int(v) for k, v in cl.groupby("treatment").size().items()},
        "clients_with_profit_by_arm": {str(k): int(v) for k, v in
                                       prof.drop_duplicates("client_key").groupby("treatment").size().items()},
        "profit_rows": len(prof),
        "itt_profit_round1": dict(zip(("estimate", "se", "n_rows"), itt_r1)),
        "itt_profit_round2": dict(zip(("estimate", "se", "n_rows"), itt_r2)),
        "itt_perfect_repayment": dict(zip(("estimate", "se", "n"), itt_rep)),
        "bonferroni_tests": P["n_primary_tests"]})
    pd.set_option("display.width", 220)
    print(conf.round(3).to_string(index=False))
    print(pol.round(3).to_string(index=False))
    print(fc.round(3).to_string())
    print(ov[["a", "b", "overlap", "identical_upper", "perm_mean"]].round(3).to_string(index=False))
    print("ITT perfect repayment", itt_rep)


if __name__ == "__main__":
    main()
