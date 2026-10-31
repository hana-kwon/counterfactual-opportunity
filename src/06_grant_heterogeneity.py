# ============================================================
# START DATE   : 2026-10-29
# CODE RUN DATE: 2026-10-31
# TIME TAKEN   : ~1 min
# DESCRIPTION  : Grant lottery: does a score separate firms with larger grant effects? McKenzie (2017)
#                Appendix Table A18 form, `areg y treat treat*S S, a(strata) robust`, with S the judges'
#                business-plan score or the AI added-employment score (financing facts included),
#                each standardized within the new / existing subsample before outcomes are used.
#                Outcomes: employment and the inverse hyperbolic sine of real profits. The difference
#                between the AI and judges' coefficients (employment) uses a paired bootstrap within
#                stratum x arm cells. Same code for YouWiN! (full) and the synthetic lottery (demo); in
#                demo mode the true slope of the effect on the standardized score is reported too.
# INPUT FILES  : <run>/work/grant_frame.pkl; answers as in step 05; demo truth <run>/work/synthetic_grant_truth.pkl
# OUTPUT FILES : <run>/06_grant_heterogeneity/{interactions.csv, difference_bootstrap.csv, summary.json}
# ============================================================
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, args, params, step_dir, write_json  # noqa: E402
import evallib as E  # noqa: E402

SCORES = {"totalmarks": "judges' business-plan score", "A_XR": "AI added employment, with financing facts"}
OUTCOMES = ["t_totalemp1", "t_IHSprofits"]


def main():
    a = args("Grant lottery: treatment x score interactions")
    out = step_dir(a, "06_grant_heterogeneity")
    P = params()["grant"]
    frame = pd.read_pickle(a.work / "grant_frame.pkl")
    path = DATA / "model_outputs" / "youwin_answers_claude-sonnet-5-5.csv" if a.mode == "full" \
        else a.work / "synthetic_grant_answers.csv"
    d = frame.merge(E.grant_scores_from_answers(pd.read_csv(path, dtype={"answer": str})), on="uid", how="left")
    d = d.sort_values("uid").reset_index(drop=True)
    for ex in (0, 1):
        m = d["existing"] == ex
        for s in SCORES:
            # ★ IMPORTANT: standardized over all applicants in the subsample, before outcomes are looked at
            d.loc[m, "z_" + s] = E.zscore(d.loc[m, s].astype(float))

    rows = []
    for ex in (0, 1):
        sub = d[d["existing"] == ex].sort_values("uid")
        for s in SCORES:
            for o in OUTCOMES:
                q = sub[sub[o].notna()]
                y = q[o].to_numpy(float)
                t = q["assigntreat"].to_numpy(float)
                z = q["z_" + s].to_numpy(float)
                b, se, n = E.areg(y, np.column_stack([t, t * z, z]), q["strata"].to_numpy())
                rows.append(dict(existing=ex, score=s, outcome=o, n=n, treat_at_mean=b[0], inter_per_sd=b[1],
                                 se=se[1], lo=b[1] - 1.96 * se[1], hi=b[1] + 1.96 * se[1], p=E.pval(b[1], se[1])))
    r = pd.DataFrame(rows)

    # ★ CORE LOGIC: primary contrast (AI minus judges, employment), paired bootstrap over applicants
    # within stratum x arm cells; both coefficients are re-estimated on every resample.
    rng = np.random.default_rng(P["heterogeneity_seed"])
    boot = []
    for ex in (0, 1):
        q = d[(d["existing"] == ex) & d["t_totalemp1"].notna()].reset_index(drop=True)
        cells = q.groupby(["strata", "assigntreat"]).indices
        y = q["t_totalemp1"].to_numpy(float)
        t = q["assigntreat"].to_numpy(float)
        st = q["strata"].to_numpy()
        za, zj = q["z_A_XR"].to_numpy(float), q["z_totalmarks"].to_numpy(float)

        def diff(idx):
            ba = E.within_coef(y[idx], np.column_stack([t[idx], t[idx] * za[idx], za[idx]]), st[idx])[1]
            bj = E.within_coef(y[idx], np.column_stack([t[idx], t[idx] * zj[idx], zj[idx]]), st[idx])[1]
            return ba, bj
        full = diff(np.arange(len(q)))
        draws = []
        for _ in range(P["heterogeneity_reps"]):
            idx = np.concatenate([rng.choice(ix, size=len(ix), replace=True) for ix in cells.values()])
            draws.append(diff(idx))
        draws = np.array(draws)
        dd = draws[:, 0] - draws[:, 1]
        boot.append(dict(existing=ex, n=len(q), reps=int(draws.shape[0]), inter_A_XR=full[0], inter_judges=full[1],
                         diff=full[0] - full[1], diff_lo=np.percentile(dd, 2.5), diff_hi=np.percentile(dd, 97.5)))
    bt = pd.DataFrame(boot)

    summary = {"spearman_ai_judges": {str(ex): float(d[d.existing == ex][["A_XR", "totalmarks"]]
                                                     .corr(method="spearman").iloc[0, 1]) for ex in (0, 1)}}
    if a.mode == "demo":
        # True heterogeneity: OLS slope of the true employment effect on each standardized score, by subsample.
        tr = pd.read_pickle(a.work / "synthetic_grant_truth.pkl").set_index("uid").loc[d.uid]
        d["tau_true"] = tr.tau_emp.to_numpy()
        d["tau_profit_true"] = tr.tau_ihs_profit.to_numpy()
        slope = {}
        for ex in (0, 1):
            m = d.existing == ex
            for s in SCORES:
                z = d.loc[m, "z_" + s]
                for o, tcol in (("t_totalemp1", "tau_true"), ("t_IHSprofits", "tau_profit_true")):
                    slope[(ex, s, o)] = float(np.polyfit(z, d.loc[m, tcol], 1)[0])
        r["true_slope"] = [slope[(x.existing, x.score, x.outcome)] for x in r.itertuples()]
        r["ci_covers_truth"] = (r.lo <= r.true_slope) & (r.true_slope <= r.hi)
        summary["note"] = "true_slope = OLS slope of the known effect on the standardized score"
    r.to_csv(out / "interactions.csv", index=False)
    bt.to_csv(out / "difference_bootstrap.csv", index=False)
    write_json(out / "summary.json", summary)
    pd.set_option("display.width", 200)
    print(r.round(3).to_string(index=False))
    print(bt.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
