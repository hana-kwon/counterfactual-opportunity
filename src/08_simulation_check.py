# ============================================================
# START DATE   : 2026-10-31
# CODE RUN DATE: 2026-10-31
# TIME TAKEN   : ~1-2 min (200 synthetic draws)
# DESCRIPTION  : Demo path only. One synthetic draw can land far from the truth by chance, so this step
#                redraws both synthetic experiments many times (same design, new seeds) and checks the
#                evaluation code against the known effects: mean bias of the grant policy-value contrast
#                (main AI rule minus lottery), coverage of the HC1 interval for the grant x AI-score
#                interaction (existing firms), and bias and 95 percent coverage of the Horvitz-Thompson
#                policy contrast for the lending design (profit-gain score, all information, vs random).
# INPUT FILES  : config/params.json
# OUTPUT FILES : <run>/08_simulation_check/{draws.csv, summary.json}
# ============================================================
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import args, params, step_dir, write_json  # noqa: E402
import evallib as E  # noqa: E402

G = importlib.import_module("02_synthetic_experiments")
DRAWS = 200


def grant_draw(rng, n):
    frame, ans, truth, _, _ = G.make_grant(rng, n)
    d = frame.merge(E.grant_scores_from_answers(ans), on="uid").sort_values("uid").reset_index(drop=True)
    tau = truth.set_index("uid").loc[d.uid].tau_emp.to_numpy()
    strata, arm = d.strata.to_numpy(), d.assigntreat.to_numpy().astype(int)
    y, resp = np.nan_to_num(d.t_totalemp1.to_numpy(float)), d.t_totalemp1.notna().to_numpy()
    n_total, n_by, k_by, weights, cells = E.grant_design(strata, arm)
    pr = E.random_selection(strata, k_by, n_by)
    pa = E.expected_selection(d.A_XR.to_numpy(float), strata, k_by)
    est = E.policy_value(pa, cells, y, resp, weights, n_total) - E.policy_value(pr, cells, y, resp, weights, n_total)
    true = sum(weights[s] * np.mean(((pa - pr) * tau)[strata == s]) for s in n_by)
    # interaction, existing firms
    m = (d.existing == 1).to_numpy()
    z = E.zscore(d.loc[m, "A_XR"].astype(float)).to_numpy()
    q = m & resp
    zq = z[resp[m]]
    t = arm[q].astype(float)
    b, se, _ = E.areg(d.t_totalemp1.to_numpy(float)[q], np.column_stack([t, t * zq, zq]), strata[q])
    slope = float(np.polyfit(z, tau[m], 1)[0])
    return est, true, b[1], se[1], slope


def lending_draw(rng, n, n_off):
    prof, cl, bench, ans, truth, _ = G.make_lending(rng, n, n_off)
    scores, _ = E.lending_scores_from_answers(ans)
    y = prof.groupby("client_key").e_profits_all.mean()
    base = cl.set_index("client_key").loc[y.index]
    p = E.selection_probs(scores[("P_RBP", "PI")].reindex(base.index).to_numpy(), base.loid_baseline.to_numpy())[0]
    r = E.policy_contrast_ht(p - 0.5, y.to_numpy(), base.treatment.to_numpy(float), base.block.to_numpy())
    true = float(np.mean((p - 0.5) * truth.set_index("client_key").loc[base.index].tau_profit.to_numpy()))
    return r["estimate"], r["lo95"], r["hi95"], true


def main():
    a = args("Repeated synthetic draws: bias and coverage against known effects")
    out = step_dir(a, "08_simulation_check")
    P = params()["synthetic"]
    rows = []
    for k in range(DRAWS):
        rng = np.random.default_rng(P["seed"] + 1 + k)
        ge, gt, gb, gse, gs = grant_draw(rng, P["grant_applicants"])
        le, llo, lhi, lt = lending_draw(rng, P["lending_clients"], P["lending_officers"])
        rows.append(dict(draw=k, grant_policy_est=ge, grant_policy_true=gt, grant_inter_est=gb, grant_inter_se=gse,
                         grant_inter_true=gs, lending_policy_est=le, lending_lo95=llo, lending_hi95=lhi,
                         lending_policy_true=lt))
    d = pd.DataFrame(rows)
    d.to_csv(out / "draws.csv", index=False)
    cov_g = ((d.grant_inter_est - 1.96 * d.grant_inter_se <= d.grant_inter_true)
             & (d.grant_inter_true <= d.grant_inter_est + 1.96 * d.grant_inter_se)).mean()
    cov_l = ((d.lending_lo95 <= d.lending_policy_true) & (d.lending_policy_true <= d.lending_hi95)).mean()
    s = {"draws": DRAWS,
         "grant_policy_mean_estimate": d.grant_policy_est.mean(), "grant_policy_mean_truth": d.grant_policy_true.mean(),
         "grant_policy_mean_bias": (d.grant_policy_est - d.grant_policy_true).mean(),
         "grant_policy_mc_se_of_bias": (d.grant_policy_est - d.grant_policy_true).std(ddof=1) / np.sqrt(DRAWS),
         "grant_interaction_mean_estimate": d.grant_inter_est.mean(), "grant_interaction_mean_truth": d.grant_inter_true.mean(),
         "grant_interaction_95_coverage": cov_g,
         "lending_policy_mean_estimate": d.lending_policy_est.mean(), "lending_policy_mean_truth": d.lending_policy_true.mean(),
         "lending_policy_mean_bias": (d.lending_policy_est - d.lending_policy_true).mean(),
         "lending_policy_mc_se_of_bias": (d.lending_policy_est - d.lending_policy_true).std(ddof=1) / np.sqrt(DRAWS),
         "lending_policy_95_coverage": cov_l}
    write_json(out / "summary.json", s)
    for k, v in s.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")


if __name__ == "__main__":
    main()
