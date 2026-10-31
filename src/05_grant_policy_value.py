# ============================================================
# START DATE   : 2026-10-27
# CODE RUN DATE: 2026-10-31
# TIME TAKEN   : ~2 min (2,000 paired bootstrap draws)
# DESCRIPTION  : Grant lottery: what would each allocation rule have yielded if it had awarded the same
#                number of grants as the lottery within every randomization stratum? Rules: AI scores
#                (added employment A, employment level L, outside-financing chance F; application only X
#                or with financing facts XR), the judges' business-plan score, and a cross-fit T-learner.
#                Ties are averaged (expected selection). Each rule is compared with the lottery using
#                a paired bootstrap within stratum x arm cells. Same code for the YouWiN! data (full)
#                and for the synthetic lottery (demo); in demo mode the known true gain is reported too.
# INPUT FILES  : <run>/work/grant_frame.pkl; answers: data/model_outputs/youwin_answers_claude-sonnet-5-5.csv
#                (full) or <run>/work/synthetic_grant_answers.csv (demo); demo truth <run>/work/synthetic_grant_truth.pkl
# OUTPUT FILES : <run>/05_grant_policy_value/{policy_contrasts.csv, summary.json}
# ============================================================
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, args, params, step_dir, write_json  # noqa: E402
import evallib as E  # noqa: E402

RULES = [("Business plan score", "totalmarks"), ("T-learner (cross-fit)", None),
         ("Claude A, X", "A_X"), ("Claude L, X", "L_X"), ("Claude F, X", "F_X"),
         ("Claude A, XR", "A_XR"), ("Claude L, XR", "L_XR"), ("Claude F, XR", "F_XR")]
RANDOM = "Random (actual share)"


def answers_path(a):
    if a.mode == "full":
        return DATA / "model_outputs" / "youwin_answers_claude-sonnet-5-5.csv"
    return a.work / "synthetic_grant_answers.csv"


def main():
    a = args("Grant lottery policy values")
    out = step_dir(a, "05_grant_policy_value")
    P = params()["grant"]
    frame = pd.read_pickle(a.work / "grant_frame.pkl")
    ans = pd.read_csv(answers_path(a), dtype={"answer": str})
    d = frame.merge(E.grant_scores_from_answers(ans), on="uid", how="left").sort_values("uid").reset_index(drop=True)
    assert d[["A_X", "A_XR", "F_X", "F_XR"]].notna().all().all()

    strata = d.strata.to_numpy()
    arm = d.assigntreat.to_numpy().astype(int)
    y = np.nan_to_num(d.t_totalemp1.to_numpy(dtype=float))
    resp = d.t_totalemp1.notna().to_numpy()
    n_total, n_by, k_by, weights, cells = E.grant_design(strata, arm)
    k_total = sum(k_by.values())

    X = d[[c for c in d.columns if c.startswith("x_")]].to_numpy(float)
    tau = E.tlearner_scores(X, y, arm, resp, np.random.default_rng(P["tlearner_seed"]), P["tlearner_seed"])
    pis = {RANDOM: E.random_selection(strata, k_by, n_by)}
    for name, col in RULES:
        sc = tau if col is None else d[col].to_numpy(dtype=float)
        pis[name] = E.expected_selection(sc, strata, k_by)
        assert abs(pis[name].sum() - k_total) < 1e-6

    ct, point = E.paired_policy_bootstrap(pis, cells, y, resp, weights, n_total,
                                          P["policy_nboot"], P["policy_seed"], RANDOM)
    ct["tie_share_of_places"] = [E.tie_share(pis[r], k_total) for r in ct.rule]

    summary = {"applicants": n_total, "winners": k_total, "strata": len(n_by),
               "responders": int(resp.sum()),
               "response_rate_treated": float(resp[arm == 1].mean()),
               "response_rate_control": float(resp[arm == 0].mean()),
               "nboot": P["policy_nboot"], "main_rule": P["main_rule"],
               "value_random": point[RANDOM], "value_main_rule": point[P["main_rule"]]}
    if a.mode == "demo":
        # ★ CORE LOGIC: the synthetic world stores both potential outcomes, so the true gain of each rule over
        # the lottery is sum_s (n_s/N) * mean_s[(pi - k_s/n_s) * tau], computed on all applicants.
        tr = pd.read_pickle(a.work / "synthetic_grant_truth.pkl").set_index("uid").loc[d.uid]
        tau_true = tr.tau_emp.to_numpy()
        true_gain = {}
        for r in ct.rule:
            dpi = pis[r] - pis[RANDOM]
            true_gain[r] = float(sum(weights[s] * np.mean((dpi * tau_true)[strata == s]) for s in n_by))
        ct["true_gain"] = ct.rule.map(true_gain)
        ct["ci_covers_truth"] = (ct.ci_lo <= ct.true_gain) & (ct.true_gain <= ct.ci_hi)
    ct.to_csv(out / "policy_contrasts.csv", index=False)
    write_json(out / "summary.json", summary)
    pd.set_option("display.width", 200)
    print(ct.round(3).to_string(index=False))
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in summary.items()})


if __name__ == "__main__":
    main()
