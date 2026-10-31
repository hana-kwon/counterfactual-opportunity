# ============================================================
# START DATE   : 2026-10-31
# CODE RUN DATE: 2026-10-31
# TIME TAKEN   : < 1 min
# DESCRIPTION  : Demo path only. Generates two synthetic randomized experiments with known heterogeneous
#                effects, in the same file formats the full path builds from the replication packages,
#                plus synthetic "AI answers" in the exact answer formats of the real model outputs:
#                (a) a grant lottery (strata = region x existing x gender, a fixed number of winners per
#                    stratum, employment and profit three years later, missing follow-up at random within
#                    stratum x arm). The true employment effect is large only for financially constrained
#                    firms in high-opportunity sectors; the profit effect is constant.
#                (b) a lending experiment (loan officers, randomization blocks with half of each block
#                    offered the larger loan, profit at two follow-ups, perfect repayment). The true profit
#                    gain is large only for credit-constrained owners who plan to buy equipment or inputs;
#                    the larger loan lowers perfect repayment on average.
#                The synthetic answers mimic evaluators whose information differs by condition (for
#                example, the "application only" evaluator cannot see the financing constraint), and
#                they are rounded to integers or hundreds so that ties occur as in real answers. Every
#                synthetic answer carries the SHA-256 of the prompt that the repo's prompt builders
#                produce for that synthetic profile, and example prompts are written for inspection.
# INPUT FILES  : config/params.json
# OUTPUT FILES : <run>/work/{grant_frame.pkl, synthetic_grant_answers.csv, synthetic_grant_truth.pkl,
#                lending_prof.pkl, lending_cl.pkl, lending_bench.pkl, synthetic_lending_answers.csv,
#                synthetic_lending_truth.pkl}; <run>/02_synthetic/{example_prompts.md, design.json}
# ============================================================
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import args, params, step_dir, write_json  # noqa: E402
import prompts_grant as PG  # noqa: E402
import prompts_lending as PL  # noqa: E402

GRANT_FEATURES = ["female", "age", "marital_status", "number_of_children", "postgraduate", "university",
                  "polytechnic", "existing", "business_sector", "business_accounting_training",
                  "marketing_training", "computer_skills", "vocational_training", "otherwise_employed",
                  "lived_worked_abroad", "have_computers", "have_cars_and_other_vehicles", "have_internet",
                  "years_of_experience", "employees_num", "loan_from_bank", "loan_from_microfinance_organisat",
                  "loan_from_cooperative", "lack_funding", "lack_skills", "lack_experience", "lack_interest"]
# Sector -> opportunity for added employment (known to the synthetic evaluator, as a sector is on the form)
SECTOR_OPP = {13: 0.3, 18: 0.4, 20: 0.2, 30: 0.9, 31: 0.8, 33: 0.9, 41: 0.5, 50: 0.7, 74: 0.6, 80: 0.8}
STATES = ["Lagos", "Kano", "Oyo", "Rivers", "Kaduna", "Enugu"]


def make_grant(rng, n):
    """★ CORE LOGIC: synthetic grant lottery with a known employment effect tau_i."""
    region = rng.integers(0, 6, n)
    existing = (rng.random(n) < 0.3).astype(int)
    female = (rng.random(n) < 0.25).astype(int)
    sector = rng.choice(list(SECTOR_OPP), n)
    opp = np.array([SECTOR_OPP[s] for s in sector]) + rng.normal(0, 0.15, n)       # partly unobserved
    employees = np.where(existing == 1, rng.poisson(4, n) + 1, 0)
    has_loan = (existing == 1) & (rng.random(n) < 0.3)
    lack_funding = (rng.random(n) < np.where(existing == 1, 0.6, 0.8)).astype(int)
    constrained = (~has_loan & (lack_funding == 1)).astype(int)
    # Potential outcomes (employment three years later, counting the owner)
    y0 = np.where(existing == 1, 1 + 0.9 * employees, 1 + rng.poisson(1.5, n)) + rng.normal(0, 1.0, n)
    tau = 1.0 + 6.0 * constrained * np.clip(opp, 0, None)
    tau = tau * np.where(existing == 1, 1.0, 0.6)
    eps = rng.normal(0, 2.5, n)
    # Lottery: a fixed share of winners within each stratum
    strata = pd.Series(list(zip(region, existing, female))).factorize()[0].astype(float)
    arm = np.zeros(n, int)
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        k = int(round(0.4 * len(idx)))
        arm[rng.choice(idx, k, replace=False)] = 1
    y = y0 + arm * tau + eps
    resp = rng.random(n) < np.where(arm == 1, 0.87, 0.78)
    # Profits (inverse hyperbolic sine): constant effect, no heterogeneity by design
    ihs0 = 11 + 0.15 * y0 + rng.normal(0, 1.5, n)
    tau_p = np.full(n, 0.5)
    prof = ihs0 + arm * tau_p
    resp_p = resp | (rng.random(n) < 0.05)
    # Judges' score: rewards size and polish, not the effect
    judges = 50 + 6 * (y0 - y0.mean()) / y0.std() + rng.normal(0, 6, n)
    df = pd.DataFrame(dict(
        uid=np.arange(1, n + 1), strata=strata, assigntreat=arm, existing=existing, female=female,
        t_totalemp1=np.where(resp, y, np.nan), t_IHSprofits=np.where(resp_p, prof, np.nan), totalmarks=judges,
        age=rng.integers(21, 45, n), marital_status=rng.integers(1, 5, n), number_of_children=rng.poisson(2, n),
        postgraduate=(rng.random(n) < 0.15).astype(int), university=(rng.random(n) < 0.6).astype(int),
        polytechnic=(rng.random(n) < 0.2).astype(int), state_residence=[STATES[r] for r in region],
        business_sector=sector, business_accounting_training=rng.integers(0, 2, n),
        marketing_training=rng.integers(0, 2, n), computer_skills=rng.integers(0, 2, n),
        vocational_training=rng.integers(0, 2, n), otherwise_employed=rng.integers(0, 2, n),
        lived_worked_abroad=(rng.random(n) < 0.1).astype(int), have_computers=rng.integers(0, 3, n),
        have_cars_and_other_vehicles=rng.integers(0, 2, n), have_internet=rng.integers(0, 2, n),
        years_of_experience=rng.integers(0, 15, n).astype(float),
        numyearsinbusiness=[f"{v} years" if e else "" for v, e in zip(rng.integers(1, 10, n), existing)],
        number_employees=[str(v) if e else "" for v, e in zip(employees, existing)],
        turnover=[f"N{v:,}" if e else "" for v, e in zip(rng.integers(5, 200, n) * 10000, existing)],
        business_registered=np.where(existing == 1, rng.integers(0, 2, n), np.nan),
        loan_from_bank=np.where(existing == 1, has_loan.astype(float), np.nan),
        loan_from_microfinance_organisat=np.where(existing == 1, 0.0, np.nan),
        loan_from_cooperative=np.where(existing == 1, 0.0, np.nan),
        lack_funding=lack_funding, lack_skills=(rng.random(n) < 0.2).astype(int),
        lack_experience=(rng.random(n) < 0.2).astype(int), lack_interest=(rng.random(n) < 0.05).astype(int),
        employees_num=employees.astype(float)))
    truth = pd.DataFrame(dict(uid=df.uid, tau_emp=tau, tau_ihs_profit=tau_p, constrained=constrained, opp=opp))

    # ★ CORE LOGIC: synthetic evaluator answers. X sees the application but not the financing facts, so it
    # can only guess the constraint at its average rate; XR sees the facts. Answers are integers (ties).
    opp_obs = np.array([SECTOR_OPP[s] for s in sector])
    scale = np.where(existing == 1, 1.0, 0.6)
    y0_hat = np.where(existing == 1, 1 + 0.9 * employees, 2.5)
    added = {"X": scale * (1 + 6 * 0.7 * opp_obs) + rng.normal(0, 1.5, n),
             "XR": scale * (1 + 6 * constrained * opp_obs) + rng.normal(0, 1.5, n)}
    p_out = {"X": np.full(n, 25.0) + rng.normal(0, 5, n),
             "XR": 10 + 50 * (1 - constrained) + rng.normal(0, 5, n)}
    prompts = {p["key"]: p for p in PG.prompt_records(df)}
    rows = []
    for i, r in enumerate(df.itertuples(index=False)):
        for cond in ("X", "XR"):
            f = {"p_outside_financing": int(np.clip(round(p_out[cond][i] / 5) * 5, 0, 100))}
            e0 = int(max(0, round(y0_hat[i] + rng.normal(0, 1))))
            la = {"emp_without": e0, "emp_with": e0 + int(max(0, round(added[cond][i]))),
                  "information": "insufficient" if rng.random() < 0.9 else "sufficient"}
            for task, ans in (("F", f), ("LA", la)):
                key = f"{r.uid}|{cond}|{task}"
                rows.append(dict(uid=r.uid, condition=cond, task=task, model="synthetic-evaluator",
                                 prompt_sha256=prompts[key]["prompt_sha256"], answer=json.dumps(ans)))
    frame = df[["uid", "strata", "assigntreat", "existing", "t_totalemp1", "t_IHSprofits", "totalmarks"]].copy()
    for c in GRANT_FEATURES:
        frame["x_" + c] = df[c].astype(float).fillna(-1.0).to_numpy()
    return frame, pd.DataFrame(rows), truth, df, prompts


def make_lending(rng, n, n_off):
    """★ CORE LOGIC: synthetic lending experiment with a known profit gain and repayment effect."""
    officer = np.sort(rng.integers(0, n_off, n))
    # blocks: within officer, consecutive pairs/quads; exactly half of each block gets the larger offer
    block = np.zeros(n, int)
    treat = np.zeros(n, int)
    b = 0
    for o in np.unique(officer):
        idx = np.flatnonzero(officer == o)
        rng.shuffle(idx)
        for start in range(0, len(idx), 4):
            blk = idx[start:start + 4]
            block[blk] = b
            t = np.zeros(len(blk), int)
            t[rng.choice(len(blk), len(blk) // 2, replace=False)] = 1
            if len(blk) % 2 == 1 and rng.random() < 0.5:
                t[np.flatnonzero(t == 0)[0]] = 1
            treat[blk] = t
            b += 1
    lackcredit = rng.choice([1, 2, 3, 4], n, p=[0.3, 0.2, 0.2, 0.3])
    grow_credit = (rng.random(n) < 0.4).astype(int)
    primary = rng.choice([1, 2, 3, 6, 9, 10, 12], n, p=[0.2, 0.35, 0.05, 0.1, 0.1, 0.1, 0.1])
    constrained = ((lackcredit == 1) | (grow_credit == 1)).astype(int)
    equip = np.isin(primary, [1, 2]).astype(int)
    base_profit = np.exp(rng.normal(8.3, 0.6, n))
    tau_profit = 200 + 2500 * constrained * equip + rng.normal(0, 300, n)
    low_profit = (base_profit < np.median(base_profit)).astype(int)
    p_perfect0 = 0.85 - 0.08 * low_profit
    tau_repay = -0.13 - 0.08 * (1 - equip)
    perfect = (rng.random(n) < np.clip(p_perfect0 + treat * tau_repay, 0, 1)).astype(float)
    client_key = 1000 + np.arange(n)
    prof_rows = []
    for rnd in (1, 2):
        y = base_profit * (1 + 0.1 * rnd) + treat * tau_profit + rng.normal(0, 1500, n)
        obs = rng.random(n) < 0.97
        prof_rows.append(pd.DataFrame(dict(client_key=client_key[obs], loid_baseline=officer[obs], block=block[obs],
                                           treatment=treat[obs], round=rnd, e_profits_all=y[obs])))
    prof = pd.concat(prof_rows, ignore_index=True)
    cl = pd.DataFrame(dict(client_key=client_key, loid_baseline=officer, block=block, treatment=treat,
                           perfect_repayment=perfect))
    bench = pd.DataFrame(dict(client_key=client_key, ML=tau_profit + rng.normal(0, 1200, n)))
    truth = pd.DataFrame(dict(client_key=client_key, tau_profit=tau_profit, tau_repay=tau_repay,
                              constrained=constrained, equip=equip))
    # Baseline profile fields used by the prompt builder (bands are applied there)
    base = {"client_key": client_key, "b_age": rng.integers(22, 65, n), "b_gender": (rng.random(n) < 0.3).astype(int),
            "b_education": rng.integers(1, 17, n), "b_experience": rng.integers(1, 30, n),
            "b_staff_yesno": rng.integers(0, 2, n), "b_registration": rng.integers(0, 2, n),
            "b_size_lastloan": rng.integers(2, 20, n) * 1000, "b_monthly_profits": np.round(base_profit, -1),
            "b_monthly_revenue": np.round(base_profit * 3, -2), "b_loans": rng.integers(0, 2, n),
            "b_other_loans_rosca": rng.integers(0, 2, n), "b_fin_lackcredit": lackcredit,
            "b_growconstr_credit": grow_credit, "b_fin_primaryloanuse": primary}
    for k, (c, _) in enumerate(PL.SECTORS):
        base[c] = (rng.integers(0, 5, n) == k).astype(int)
    for k, (c, _) in enumerate(PL.USES):
        base[c] = ((primary == k + 1) | (rng.random(n) < 0.1)).astype(int)
    for k in range(1, 51):
        base[f"b_psy{k}"] = rng.integers(1, 6, n)
    baseline = pd.DataFrame(base)
    statements = [f"Synthetic statement {k} (placeholder for the original survey item)" for k in range(1, 51)]
    conds = list(PL.CONDITIONS) + list(PL.OPTIONAL_CONDITIONS)
    prompts = {p["key"]: p for p in PL.prompt_records(baseline, statements, conds)}

    # ★ CORE LOGIC: synthetic evaluator. What it can use differs by condition: P0 sees neither the
    # constraint nor the loan use; P_R sees the constraint; P_B sees the use; P_RB / P_RBP / S_RBP / L_RBP
    # see both. Forecasts are rounded to hundreds or fifties, which creates ties.
    sees = {"P0": (0, 0), "P_R": (1, 0), "P_B": (0, 1), "P_RB": (1, 1), "P_RBP": (1, 1), "S_RBP": (1, 1), "L_RBP": (1, 1)}
    rows = []
    for cond in conds:
        r_on, b_on = sees[cond]
        c_hat = constrained if r_on else np.full(n, 0.5)
        e_hat = equip if b_on else np.full(n, 0.55)
        gain = 150 + 900 * c_hat * e_hat + rng.normal(0, 250, n)
        normal20 = np.round(base_profit * 1.05 + rng.normal(0, 300, n), -2)
        for i in range(n):
            n20 = float(normal20[i])
            g20 = float(np.round(gain[i] / 50) * 50)
            g30 = float(np.round(gain[i] * 1.2 / 50) * 50)
            late_n = float(np.clip(round(20 + 6 * low_profit[i] + rng.normal(0, 3)), 0, 100))
            late_l = float(np.clip(late_n + round(8 + rng.normal(0, 2)), 0, 100))
            ans = {"normal": {"accept": float(np.clip(round(80 + rng.normal(0, 4)), 0, 100)), "late_if_accepted": late_n,
                              "profit_20m": n20, "profit_30m": n20 + 300},
                   "large": {"accept": float(np.clip(round(62 + rng.normal(0, 5)), 0, 100)), "late_if_accepted": late_l,
                             "profit_20m": n20 + g20, "profit_30m": n20 + 300 + g30},
                   "information": "insufficient" if rng.random() < 0.9 else "sufficient"}
            key = f"{client_key[i]}|{cond}"
            rows.append(dict(client_key=client_key[i], condition=cond, model="synthetic-evaluator",
                             prompt_sha256=prompts[key]["prompt_sha256"], answer=json.dumps(ans)))
    return prof, cl, bench, pd.DataFrame(rows), truth, prompts


def main():
    a = args("Generate synthetic randomized experiments with known effects")
    out = step_dir(a, "02_synthetic")
    P = params()["synthetic"]
    rng = np.random.default_rng(P["seed"])
    frame, g_ans, g_truth, g_df, g_prompts = make_grant(rng, P["grant_applicants"])
    frame.to_pickle(a.work / "grant_frame.pkl")
    g_ans.to_csv(a.work / "synthetic_grant_answers.csv", index=False)
    g_truth.to_pickle(a.work / "synthetic_grant_truth.pkl")
    prof, cl, bench, l_ans, l_truth, l_prompts = make_lending(rng, P["lending_clients"], P["lending_officers"])
    prof.to_pickle(a.work / "lending_prof.pkl")
    cl.to_pickle(a.work / "lending_cl.pkl")
    bench.to_pickle(a.work / "lending_bench.pkl")
    l_ans.to_csv(a.work / "synthetic_lending_answers.csv", index=False)
    l_truth.to_pickle(a.work / "synthetic_lending_truth.pkl")
    with open(out / "example_prompts.md", "w") as fh:
        fh.write("# Example prompts built from synthetic profiles\n\nThese use the same templates as the real runs "
                 "(src/prompts_grant.py, src/prompts_lending.py). All values are synthetic.\n\n")
        for k in ("1|XR|LA", "1|X|F"):
            fh.write(f"## Grant competition, {k}\n\n```\nSYSTEM: {g_prompts[k]['system']}\n\n{g_prompts[k]['user']}\n```\n\n")
        for k in ("1000|P_RB", "1000|S_RBP"):
            fh.write(f"## Lending, {k}\n\n```\nSYSTEM: {l_prompts[k]['system']}\n\n{l_prompts[k]['user']}\n```\n\n")
    write_json(out / "design.json", {
        "grant": {"applicants": len(frame), "winners": int(frame.assigntreat.sum()), "strata": int(frame.strata.nunique()),
                  "responders": int(frame.t_totalemp1.notna().sum()),
                  "mean_true_employment_effect": float(g_truth.tau_emp.mean()),
                  "mean_true_effect_existing": float(g_truth.tau_emp[frame.existing == 1].mean()),
                  "mean_true_effect_new": float(g_truth.tau_emp[frame.existing == 0].mean())},
        "lending": {"clients": len(cl), "profit_rows": len(prof), "officers": int(cl.loid_baseline.nunique()),
                    "blocks": int(cl.block.nunique()), "larger_offer": int(cl.treatment.sum()),
                    "mean_true_profit_gain": float(l_truth.tau_profit.mean()),
                    "mean_true_repayment_effect": float(l_truth.tau_repay.mean())}})
    print("synthetic grant lottery:", frame.shape, "answers", len(g_ans), "| lending:", cl.shape, "answers", len(l_ans))


if __name__ == "__main__":
    main()
