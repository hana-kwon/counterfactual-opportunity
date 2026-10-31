# ============================================================
# START DATE   : 2026-10-29
# CODE RUN DATE: 2026-10-31
# TIME TAKEN   : < 1 min
# DESCRIPTION  : Draws the four figures from the aggregate outputs of steps 01, 05, 06 and 07 (no new
#                estimation): (1) correct decisions on hypothetical firm pairs by information and model;
#                (2) allocation gains over the lottery for every rule; (3) grant x score interactions by
#                firm type; (4) lending: (a) profit gain of score-based allocation over random assignment,
#                (b) forecast vs measured effect of the larger offer on repayment. In demo mode the
#                panels show synthetic data and the known true values.
# INPUT FILES  : <run>/01_hypothetical_firms/information_accuracy.csv, <run>/05_grant_policy_value/policy_contrasts.csv,
#                <run>/06_grant_heterogeneity/interactions.csv, <run>/07_lending_evaluation/{policy_values.csv,
#                forecasts_by_condition.csv, summary.json}
# OUTPUT FILES : <run>/09_figures/fig{1,2,3,4}_*.png and .pdf
# ============================================================
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import args, step_dir  # noqa: E402

LABELS = {"Claude A, X": "AI added jobs: application only", "Claude L, X": "AI employment level: application only",
          "Claude F, X": "AI funding chance: application only", "Claude A, XR": "AI added jobs: + financing facts",
          "Claude L, XR": "AI employment level: + financing facts", "Claude F, XR": "AI funding chance: + financing facts",
          "Business plan score": "Judges' business-plan score", "T-learner (cross-fit)": "Statistical model (cross-fit)"}
COND = {"P0": "baseline facts", "P_R": "+ financing alternatives", "P_B": "+ loan use", "P_RB": "+ both",
        "P_RBP": "+ psychological measures", "S_RBP": "repayment objective", "L_RBP": "profit-level objective"}


def save(fig, out, stem):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    a = args("Figures from aggregate outputs")
    out = step_dir(a, "09_figures")
    demo = a.mode == "demo"
    tag = " (synthetic data)" if demo else ""
    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})

    # Figure 1: hypothetical firm pairs (real cached model answers in both modes)
    acc = pd.read_csv(a.run_dir / "01_hypothetical_firms" / "information_accuracy.csv")
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    infos = ["X", "XB", "XR", "XRB"]
    for model, mk, ls in (("Sonnet 5.5", "o", "-"), ("Haiku 5.5", "s", "--"), ("Qwen2.5-3B", "^", ":")):
        s = acc[acc.model == model].set_index("info").loc[infos]
        ax.plot(range(4), s.pct_correct, marker=mk, linestyle=ls, color="black", label=model)
    ax.set_xticks(range(4), ["Baseline\nonly", "Investment\neffect", "Financing\nalternative", "Both"])
    ax.set_ylim(-4, 106)
    ax.set_ylabel("Correct decisions (%)")
    ax.set_title("Hypothetical firm pairs: 72 answers per cell", fontsize=9)
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.08))
    save(fig, out, "fig1_information_accuracy")

    # Figure 2: grant allocation gains over the lottery
    pc = pd.read_csv(a.run_dir / "05_grant_policy_value" / "policy_contrasts.csv")
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for i, r in enumerate(pc.itertuples()):
        main = r.rule == "Claude A, XR"
        ax.errorbar(r.diff, i, xerr=[[r.diff - r.ci_lo], [r.ci_hi - r.diff]], fmt="o", color="black",
                    mfc="black" if main else "white", capsize=2)
        if demo:
            ax.plot(r.true_gain, i, marker="x", color="gray", ms=7)
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_yticks(range(len(pc)), [LABELS.get(x, x) for x in pc.rule])
    ax.invert_yaxis()
    ax.set_xlabel("Jobs per applicant relative to the lottery (paired 95% bootstrap interval)")
    ax.set_title("Grant allocation by score vs lottery" + tag + ("; x = true gain" if demo else ""), fontsize=9)
    save(fig, out, "fig2_grant_policy")

    # Figure 3: interactions
    it = pd.read_csv(a.run_dir / "06_grant_heterogeneity" / "interactions.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6), sharey=True)
    for ax, (o, ttl) in zip(axes, (("t_totalemp1", "(a) Employment"), ("t_IHSprofits", "(b) IHS profits"))):
        sub = it[it.outcome == o].reset_index(drop=True)
        for i, r in enumerate(sub.itertuples()):
            ax.errorbar(r.inter_per_sd, i, xerr=[[r.inter_per_sd - r.lo], [r.hi - r.inter_per_sd]], fmt="o",
                        color="black", mfc="black" if r.score == "A_XR" else "white", capsize=2)
            if demo:
                ax.plot(r.true_slope, i, marker="x", color="gray", ms=7)
        ax.axvline(0, color="gray", lw=0.8)
        ax.set_title(ttl, fontsize=9)
        ax.set_yticks(range(len(sub)), [f"{'Existing' if r.existing else 'New'}: {'AI score' if r.score == 'A_XR' else 'Judges'}"
                                        for r in sub.itertuples()])
        ax.set_xlabel("Grant x score, per SD")
    axes[0].invert_yaxis()
    fig.suptitle("Do scores separate grant effects?" + tag, fontsize=9, y=1.03)
    save(fig, out, "fig3_grant_heterogeneity")

    # Figure 4: lending
    pv = pd.read_csv(a.run_dir / "07_lending_evaluation" / "policy_values.csv")
    fc = pd.read_csv(a.run_dir / "07_lending_evaluation" / "forecasts_by_condition.csv")
    sm = json.loads((a.run_dir / "07_lending_evaluation" / "summary.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    keys = [("P_RBP:PI", "AI profit gain, all information"), ("P0:PI", "AI profit gain, baseline facts"),
            ("BKO:ML", "Statistical benchmark" if demo else "Original study's ML\n(fitted to these outcomes)")]
    sub = pv[(pv.outcome == "profit") & (pv.versus == "random")].set_index("rule")
    for i, (k, lab) in enumerate(keys):
        r = sub.loc[k]
        axes[0].errorbar(r.estimate, i, xerr=[[r.estimate - r.lo95], [r.hi95 - r.estimate]], fmt="o", color="black",
                         mfc="black" if i == 0 else "white", capsize=2)
        if demo:
            axes[0].plot(r.true_gain, i, marker="x", color="gray", ms=7)
    axes[0].axvline(0, color="gray", lw=0.8)
    axes[0].set_yticks(range(3), [k[1] for k in keys])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Monthly profit gain vs random (95% interval)")
    axes[0].set_title("(a) Allocation by score", fontsize=9)
    axes[1].barh(range(len(fc)), fc.mean_RHO_pp, color="white", edgecolor="black")
    itt = sm["itt_perfect_repayment"]
    axes[1].errorbar(100 * itt["estimate"], len(fc), xerr=196 * itt["se"], fmt="s", color="black", capsize=2)
    axes[1].set_yticks(range(len(fc) + 1), [COND.get(c, c) for c in fc.cond] + ["Measured effect"])
    axes[1].invert_yaxis()
    axes[1].axvline(0, color="gray", lw=0.8)
    axes[1].set_xlabel("Larger-offer effect on repaying\nwithout delay (pp)")
    axes[1].set_title("(b) Forecast vs measured", fontsize=9)
    fig.suptitle("Lending experiment" + tag, fontsize=9)
    fig.tight_layout()
    save(fig, out, "fig4_lending")
    print("figures written to", out)


if __name__ == "__main__":
    main()
