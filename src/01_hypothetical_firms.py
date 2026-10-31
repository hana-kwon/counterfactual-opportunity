# ============================================================
# START DATE   : 2026-10-26
# CODE RUN DATE: 2026-10-31
# TIME TAKEN   : < 1 min
# DESCRIPTION  : Hypothetical firm pairs with a known answer key. Builds the 18 pairs and every prompt
#                (two orders x two formats x four information sets x three questions, plus joint
#                prompts), checks that each rebuilt prompt has the SHA-256 recorded with the cached
#                model answer, scores the cached answers of Claude Sonnet 5.5, Claude Haiku 5.5 and
#                Qwen2.5-3B-Instruct against the key, and writes the accuracy table behind the
#                proposal's Figure 1. Runs in both modes (all inputs are synthetic and in the repo).
#                Each pair is a known world: base employment b, probability p of outside financing
#                for the same equipment, employment d the equipment adds.
#                E[Y(0)] = b + p*d, Y(1) = b + d, added employment = (1 - p)*d.
# INPUT FILES  : data/hypothetical_firms/answers_{claude-sonnet-5-5,claude-haiku-5-5,qwen2.5-3b}.jsonl
# OUTPUT FILES : <run>/01_hypothetical_firms/{cases.json, prompts.jsonl, truth.csv,
#                information_accuracy.csv, summary.json}
# ============================================================
import hashlib
import json
import sys

import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import DATA, args, step_dir, write_json  # noqa: E402
from evallib import jparse, norm_choice, num  # noqa: E402

# ★ CORE LOGIC: four case types separate the candidate decision rules for objective A.
#   reverse   : F=1, L=1, A=2  (higher-p rule wrong, lower-p rule right, larger-L rule wrong)
#   aligned   : F=1, L=1, A=1  (higher-p right, lower-p wrong, larger-L right)
#   size_trap : F=1, L=2, A=1  (higher-p right, lower-p wrong, larger-L wrong)
#   unknown   : firm 1's p is not known even with full information, so F and A are undetermined
# p = None marks "no estimate exists" (unknown type only).
PAIRS = [
    # id, type, industry, firm1 (b, p, d), firm2 (b, p, d)
    ("R1", "reverse", "bakery", (4, .80, 20), (1, .20, 15)),
    ("R2", "reverse", "furniture", (6, .90, 30), (2, .30, 12)),
    ("R3", "reverse", "garments", (10, .70, 18), (3, .10, 10)),
    ("R4", "reverse", "printing", (5, .75, 16), (4, .25, 10)),
    ("R5", "reverse", "poultry_feed", (8, .60, 15), (2, .20, 12)),
    ("S1", "aligned", "bakery", (4, .50, 24), (1, .20, 5)),
    ("S2", "aligned", "furniture", (6, .60, 30), (2, .30, 8)),
    ("S3", "aligned", "garments", (5, .40, 20), (3, .10, 6)),
    ("S4", "aligned", "printing", (8, .70, 40), (2, .20, 8)),
    ("S5", "aligned", "poultry_feed", (3, .50, 18), (1, .25, 6)),
    ("V1", "size_trap", "bakery", (2, .50, 20), (20, .10, 5)),
    ("V2", "size_trap", "furniture", (3, .40, 15), (18, .20, 4)),
    ("V3", "size_trap", "garments", (1, .60, 25), (22, .30, 8)),
    ("V4", "size_trap", "printing", (4, .30, 12), (16, .10, 3)),
    ("V5", "size_trap", "poultry_feed", (2, .55, 30), (25, .15, 10)),
    ("U1", "unknown", "bakery", (4, None, 20), (1, .20, 15)),
    ("U2", "unknown", "furniture", (6, None, 25), (3, .30, 10)),
    ("U3", "unknown", "garments", (2, None, 12), (8, .50, 20)),
]

INDUSTRY = {
    "bakery": ("bakes bread and pastries for shops in its town", "a second industrial oven"),
    "furniture": ("makes wooden furniture to order", "a computer-controlled cutting machine"),
    "garments": ("sews uniforms for schools and employers", "a set of industrial sewing machines"),
    "printing": ("prints labels and packaging for local manufacturers", "a digital label press"),
    "poultry_feed": ("mixes and bags poultry feed for farms", "a pelleting machine"),
}

# Probabilities in the narrative format are written in words for half of the pairs, so the
# narrative is not a copy of the table.
WORDS = {.10: "one in ten", .15: "about one in seven (15 percent)", .20: "one in five", .25: "one in four",
         .30: "three in ten", .40: "two in five", .50: "even (one in two)", .55: "55 percent",
         .60: "three in five", .70: "seven in ten", .75: "three in four", .80: "four in five",
         .90: "nine in ten"}
SYSTEM = ("You are an analyst at a business support organization that evaluates small firms. "
          "Use only the information provided. Answer with a single JSON object and nothing else.")
COMMON = ("Rules that hold in this exercise:\n"
          "- The organization can give one grant of US$50,000, paid now and not repaid, to one of the two firms.\n"
          "- A firm that receives the grant uses it to buy the production equipment named in its file, for certain.\n"
          "- A firm that obtains US$50,000 of outside financing buys the same equipment at the same time. "
          "Money beyond the cost of the equipment creates no additional employment.\n"
          "- There are no other constraints and no effects between the two firms.\n"
          "- Use only the facts stated. Do not assume any value that is not stated.")
OBJECTIVE = {
    "F": ("Question: Which firm has the higher probability of obtaining at least US$50,000 of outside "
          "financing within the next 12 months if it does not receive this grant?",
          "value_1 and value_2: each firm's probability (0-100) of obtaining that financing without the grant"),
    "L": ("Question: The organization wants to fund the firm that will be larger. Which firm would employ "
          "more people 36 months from now if it received the grant?",
          "value_1 and value_2: each firm's expected number of employees 36 months from now if it receives the grant"),
    "A": ("Question: The organization's goal is the largest expected total employment across both firms "
          "36 months from now. Which firm should receive the grant?",
          "value_1 and value_2: the expected number of employees that the grant adds at each firm, "
          "compared with what that firm would have without the grant"),
}
FORMAT_NOTE = ('Return exactly: {{"choice": "1", "2", or "ABSTAIN", "value_1": number or null, '
               '"value_2": number or null, "missing": "facts needed but not stated, or empty"}}\n'
               "Answer ABSTAIN if the stated facts do not determine the answer. {values}.")
INFO = {"X": ("X",), "XR": ("X", "R"), "XB": ("X", "B"), "XRB": ("X", "R", "B")}


def pct(p):
    return f"{round(100 * p)} percent"


def narrative(firm, pair_idx, industry, info):
    b, p, d = firm
    what, equip = INDUSTRY[industry]
    lines = [f"Operations: The firm {what}. With its current resources it expects to employ {b} "
             f"{'person' if b == 1 else 'people'} 36 months from now."]
    if "R" in info:
        if p is None:
            lines.append("Financing: The firm has applied to a commercial bank for a US$50,000 loan for "
                         f"{equip}. The bank has not yet assessed the application, and no estimate of the "
                         "probability of approval exists.")
        else:
            ptxt = WORDS[p] if pair_idx % 2 == 0 else pct(p)
            lines.append("Financing: The firm has applied to a commercial bank for a US$50,000 loan for "
                         f"{equip}. The bank's assessment puts the chance of approval within 12 months at "
                         f"{ptxt}. If the loan is approved, the firm buys the equipment.")
    if "B" in info:
        lines.append(f"Production: The firm holds signed orders it cannot fill with its current equipment. "
                     f"With {equip} it would hire {d} additional workers to fill them; without it these "
                     "hires do not happen.")
    return "\n".join(lines)


def table(f1, f2, info):
    cols = [("Employees 36 months from now with current resources", lambda f: f[0])]
    if "R" in info:
        cols.append(("Probability of US$50,000 outside financing for the equipment within 12 months "
                     "(without the grant)", lambda f: "unknown (no estimate exists)" if f[1] is None else pct(f[1])))
    if "B" in info:
        cols.append(("Additional employees if the equipment is bought", lambda f: f[2]))
    rows = ["| Item | Firm 1 | Firm 2 |", "|---|---|---|"]
    rows += [f"| {name} | {fn(f1)} | {fn(f2)} |" for name, fn in cols]
    return "\n".join(rows)


def truth(f1, f2, obj, info):
    # ★ KEY FUNCTION: the correct answer is a firm only when the stated facts fix the ranking for
    # every admissible value of the missing quantities (p in [0, 1], d >= 0); otherwise ABSTAIN.
    need = {"F": {"R"}, "L": {"B"}, "A": {"R", "B"}}[obj]
    if not need.issubset(set(info)):
        return "ABSTAIN", None, None
    (b1, p1, d1), (b2, p2, d2) = f1, f2
    if obj == "L":
        v1, v2 = b1 + d1, b2 + d2
    elif obj == "F":
        if p1 is None or p2 is None:
            return "ABSTAIN", None, None
        v1, v2 = 100 * p1, 100 * p2
    else:
        if p1 is None or p2 is None:
            known = (1 - p2) * d2 if p1 is None else (1 - p1) * d1
            span = d1 if p1 is None else d2
            # unknown p spans additional employment over [0, d]; determined only if `known` lies outside
            if 0 < known < span:
                return "ABSTAIN", None, None
            raise ValueError("unknown case built with a determined A ranking")
        v1, v2 = (1 - p1) * d1, (1 - p2) * d2
    assert v1 != v2
    return ("1" if v1 > v2 else "2"), round(v1, 4), round(v2, 4)


def single_prompt(pair, order, fmt, info, obj):
    pid, ctype, ind, fa, fb = pair
    f1, f2 = (fa, fb) if order == "orig" else (fb, fa)
    idx = [p[0] for p in PAIRS].index(pid)
    if fmt == "narrative":
        body = (f"Firm 1\n{narrative(f1, idx, ind, INFO[info])}\n\nFirm 2\n{narrative(f2, idx, ind, INFO[info])}")
    else:
        body = table(f1, f2, INFO[info])
    q, vals = OBJECTIVE[obj]
    user = f"{COMMON}\n\n{body}\n\n{q}\n{FORMAT_NOTE.format(values=vals)}"
    return f1, f2, user


def joint_prompt(pair, order, qorder):
    # Both questions in one call, full information, narrative format (sequence check only).
    pid, ctype, ind, fa, fb = pair
    f1, f2 = (fa, fb) if order == "orig" else (fb, fa)
    idx = [p[0] for p in PAIRS].index(pid)
    body = f"Firm 1\n{narrative(f1, idx, ind, INFO['XRB'])}\n\nFirm 2\n{narrative(f2, idx, ind, INFO['XRB'])}"
    qs = [OBJECTIVE[o] for o in qorder]
    user = (f"{COMMON}\n\n{body}\n\nQ1. {qs[0][0][len('Question: '):]}\nQ2. {qs[1][0][len('Question: '):]}\n"
            'Return exactly: {"q1": {"choice": "1", "2", or "ABSTAIN", "value_1": number or null, '
            '"value_2": number or null, "missing": "text or empty"}, "q2": {same fields}}\n'
            f"Answer ABSTAIN if the stated facts do not determine an answer. For Q1, {qs[0][1]}. "
            f"For Q2, {qs[1][1]}.")
    return f1, f2, user



SHORT = {"claude-sonnet-5-5": "Sonnet 5.5", "claude-haiku-5-5": "Haiku 5.5"}


def short(m):
    return SHORT.get(m, "Qwen2.5-3B" if "Qwen" in m else m)


def build():
    prompts, truths = [], []
    for pair in PAIRS:
        pid, ctype = pair[0], pair[1]
        for order in ("orig", "swap"):
            for fmt in ("narrative", "table"):
                for info in INFO:
                    for obj in ("F", "L", "A"):
                        f1, f2, user = single_prompt(pair, order, fmt, info, obj)
                        ans, v1, v2 = truth(f1, f2, obj, INFO[info])
                        key = f"{pid}|{order}|{fmt}|{info}|{obj}|single"
                        prompts.append({"key": key, "system": SYSTEM, "user": user})
                        truths.append(dict(key=key, pair=pid, ctype=ctype, order=order, fmt=fmt, info=info,
                                           obj=obj, mode="single", qpos=1, truth=ans, true_v1=v1, true_v2=v2,
                                           p1=f1[1], p2=f2[1]))
            for qorder in (("F", "A"), ("A", "F")):
                f1, f2, user = joint_prompt(pair, order, qorder)
                key = f"{pid}|{order}|narrative|XRB|{''.join(qorder)}|joint"
                prompts.append({"key": key, "system": SYSTEM, "user": user})
                for pos, obj in enumerate(qorder, start=1):
                    ans, v1, v2 = truth(f1, f2, obj, INFO["XRB"])
                    truths.append(dict(key=key, pair=pid, ctype=ctype, order=order, fmt="narrative", info="XRB",
                                       obj=obj, mode="joint_" + "".join(qorder), qpos=pos, truth=ans,
                                       true_v1=v1, true_v2=v2, p1=f1[1], p2=f2[1]))
    for p in prompts:
        p["prompt_sha256"] = hashlib.sha256((p["system"] + "\n----\n" + p["user"]).encode()).hexdigest()
    return prompts, pd.DataFrame(truths)


def main():
    a = args("Hypothetical firm pairs: build, verify prompt hashes, score cached answers")
    out = step_dir(a, "01_hypothetical_firms")
    prompts, truth_df = build()
    with open(out / "cases.json", "w") as fh:
        json.dump({"pairs": PAIRS, "industry": INDUSTRY, "system": SYSTEM, "common": COMMON,
                   "objective": OBJECTIVE}, fh, indent=1)
    with open(out / "prompts.jsonl", "w") as fh:
        for p in prompts:
            fh.write(json.dumps(p) + "\n")
    truth_df.to_csv(out / "truth.csv", index=False)
    want = {p["key"]: p["prompt_sha256"] for p in prompts}

    rows, hash_check = [], {}
    for path in sorted((DATA / "hypothetical_firms").glob("answers_*.jsonl")):
        n_ok = n = 0
        for line in open(path):
            r = json.loads(line)
            n += 1
            n_ok += want.get(r["key"]) == r["prompt_sha256"]
            ans = jparse(r["text"])
            joint = r["key"].endswith("|joint")
            for qpos in ((1, 2) if joint else (1,)):
                sub = (ans or {}).get(f"q{qpos}") if joint else ans
                ch = norm_choice(sub.get("choice")) if isinstance(sub, dict) else None
                rows.append({"key": r["key"], "qpos": qpos, "model": short(r["model"]), "choice": ch,
                             "v1": num(sub.get("value_1")) if isinstance(sub, dict) else None,
                             "v2": num(sub.get("value_2")) if isinstance(sub, dict) else None})
        hash_check[path.name] = dict(answers=n, prompt_hash_matches=n_ok)
        # ★ IMPORTANT: every cached answer must belong to a prompt this code rebuilds byte for byte.
        assert n_ok == n == len(prompts), (path.name, n_ok, n, len(prompts))
    truth_df["truth"] = truth_df["truth"].astype(str)
    d = pd.DataFrame(rows).merge(truth_df, on=["key", "qpos"], how="left", validate="many_to_one")
    assert d.truth.notna().all()
    d["correct"] = (d.choice == d.truth).astype(float)
    # Objective A (employment the grant adds), single calls: the proposal's Figure 1 cells (72 answers each)
    sub = d[(d.obj == "A") & (d["mode"] == "single")]
    agg = sub.groupby(["model", "info"]).agg(
        n=("correct", "size"), pct_correct=("correct", lambda s: 100 * s.mean()),
        pct_abstain=("choice", lambda s: 100 * (s == "ABSTAIN").mean())).reset_index()
    agg.to_csv(out / "information_accuracy.csv", index=False)
    write_json(out / "summary.json", {"prompts": len(prompts), "hash_check": hash_check,
                                      "accuracy_objective_A": agg.to_dict("records")})
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
