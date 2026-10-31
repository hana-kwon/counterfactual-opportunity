# ============================================================
# START DATE   : 2026-10-27
# CODE RUN DATE: 2026-10-31
# TIME TAKEN   : (library, imported by the numbered steps)
# DESCRIPTION  : Shared evaluation code. The same functions score AI rankings against the two
#                randomized experiments (full path) and against the synthetic experiments (demo
#                path): allocation by score within randomization strata, policy value relative to
#                the lottery with a paired stratified bootstrap, treatment x score interaction
#                regressions with stratum fixed effects (McKenzie 2017, Appendix Table A18 form),
#                and, for the lending design, contrasts of interaction coefficients with
#                influence-function standard errors clustered by randomization block, the
#                Horvitz-Thompson policy contrast, and the expected overlap of two selections.
# INPUT FILES  : none (imported)
# OUTPUT FILES : none (imported)
# ============================================================
import json
import math
import re
from math import erfc, sqrt

import numpy as np
import pandas as pd
from scipy.stats import norm

Z95 = float(norm.ppf(0.975))
Z80 = float(norm.ppf(0.80))


# ============================================================
# Part 1. Grant lottery design (strata, fixed number of winners per stratum)
# ============================================================

def random_selection(strata, k_by_stratum, n_by_stratum):
    """Lottery benchmark: every applicant in stratum s is selected with probability k_s / n_s."""
    pi = np.zeros(len(strata))
    for s, k in k_by_stratum.items():
        pi[strata == s] = k / n_by_stratum[s]
    return pi


# ★ CORE LOGIC: expected selection probability under random tie-breaking. Within each stratum the
# rule awards the same number of grants the lottery awarded. Above the cutoff score: 1. At the cutoff:
# (places left) / (applicants tied there). Below: 0. The policy value is linear in the selection
# indicator, so this is the exact expectation over tie-breaks and needs no random seed.
def expected_selection(score, strata, k_by):
    pi = np.zeros(len(score))
    sc = np.where(np.isnan(score), -np.inf, score)
    for s, k in k_by.items():
        idx = np.flatnonzero(strata == s)
        if k <= 0:
            continue
        if k >= len(idx):
            pi[idx] = 1.0
            continue
        v = np.sort(sc[idx])[::-1]
        cut = v[k - 1]
        above = sc[idx] > cut
        at = sc[idx] == cut
        pi[idx[above]] = 1.0
        left = k - int(above.sum())
        pi[idx[at]] = left / int(at.sum())
    return pi


def tie_share(pi, k_total):
    """Share of awarded places that a tie-break settles (selection probability strictly between 0 and 1)."""
    m = (pi > 0) & (pi < 1)
    return float(pi[m].sum() / k_total) if m.any() else 0.0


# ★ CORE LOGIC: V = sum_s (n_s/N) * [ mean over treated responders of pi*Y
#                                   + mean over control responders of (1-pi)*Y ].
# Within stratum x arm the assignment is random, so each responder mean is the Hajek inverse-probability
# estimate of that arm's mean under the rule. Non-response is assumed random within stratum x arm.
def policy_value(pi, cells, y, resp, weights, n_total, draws=None, rep=None, rw=None):
    total = 0.0
    for (s, arm), idx in cells.items():
        take = idx if draws is None else idx[draws[(s, arm)][rep]]
        r = resp[take]
        if not r.any():
            take = idx
            r = resp[take]
        q = pi[take] if arm == 1 else (1.0 - pi[take])
        num = q[r] * y[take][r]
        if rw is None:
            val = num.mean()
        else:
            ww = rw[take][r]
            val = (num * ww).sum() / ww.sum()
        total += weights[s] * val
    return total


def grant_design(strata, arm):
    """Strata sizes, lottery winners per stratum, stratum weights and stratum x arm cells.
    ★ IMPORTANT: the dict orders (value_counts order, then arm 0/1) fix the order of the bootstrap
    draws, so they must be built exactly this way for the reference numbers to reproduce."""
    n_total = len(strata)
    n_by = pd.Series(strata).value_counts().to_dict()
    k_by = pd.Series(strata[arm == 1]).value_counts().to_dict()
    k_by = {s: int(k_by.get(s, 0)) for s in n_by}
    weights = {s: n_by[s] / n_total for s in n_by}
    cells = {(s, a): np.flatnonzero((strata == s) & (arm == a)) for s in n_by for a in (0, 1)}
    return n_total, n_by, k_by, weights, cells


def paired_policy_bootstrap(pis, cells, y, resp, weights, n_total, nboot, seed, reference):
    """Point policy values and paired bootstrap contrasts against `reference`.
    ★ IMPORTANT: one set of resampled indices (within stratum x arm cells) is shared by every rule,
    so the contrasts are paired."""
    order = list(pis)
    bs = np.random.default_rng(seed)
    draws = {c: bs.integers(0, len(idx), size=(nboot, len(idx))) for c, idx in cells.items()}
    point = {k: policy_value(pis[k], cells, y, resp, weights, n_total) for k in order}
    boot = {k: np.array([policy_value(pis[k], cells, y, resp, weights, n_total, draws, b) for b in range(nboot)])
            for k in order}
    rows = []
    for k in order:
        if k == reference:
            continue
        dd = boot[k] - boot[reference]
        se = dd.std(ddof=1)
        rows.append({"rule": k, "value": point[k], "value_reference": point[reference],
                     "diff": point[k] - point[reference],
                     "ci_lo": float(np.percentile(dd, 2.5)), "ci_hi": float(np.percentile(dd, 97.5)),
                     "boot_se": float(se), "mde_80pct": float((Z95 + Z80) * se),
                     "excludes_zero": bool(np.percentile(dd, 2.5) > 0 or np.percentile(dd, 97.5) < 0)})
    return pd.DataFrame(rows), point


def tlearner_scores(X, y, w, resp, rng, seed):
    """2-fold cross-fit T-learner (gradient boosting). Trains on responders only; returns tau-hat for
    everyone. Used as the statistical-model benchmark."""
    from sklearn.ensemble import GradientBoostingRegressor
    fold = rng.integers(0, 2, size=len(y))
    tau = np.zeros(len(y))
    for f in (0, 1):
        tr = fold != f
        out = {}
        for arm in (0, 1):
            m = tr & (w == arm) & resp
            g = GradientBoostingRegressor(random_state=seed + arm + 10 * f)
            g.fit(X[m], y[m])
            out[arm] = g.predict(X[fold == f])
        tau[fold == f] = out[1] - out[0]
    return tau


def areg(y, X, strata):
    """OLS with strata dummies and HC1, as Stata `areg ..., a(strata) robust`."""
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    y, X, s = y[ok], X[ok, :], strata[ok]
    D = pd.get_dummies(pd.Series(s), drop_first=True).to_numpy(float)
    M = np.column_stack([X, D, np.ones(len(y))])
    b, *_ = np.linalg.lstsq(M, y, rcond=None)
    e = y - M @ b
    XtXi = np.linalg.pinv(M.T @ M)
    n, k = len(y), np.linalg.matrix_rank(M)
    V = XtXi @ (M.T @ (M * (e ** 2)[:, None])) @ XtXi * (n / (n - k))
    return b[:X.shape[1]], np.sqrt(np.diag(V))[:X.shape[1]], n


def within_coef(y, X, strata):
    """Point estimate only, via the within-strata transform (same coefficients as areg). Bootstrap use."""
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    y, X, s = y[ok], X[ok, :], strata[ok]
    df = pd.DataFrame(np.column_stack([y, X]))
    dm = df - df.groupby(s).transform("mean")
    b, *_ = np.linalg.lstsq(dm.iloc[:, 1:].to_numpy(), dm.iloc[:, 0].to_numpy(), rcond=None)
    return b


def pval(b, se):
    return erfc(abs(b / se) / sqrt(2))


def zscore(s):
    return (s - s.mean()) / s.std(ddof=0)


def grant_scores_from_answers(answers):
    """One row per applicant from the model answers (one answer per applicant x condition x task).
    A = added employment (emp_with - emp_without), L = employment with the grant,
    F = probability of outside financing without the grant."""
    rows = {}
    for r in answers.itertuples(index=False):
        j = json.loads(r.answer)
        d = rows.setdefault(int(r.uid), {})
        if r.task == "F":
            d[f"F_{r.condition}"] = float(j["p_outside_financing"])
        else:
            d[f"L_{r.condition}"] = float(j["emp_with"])
            d[f"A_{r.condition}"] = float(j["emp_with"]) - float(j["emp_without"])
            d[f"info_{r.condition}"] = j.get("information")
    s = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    s.index.name = "uid"
    return s.reset_index()


# ============================================================
# Part 2. Lending design (loan officer groups, randomization blocks, assignment probability 0.5)
# ============================================================

ANSWER_KEYS = ("accept", "late_if_accepted", "profit_20m", "profit_30m")


def parse_answer(text):
    """★ KEY FUNCTION: validate one lending answer. Raises ValueError on anything outside the format."""
    s = (text or "").strip()
    if s.startswith("```"):
        s = s.strip("`")
        s = s[s.find("{"):]
    j = json.loads(s)
    if not isinstance(j, dict) or set(j) != {"normal", "large", "information"}:
        raise ValueError("top-level keys")
    out = {}
    for offer in ("normal", "large"):
        o = j[offer]
        if not isinstance(o, dict) or set(o) != set(ANSWER_KEYS):
            raise ValueError(f"{offer} keys")
        for k in ANSWER_KEYS:
            v = o[k]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValueError(f"{offer}.{k} not a number")
            if k in ("accept", "late_if_accepted") and not 0 <= v <= 100:
                raise ValueError(f"{offer}.{k} outside 0-100")
            out[f"{offer}_{k}"] = float(v)
    if j["information"] not in ("sufficient", "insufficient"):
        raise ValueError("information")
    out["information"] = j["information"]
    return out


def scores_from(a):
    """PI = mean over the two horizons of (large-offer profit - normal-offer profit) (profit-gain score);
    RHO = reduction in P(any late installment) from the large offer, where a declined offer has no late
    installment (repayment score); LEVEL = expected profit under the large offer (profit-level score)."""
    pi20 = a["large_profit_20m"] - a["normal_profit_20m"]
    pi30 = a["large_profit_30m"] - a["normal_profit_30m"]
    late_n = a["normal_accept"] / 100 * a["normal_late_if_accepted"] / 100
    late_l = a["large_accept"] / 100 * a["large_late_if_accepted"] / 100
    return {"PI_20": pi20, "PI_30": pi30, "PI": (pi20 + pi30) / 2, "RHO": late_n - late_l,
            "LEVEL": (a["large_profit_20m"] + a["large_profit_30m"]) / 2}


def lending_scores_from_answers(answers):
    """Returns (scores dict keyed by (condition, score type) -> Series by client_key, flat table)."""
    rows = []
    for r in answers.itertuples(index=False):
        a = parse_answer(r.answer)
        d = dict(client_key=int(r.client_key), cond=r.condition, information=a["information"],
                 normal_accept=a["normal_accept"], large_accept=a["large_accept"],
                 normal_late=a["normal_late_if_accepted"], large_late=a["large_late_if_accepted"])
        d.update(scores_from(a))
        rows.append(d)
    t = pd.DataFrame(rows)
    out = {}
    for cond, sub in t.groupby("cond"):
        sub = sub.set_index("client_key").sort_index()
        for k in ("PI", "RHO", "LEVEL", "PI_20", "PI_30"):
            out[(cond, k)] = sub[k].astype(float)
    return out, t


class NotEstimable(Exception):
    pass


def standardize_common(score_cols):
    """★ CORE LOGIC: the analysis sample of a contrast = clients with every involved score valid; each
    score is standardized (mean 0, SD 1) over that common set before outcomes are used. Identical scores
    therefore give identical z, and a contrast between them is exactly 0."""
    df = pd.concat(score_cols, axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    z = {}
    for c in df.columns:
        sd = df[c].std(ddof=0)
        if not sd > 0:
            raise NotEstimable(f"score {c} has no variation in the common sample")
        z[c] = (df[c] - df[c].mean()) / sd
    return pd.DataFrame(z, index=df.index)


def _demean(x, g, ng, cnt):
    return x - (np.bincount(g, x, ng) / cnt)[g]


def interaction_fits(rows, y, z_by_client, controls=(), clustered=True, cluster="block"):
    """For each score column, OLS of y on T, T*z, z (+controls) absorbing loan officer fixed effects.
    Returns coefficients on T*z, cluster-summed influence functions (G x k) and the dof factor."""
    d = rows[rows.client_key.isin(z_by_client.index)]
    d = d[np.isfinite(d[y].to_numpy(float))]
    g = pd.factorize(d.loid_baseline)[0]
    ng = g.max() + 1
    cnt = np.bincount(g, minlength=ng).astype(float)
    cl = pd.factorize(d.client_key if cluster == "client" else d[cluster])[0]
    G = cl.max() + 1
    yy = _demean(d[y].to_numpy(float), g, ng, cnt)
    t = d.treatment.to_numpy(float)
    C = [_demean(d[c].to_numpy(float), g, ng, cnt) for c in controls]
    b, P = [], []
    for col in z_by_client.columns:
        z = z_by_client[col].reindex(d.client_key).to_numpy(float)
        X = np.column_stack([_demean(t, g, ng, cnt), _demean(z, g, ng, cnt)] + C)
        x = _demean(t * z, g, ng, cnt)
        full = np.column_stack([x, X])
        coef = np.linalg.lstsq(full, yy, rcond=None)[0]
        e = yy - full @ coef
        xr = x - X @ np.linalg.lstsq(X, x, rcond=None)[0]       # FWL residual of T*z
        b.append(coef[0])
        P.append(np.bincount(cl, xr * e / (xr @ xr), G))
    n, k = len(d), 3 + len(controls) + ng
    adj = G / (G - 1) * (n - 1) / (n - k) if (clustered or cluster != "client") else n / (n - k)
    return np.array(b), np.column_stack(P), adj, dict(n_rows=n, n_clusters=G, n_clients=d.client_key.nunique())


# outcome name -> (row set, column, controls, clustered)
OUTCOME_SPEC = {"profit": ("prof", "e_profits_all", ("round",), True),
                "repay": ("cl", "perfect_repayment", (), False)}


def contrast(prof, cl, scores, spec, n_tests=1, cluster="block"):
    """★ KEY FUNCTION: a contrast sum_k w_k * beta_k of treatment x score coefficients with the joint
    influence-function SE clustered by randomization block. With n_tests > 1 the Bonferroni interval
    (two-sided, n_tests tests) and the matching minimum detectable difference are reported."""
    rows_name, y, ctrl, clustered = OUTCOME_SPEC[spec["outcome"]]
    rows = prof if rows_name == "prof" else cl
    cols = [scores[(c, s)].rename(f"{c}|{s}") for _, c, s in spec["terms"]]
    z = standardize_common(cols)
    b, P, adj, info = interaction_fits(rows, y, z, ctrl, clustered, cluster)
    w = np.array([t[0] for t in spec["terms"]], float)
    est = float(w @ b)
    se = float(np.sqrt(adj * ((P @ w) @ (P @ w))))
    zb = float(norm.ppf(1 - 0.05 / (2 * n_tests)))
    return dict(outcome=spec["outcome"], estimate=est, se=se, lo95=est - Z95 * se, hi95=est + Z95 * se,
                lo_adj=est - zb * se, hi_adj=est + zb * se, z_adj=zb, mde_adj=(zb + Z80) * se,
                n_common_clients=len(z), coefs={k: float(v) for k, v in zip(z.columns, b)}, **info)


def selection_probs(score, group):
    """★ CORE LOGIC: each loan officer group selects n/2 expected slots by score (an odd group's middle
    slot is a half slot). Clients tied at the cutoff share the remaining slots equally."""
    s = np.asarray(score, float)
    return _selection_probs_idx(s, _group_index(group))


def _group_index(group):
    g = np.asarray(group)
    order = np.argsort(g, kind="stable")
    cuts = np.flatnonzero(np.diff(g[order])) + 1
    return np.split(order, cuts)


def _selection_probs_idx(s, groups):
    p = np.zeros(len(s))
    tie_slots = half_slots = 0.0
    for idx in groups:
        k = len(idx) / 2.0
        vals, inv, cnt = np.unique(-s[idx], return_inverse=True, return_counts=True)
        before = np.concatenate([[0], np.cumsum(cnt)[:-1]])
        rem = k - before
        share = np.clip(rem / cnt, 0.0, 1.0)
        p[idx] = share[inv]
        b = np.flatnonzero((rem > 0) & (rem < cnt))
        for j in b:
            if cnt[j] >= 2:
                tie_slots += rem[j]
            else:
                half_slots += rem[j]
    return p, tie_slots, half_slots


def policy_contrast_ht(d, y, t, block, p_assign=0.5):
    """★ KEY FUNCTION: Horvitz-Thompson estimate of (1/N) sum_i d_i (Y_i(1) - Y_i(0)), with
    d = pi_A - pi_B (or pi - 0.5 for the comparison with random assignment) and the design's assignment
    probability, SE clustered by randomization block. Normal interval."""
    d, y, t, block = (np.asarray(x) for x in (d, y, t, block))
    psi = d * y * (t / p_assign - (1 - t) / (1 - p_assign))
    n = len(psi)
    codes = pd.factorize(block)[0]
    G = codes.max() + 1
    est = float(psi.mean())
    csum = np.bincount(codes, psi, G)
    csize = np.bincount(codes, minlength=G)
    se = float(np.sqrt(G / (G - 1) * ((csum - csize * est) ** 2).sum()) / n)
    return dict(estimate=est, se=se, lo95=est - Z95 * se, hi95=est + Z95 * se, n=n, clusters=G)


def overlap_stats(score_a, score_b, group, reps=2000, seed=20261029, min_size=2):
    """Expected overlap of two independently implemented tie-aware selections, O = sum(pA*pB)/sum(pA),
    with the value for identical untied rankings, 0.5 for random rankings, and a within-group
    permutation reference that keeps B's ties."""
    a, b, g = (np.asarray(x) for x in (score_a, score_b, group))
    sizes = pd.Series(g).map(pd.Series(g).value_counts()).to_numpy()
    keep = sizes >= min_size
    a, b, g = a[keep], b[keep], g[keep]
    pa = selection_probs(a, g)[0]
    pb = selection_probs(b, g)[0]
    k = pa.sum()
    obs = float(pa @ pb / k)
    n_by = pd.Series(g).value_counts()
    upper = float((sum(n // 2 for n in n_by) + 0.25 * sum(n % 2 for n in n_by)) / k)
    rng = np.random.default_rng(seed)
    groups = _group_index(g)
    null = np.empty(reps)
    for r in range(reps):
        bp = b.astype(float).copy()
        for ix in groups:
            bp[ix] = b[rng.permutation(ix)]
        null[r] = _selection_probs_idx(bp, groups)[0] @ pa / k
    return dict(overlap=obs, identical_upper=upper, random_expectation=0.5, perm_mean=float(null.mean()),
                perm_lo=float(np.percentile(null, 2.5)), perm_hi=float(np.percentile(null, 97.5)),
                clients=int(keep.sum()))


def itt(rows, y, ctrl=(), cluster="block"):
    """Average effect of the larger offer: y on T (+controls) absorbing officer fixed effects, clustered SE."""
    d = rows[np.isfinite(rows[y].to_numpy(float))]
    g = pd.factorize(d.loid_baseline)[0]
    ng = g.max() + 1
    cnt = np.bincount(g, minlength=ng).astype(float)
    dm = lambda x: x - (np.bincount(g, x, ng) / cnt)[g]  # noqa: E731
    X = np.column_stack([dm(d.treatment.to_numpy(float))] + [dm(d[c].to_numpy(float)) for c in ctrl])
    yy = dm(d[y].to_numpy(float))
    b = np.linalg.lstsq(X, yy, rcond=None)[0]
    e = yy - X @ b
    codes = pd.factorize(d.client_key if cluster == "client" else d[cluster])[0]
    G = codes.max() + 1
    XtXi = np.linalg.pinv(X.T @ X)
    S = np.column_stack([np.bincount(codes, X[:, j] * e, G) for j in range(X.shape[1])])
    n, k = len(d), X.shape[1] + ng
    V = XtXi @ (S.T @ S) @ XtXi * G / (G - 1) * (n - 1) / (n - k)
    return float(b[0]), float(np.sqrt(V[0, 0])), n


# ============================================================
# Part 3. Hypothetical firm pairs (answer parsing)
# ============================================================

def jparse(text):
    if text is None:
        return None
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    m = re.search(r"\{.*\}", t, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def norm_choice(c):
    s = str(c).strip().upper().replace("FIRM", "").strip()
    return s if s in ("1", "2", "ABSTAIN") else None


def num(x):
    try:
        return float(re.sub(r"[$,%\s]", "", x)) if isinstance(x, str) else (None if x is None else float(x))
    except ValueError:
        return None
