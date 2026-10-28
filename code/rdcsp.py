"""
rdcsp.py -- Exact study contributions for network meta-analysis of sparse binary
outcomes on the risk-difference scale (exact-binomial contrast-space projection).

Model
-----
Study k has arms u in S_k with independent counts R_ku ~ Binomial(n_ku, p_ku).
Consistency on the risk scale: p_ku = alpha_k + mu_u.  Target effect for a:b is
theta_ab = mu_b - mu_a (risk under b minus risk under a).

Estimator
---------
Weighted least squares of the observed proportions on study indicators and
treatment indicators with design-fixed weights w_ku (default: w_ku = n_ku).
Equivalently (Laplacian form):
    K_k = diag(w_k) - w_k w_k' / (1'w_k),   L = sum_k T_k' K_k T_k,
    b_{t,k} = K_k T_k L^+ (e_a - e_b)        (study-arm currents),
    C_{t,k} = - b_{t,k}' phat_k,             theta_hat_t = sum_k C_{t,k}.
Because w depends only on the design, the coefficients are outcome-independent
and the estimator is linear in the observed proportions, hence exactly unbiased
under the model with no distributional assumption beyond E[phat] = p.

All functions are plain numpy; no normal approximation is used anywhere except
where explicitly labelled 'normal' for comparison.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from itertools import combinations
from math import comb, log, exp, lgamma


# --------------------------------------------------------------------------- #
# Data structure
# --------------------------------------------------------------------------- #
class Network:
    """Arm-level network.  df columns: study, treatment, events, n."""

    def __init__(self, df: pd.DataFrame, weights: str | np.ndarray = "n"):
        df = df.copy()
        df["study"] = df["study"].astype(str)
        df["treatment"] = df["treatment"].astype(str)
        self.df = df.reset_index(drop=True)
        self.studies = list(dict.fromkeys(self.df["study"]))
        self.trts = sorted(self.df["treatment"].unique())
        self.T = len(self.trts)
        self.tix = {t: i for i, t in enumerate(self.trts)}
        self.n = self.df["n"].to_numpy(float)
        self.r = self.df["events"].to_numpy(float)
        self.phat = self.r / self.n
        if isinstance(weights, str) and weights == "n":
            self.w = self.n.copy()
        else:
            self.w = np.asarray(weights, float)
        self.arm_study = self.df["study"].to_numpy()
        self.arm_trt = self.df["treatment"].to_numpy()
        self.rows = {k: np.where(self.arm_study == k)[0] for k in self.studies}
        self._build()

    # Laplacians ---------------------------------------------------------- #
    def _build(self):
        self.K = {}
        self.Tmat = {}
        L = np.zeros((self.T, self.T))
        for k in self.studies:
            idx = self.rows[k]
            wk = self.w[idx]
            if len(idx) > 1 and wk.sum() > 0:
                Kk = np.diag(wk) - np.outer(wk, wk) / wk.sum()
            else:                      # single-arm study: no comparative information
                Kk = np.zeros((len(idx), len(idx)))
            Tk = np.zeros((len(idx), self.T))
            for j, i in enumerate(idx):
                Tk[j, self.tix[self.arm_trt[i]]] = 1.0
            self.K[k], self.Tmat[k] = Kk, Tk
            L += Tk.T @ Kk @ Tk
        self.L = L
        self.Lp = np.linalg.pinv(L)
        # connectivity check (rank T-1 for a connected network)
        self.connected = np.linalg.matrix_rank(L) == self.T - 1

    def e(self, a: str, b: str) -> np.ndarray:
        v = np.zeros(self.T)
        v[self.tix[a]] = 1.0
        v[self.tix[b]] = -1.0
        return v

    # Study currents and contributions ------------------------------------- #
    def currents(self, a: str, b: str) -> dict:
        """Signed study-arm currents b_{t,k} for target a:b (flow a -> b)."""
        v = self.Lp @ self.e(a, b)
        return {k: self.K[k] @ self.Tmat[k] @ v for k in self.studies}

    def contributions(self, a: str, b: str, phat: np.ndarray | None = None) -> pd.DataFrame:
        """Exact additive study contributions to the estimate of theta_ab = p_b - p_a.

        Returns one row per study: direct weight d, indirect throughput WI,
        throughput W, and additive pieces C_D, C_I, C (all sum to the estimate).
        """
        phat = self.phat if phat is None else phat
        cur = self.currents(a, b)
        out = []
        for k in self.studies:
            idx = self.rows[k]
            bk = cur[k]
            pk = phat[idx]
            C = -bk @ pk
            W = 0.5 * np.abs(bk).sum()
            arms = list(self.arm_trt[idx])
            s = np.maximum(bk, 0)           # inflows (supplies)
            t = np.maximum(-bk, 0)          # outflows (demands)
            if a in arms and b in arms:
                ia, ib = arms.index(a), arms.index(b)
                d = min(s[ia], t[ib])       # CSP direct-first extraction
                Dk = pk[ib] - pk[ia]        # study's own direct RD estimate
                CD = d * Dk
                bI = bk.copy()
                bI[ia] -= d
                bI[ib] += d
            else:
                d, Dk, CD, bI = 0.0, np.nan, 0.0, bk.copy()
            CI = -bI @ pk
            out.append(dict(study=k, arms="/".join(arms), d=d, WI=W - d, W=W,
                            direct_est=Dk, C_D=CD, C_I=CI, C=C))
        res = pd.DataFrame(out)
        res.attrs["estimate"] = float(res["C"].sum())
        res.attrs["q"] = float(1.0 - res["d"].sum())
        return res

    def estimate(self, a: str, b: str, phat: np.ndarray | None = None) -> float:
        phat = self.phat if phat is None else phat
        cur = self.currents(a, b)
        return float(sum(-cur[k] @ phat[self.rows[k]] for k in self.studies))

    def coefficient_vector(self, a: str, b: str) -> np.ndarray:
        """Arm-level coefficients c such that theta_hat = c' phat."""
        cur = self.currents(a, b)
        c = np.zeros(len(self.n))
        for k in self.studies:
            c[self.rows[k]] = -cur[k]
        return c

    def exact_variance(self, a: str, b: str, p: np.ndarray | None = None) -> float:
        """Exact variance of the linear estimator at arm risks p (plug-in phat by default)."""
        p = self.phat if p is None else p
        c = self.coefficient_vector(a, b)
        return float(np.sum(c ** 2 * p * (1 - p) / self.n))

    # Design matrices ------------------------------------------------------ #
    def design_consistency(self) -> np.ndarray:
        """Study intercepts + treatment effects (reference = first treatment)."""
        S = np.array([[1.0 if s == k else 0.0 for k in self.studies] for s in self.arm_study])
        Tm = np.array([[1.0 if t == u else 0.0 for u in self.trts[1:]] for t in self.arm_trt])
        return np.hstack([S, Tm])

    def design_specific(self) -> np.ndarray:
        """Study intercepts + treatment effects that differ by design (treatment set)."""
        design = {k: "|".join(sorted(set(self.arm_trt[self.rows[k]]))) for k in self.studies}
        des = sorted(set(design.values()))
        cols = []
        for d in des:
            trs = d.split("|")
            for t in trs[1:]:
                cols.append((d, t))
        S = np.array([[1.0 if s == k else 0.0 for k in self.studies] for s in self.arm_study])
        D = np.array([[1.0 if (design[s] == d and t == tt) else 0.0 for d, tt in cols]
                      for s, t in zip(self.arm_study, self.arm_trt)])
        return np.hstack([S, D])


# --------------------------------------------------------------------------- #
# Weighted least squares and Q decomposition
# --------------------------------------------------------------------------- #
def wls_fit(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    sw = np.sqrt(w)
    beta = np.linalg.lstsq(sw[:, None] * X, sw * y, rcond=None)[0]
    return X @ beta


def q_decomposition(net: Network, y: np.ndarray | None = None, w: np.ndarray | None = None) -> dict:
    """Q_net = Q_het + Q_inc for any fixed positive weight vector w (exact algebra)."""
    y = net.phat if y is None else y
    w = net.w if w is None else w
    if not hasattr(net, "_Xc"):
        net._Xc, net._Xd = net.design_consistency(), net.design_specific()
        net._rc, net._rd = np.linalg.matrix_rank(net._Xc), np.linalg.matrix_rank(net._Xd)
    Xc, Xd, rc, rd = net._Xc, net._Xd, net._rc, net._rd
    yc, yd = wls_fit(Xc, y, w), wls_fit(Xd, y, w)
    m = len(y)
    return dict(Q_net=float(np.sum(w * (y - yc) ** 2)),
                Q_het=float(np.sum(w * (y - yd) ** 2)),
                Q_inc=float(np.sum(w * (yd - yc) ** 2)),
                df_net=m - rc, df_het=m - rd, df_inc=rd - rc, fit_c=yc, fit_d=yd)


def pearson_weights(net: Network, fit: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    p = np.clip(fit, floor, 1 - floor)
    return net.n / (p * (1 - p))


def null_fit_probabilities(net: Network, floor="half", y: np.ndarray | None = None) -> np.ndarray:
    """Fitted arm risks under the consistency (null) model, used only to generate
    the binomial reference distribution.  The additive model can return fitted
    risks below zero for zero-event arms; floor="half" replaces any fitted risk
    below 0.5/n_ku by 0.5/n_ku (a half event), floor=0 clips at zero, or pass a
    number.  The choice affects only the calibration, never the estimates."""
    q = q_decomposition(net, y)
    f = q["fit_c"]
    if isinstance(floor, str) and floor == "half":
        lo = 0.5 / net.n
    else:
        lo = np.full_like(f, float(floor))
    return np.minimum(np.maximum(f, lo), 1.0)


def binomial_calibration(net: Network, B: int = 4000, seed: int = 1, pearson: bool = False,
                         floor="half", p_null: np.ndarray | None = None) -> dict:
    """Reference distribution of (Q_net, Q_inc, Q_het) by simulating independent
    binomial arms from the fitted null model.  No normal approximation.
    Default: the design-fixed weights w (same weights as the estimator).  With
    pearson=True the statistics use n/(p(1-p)) at the null fit; with rare events
    these weights are dominated by arms whose fitted risk is near zero and the
    statistic becomes extremely heavy-tailed (see the supplement), so the
    design-fixed version is the one reported.
    Returns observed statistics, Monte Carlo p-values and the simulated draws."""
    rng = np.random.default_rng(seed)
    p0 = null_fit_probabilities(net, floor) if p_null is None else p_null
    nint = net.n.astype(int)

    def stats(y):
        if pearson:
            q0 = q_decomposition(net, y)
            w = pearson_weights(net, q0["fit_c"])
            q = q_decomposition(net, y, w)
        else:
            q = q_decomposition(net, y)
        return q["Q_net"], q["Q_inc"], q["Q_het"]

    obs = stats(net.phat)
    draws = np.empty((B, 3))
    for b in range(B):
        yb = rng.binomial(nint, p0) / net.n
        draws[b] = stats(yb)
    pvals = tuple(float(np.mean(draws[:, j] >= obs[j])) for j in range(3))
    q = q_decomposition(net)
    return dict(Q_net=obs[0], Q_inc=obs[1], Q_het=obs[2],
                p_net=pvals[0], p_inc=pvals[1], p_het=pvals[2],
                df_net=q["df_net"], df_inc=q["df_inc"], df_het=q["df_het"],
                draws=draws, p_null=p0)


def berger_boos_envelope(net: Network, B: int = 1500, M: int = 15, gamma: float = 0.001,
                         seed: int = 2, pearson: bool = False, floor="half") -> dict:
    """Approximate Berger-Boos adjustment: maximise the Monte Carlo p-value over
    M alternative null parameter vectors drawn from the parametric bootstrap of
    the null fit, then add gamma.  This is an approximation to the supremum over
    a (1-gamma) confidence set; it is reported as a sensitivity envelope."""
    rng = np.random.default_rng(seed)
    base = binomial_calibration(net, B=B, seed=seed, pearson=pearson, floor=floor)
    p0 = base["p_null"]
    nint = net.n.astype(int)
    pmax = np.array([base["p_net"], base["p_inc"], base["p_het"]])
    for m in range(M):
        yb = rng.binomial(nint, p0) / net.n
        alt = null_fit_probabilities(net, floor, y=yb)
        cal = binomial_calibration(net, B=B, seed=seed + 100 + m, pearson=pearson, floor=floor, p_null=alt)
        pmax = np.maximum(pmax, [cal["p_net"], cal["p_inc"], cal["p_het"]])
    return dict(p_net=min(1.0, pmax[0] + gamma), p_inc=min(1.0, pmax[1] + gamma),
                p_het=min(1.0, pmax[2] + gamma), plugin=base)


def null_expectation_Q(net: Network, p0: np.ndarray) -> dict:
    """Exact null expectations of Q_net, Q_het, Q_inc with design-fixed weights:
    E[Q] = sum_i w_i (1-h_i) p_i(1-p_i)/n_i, with h the leverages of the relevant
    weighted projection (Q_inc uses the difference of the two hat matrices)."""
    w = net.w
    def hat(X):
        sw = np.sqrt(w)
        Xw = sw[:, None] * X
        return Xw @ np.linalg.pinv(Xw.T @ Xw) @ Xw.T
    Hc, Hd = hat(net.design_consistency()), hat(net.design_specific())
    v = p0 * (1 - p0) / net.n
    I = np.eye(len(w))
    def ex(M):   # E[(y-My)' W (y-My)] with y ~ (p0, diag v): trace(W (I-M) diag(v) (I-M)') since mean fitted exactly
        A = (I - M)
        return float(np.sum(w * np.sum(A * A * v[None, :], axis=1)))
    # expectation of Q_net = y'(I-Hc)'W(I-Hc)y on residual scale; here M acts on sqrt(w) scale, so convert
    sw = np.sqrt(w)
    def exQ(H):
        A = (np.eye(len(w)) - H)          # on the sqrt(w)-scaled scale
        # Q = || A sw*y ||^2 ; E = trace(A diag(w v) A') + || A sw*p0 ||^2
        # (the second term vanishes when p0 lies exactly in the model space;
        # it is kept because the floored null risks need not)
        m = A @ (sw * p0)
        return float(np.trace(A @ np.diag(w * v) @ A.T) + m @ m)
    return dict(E_Q_net=exQ(Hc), E_Q_het=exQ(Hd), E_Q_inc=exQ(Hc) - exQ(Hd))


# --------------------------------------------------------------------------- #
# Interval for the estimate (parametric bootstrap from binomial arms)
# --------------------------------------------------------------------------- #
def binomial_interval(net: Network, a: str, b: str, B: int = 20000, seed: int = 3,
                      level: float = 0.95, p: np.ndarray | None = None) -> dict:
    """Interval for theta_ab from the exact binomial sampling distribution of the
    linear estimator, with arm risks plugged in (default: observed proportions).
    Returns percentile interval and the exact-variance normal interval for comparison."""
    rng = np.random.default_rng(seed)
    p = net.phat if p is None else p
    c = net.coefficient_vector(a, b)
    nint = net.n.astype(int)
    est = float(c @ net.phat)
    sims = (rng.binomial(nint, p, size=(B, len(c))) / net.n) @ c
    lo, hi = np.quantile(sims, [(1 - level) / 2, 1 - (1 - level) / 2])
    # basic (reflected) interval, which corrects skewness in the other direction
    blo, bhi = 2 * est - hi, 2 * est - lo
    se = np.sqrt(net.exact_variance(a, b, p))
    z = 1.959963984540054
    return dict(estimate=est, percentile=(float(lo), float(hi)), basic=(float(blo), float(bhi)),
                exact_se=float(se), normal=(est - z * se, est + z * se))


# --------------------------------------------------------------------------- #
# Exact test for a single study's own comparison (reported-significance proxy)
# --------------------------------------------------------------------------- #
def fisher_exact_p(r1: int, n1: int, r2: int, n2: int) -> float:
    """Two-sided Fisher exact p-value (sum of probabilities <= observed)."""
    m = r1 + r2
    N = n1 + n2
    lo, hi = max(0, m - n2), min(m, n1)

    def lp(x):
        return (lgamma(n1 + 1) - lgamma(x + 1) - lgamma(n1 - x + 1)
                + lgamma(n2 + 1) - lgamma(m - x + 1) - lgamma(n2 - m + x + 1)
                - (lgamma(N + 1) - lgamma(m + 1) - lgamma(N - m + 1)))

    probs = {x: exp(lp(x)) for x in range(lo, hi + 1)}
    pobs = probs[r1]
    return float(min(1.0, sum(v for v in probs.values() if v <= pobs * (1 + 1e-9))))


def study_significance(net: Network) -> pd.DataFrame:
    """For every pair of arms within each study: Fisher exact two-sided p-value."""
    out = []
    for k in net.studies:
        idx = net.rows[k]
        for i, j in combinations(range(len(idx)), 2):
            a, b = net.arm_trt[idx[i]], net.arm_trt[idx[j]]
            if a > b:
                i, j, a, b = j, i, b, a
            p = fisher_exact_p(int(net.r[idx[j]]), int(net.n[idx[j]]), int(net.r[idx[i]]), int(net.n[idx[i]]))
            out.append(dict(study=k, a=a, b=b, r_a=int(net.r[idx[i]]), n_a=int(net.n[idx[i]]),
                            r_b=int(net.r[idx[j]]), n_b=int(net.n[idx[j]]),
                            rd=net.phat[idx[j]] - net.phat[idx[i]], fisher_p=p))
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# Comparative information and design-stage contribution
# --------------------------------------------------------------------------- #
def throughput_table(net: Network, targets=None) -> pd.DataFrame:
    """Study throughput W_{t,k} for every target (default: all pairs)."""
    targets = list(combinations(net.trts, 2)) if targets is None else targets
    rows = {k: {} for k in net.studies}
    for a, b in targets:
        cur = net.currents(a, b)
        for k in net.studies:
            rows[k][f"{a}:{b}"] = 0.5 * np.abs(cur[k]).sum()
    tab = pd.DataFrame(rows).T
    tab["max_throughput"] = tab.max(axis=1)
    tab["total_n"] = [net.n[net.rows[k]].sum() for k in net.studies]
    return tab


def add_planned_trial(net: Network, arms: dict, name: str = "PLANNED") -> Network:
    """Return a new network with a planned trial added (arms: treatment -> n).
    Events are set to zero placeholders; they do not affect the coefficients."""
    rows = [dict(study=name, treatment=t, events=0, n=n) for t, n in arms.items()]
    df = pd.concat([net.df, pd.DataFrame(rows)], ignore_index=True)
    return Network(df)


# --------------------------------------------------------------------------- #
# Comparator: inverse-variance risk-difference NMA with continuity correction
# --------------------------------------------------------------------------- #
def iv_rd_estimate(net: Network, a: str, b: str, cc: float = 0.5) -> tuple[float, float]:
    """Contrast-level inverse-variance NMA of risk differences with 0.5 added to
    every cell of a study that has a zero cell (the usual practice), and
    independent-arm variances p(1-p)/n.  Multi-arm studies enter with their
    arm-based covariance.  Normal theory throughout.  For comparison only."""
    y, Vblocks, Xrows = [], [], []
    for k in net.studies:
        idx = net.rows[k]
        r, n = net.r[idx].copy(), net.n[idx].copy()
        if np.any(r == 0) or np.any(r == n):
            r = r + cc
            n = n + 2 * cc
        p = r / n
        v = p * (1 - p) / n
        ref = 0
        for j in range(1, len(idx)):
            y.append(p[j] - p[ref])
            row = np.zeros(net.T)
            row[net.tix[net.arm_trt[idx[j]]]] += 1
            row[net.tix[net.arm_trt[idx[ref]]]] -= 1
            Xrows.append(row)
        m = len(idx) - 1
        Vk = np.full((m, m), v[ref]) + np.diag(v[1:])
        Vblocks.append(Vk)
    y = np.array(y)
    X = np.array(Xrows)[:, 1:]            # drop reference treatment column
    V = np.zeros((len(y), len(y)))
    pos = 0
    for Vk in Vblocks:
        m = Vk.shape[0]
        V[pos:pos + m, pos:pos + m] = Vk
        pos += m
    Vi = np.linalg.inv(V)
    M = np.linalg.inv(X.T @ Vi @ X)
    beta = M @ X.T @ Vi @ y
    c = net.e(b, a)[1:]
    return float(c @ beta), float(np.sqrt(c @ M @ c))


# --------------------------------------------------------------------------- #
# Verification helpers
# --------------------------------------------------------------------------- #
def verify_identities(net: Network, a: str, b: str) -> dict:
    """Numerical checks of the algebraic identities used in the paper."""
    res = net.contributions(a, b)
    q = q_decomposition(net)
    Xc = net.design_consistency()
    yc = wls_fit(Xc, net.phat, net.w)
    # WLS target from fitted values: fitted contrast in any study containing both arms,
    # else from the fitted treatment effects
    beta = np.linalg.lstsq(np.sqrt(net.w)[:, None] * Xc, np.sqrt(net.w) * net.phat, rcond=None)[0]
    mu = np.concatenate([[0.0], beta[len(net.studies):]])
    wls_theta = mu[net.tix[b]] - mu[net.tix[a]]
    cur = net.currents(a, b)
    conserve = max(abs(net.K[k] @ np.ones(len(net.rows[k]))).max() if len(net.rows[k]) > 1 else 0.0
                   for k in net.studies)
    return dict(contributions_sum_minus_estimate=res["C"].sum() - net.estimate(a, b),
                estimate_minus_wls=net.estimate(a, b) - wls_theta,
                direct_plus_indirect_minus_total=float(np.abs(res["C_D"] + res["C_I"] - res["C"]).max()),
                weights_sum_direct_plus_q=float(res["d"].sum() + res.attrs["q"]) - 1.0,
                Q_additivity=q["Q_net"] - q["Q_het"] - q["Q_inc"],
                study_current_conservation=float(max(abs(cur[k].sum()) for k in net.studies)),
                laplacian_rowsum=float(conserve))


# =========================================================================== #
# Exact and nuisance-robust testing of consistency (Q_inc) -- added Oct 25
# =========================================================================== #
from scipy.stats import binom as _binom, chi2 as _chi2, beta as _beta


def inconsistency_projector(net: Network, w: np.ndarray | None = None):
    """Return (P, B, sw): P = H_D - H_C is the orthogonal projector (on the sqrt(w)-scaled
    proportions) whose squared norm is Q_inc; B is an orthonormal basis of its column
    space (df_inc columns); sw = sqrt(w)."""
    w = net.w if w is None else w
    sw = np.sqrt(w)
    def H(X):
        Xw = sw[:, None] * X
        return Xw @ np.linalg.pinv(Xw.T @ Xw) @ Xw.T
    Hc, Hd = H(net.design_consistency()), H(net.design_specific())
    P = Hd - Hc
    P = (P + P.T) / 2
    vals, vecs = np.linalg.eigh(P)
    B = vecs[:, vals > 0.5]
    return P, B, sw


def qinc_from_P(P, sw, Y):
    """Q_inc for proportions Y (vector or (B,m) matrix) given the projector P."""
    Z = np.atleast_2d(Y) * sw[None, :]
    R = Z @ P.T
    return (R ** 2).sum(1)


def qinc_exact_moments(net: Network, p: np.ndarray, P=None, sw=None):
    """Exact mean and variance of Q_inc = z'Pz, z_i = sqrt(w_i) phat_i, for independent
    binomial arms with risks p.  Writing z = mu + eps with mu = sqrt(w) p and m = P mu
    (m = 0 when p satisfies the consistency model exactly),
        E Q   = tr(P Sigma) + mu'P mu,
        Var Q = sum_i P_ii^2 (mu4_i - 3 s_i^4) + 2 tr(P Sigma P Sigma) + 4 m'Sigma m
                + 4 sum_i P_ii m_i mu3_i,
    with Sigma = diag(s_i^2), s_i^2 = w p q / n, and the binomial central moments
    mu3 = npq(q-p), mu4 = npq(1 + 3(n-2)pq) scaled to eps_i."""
    if P is None:
        P, _, sw = inconsistency_projector(net)
    w, n = net.w, net.n
    q = 1 - p
    s2 = w * p * q / n
    mu3 = w ** 1.5 * (n * p * q * (q - p)) / n ** 3
    mu4 = w ** 2 * (n * p * q * (1 + 3 * (n - 2) * p * q)) / n ** 4
    d = np.diag(P)
    mu = sw * p
    m = P @ mu
    mean = float(np.sum(d * s2) + mu @ m)
    var = float(np.sum(d ** 2 * (mu4 - 3 * s2 ** 2)) + 2 * np.sum((P ** 2) * np.outer(s2, s2))
                + 4 * np.sum(m ** 2 * s2) + 4 * np.sum(d * m * mu3))
    return mean, var


def satterthwaite_p(q_obs, mean, var):
    """Two-moment chi-square approximation Q ~ c * chi2(nu)."""
    c = var / (2 * mean)
    nu = 2 * mean ** 2 / var
    return float(_chi2.sf(q_obs / c, nu)), c, nu


def clopper_pearson_box(net: Network, level: float):
    """Per-arm Clopper-Pearson intervals at confidence `level` (joint by Bonferroni outside)."""
    r, n = net.r, net.n
    a = (1 - level) / 2
    lo = np.where(r > 0, _beta.ppf(a, r, n - r + 1), 0.0)
    hi = np.where(r < n, _beta.ppf(1 - a, r + 1, n - r), 1.0)
    return lo, hi


def _lp_null_point(Xc, lo, hi):
    """A point of the null model space inside the box [lo, hi] (a linear-programming feasibility problem,
    maximizing the smallest slack), or None if the intersection C_gamma is empty."""
    from scipy.optimize import linprog
    m, k = Xc.shape
    A = np.vstack([np.hstack([-Xc, np.ones((m, 1))]), np.hstack([Xc, np.ones((m, 1))])])
    b = np.r_[-lo, hi]
    c = np.zeros(k + 1)
    c[-1] = -1.0
    res = linprog(c, A_ub=A, b_ub=b, bounds=[(None, None)] * k + [(0, None)], method="highs")
    if res.status != 0:
        return None
    return Xc @ res.x[:k]


def berger_boos_sup_p(net: Network, q_obs=None, gamma: float = 0.001, B: int = 2000, seed: int = 11,
                      n_starts: int = 6, n_iter: int = 60, step: float = 0.5, verbose=False):
    """Exact unconditional test of consistency: p = sup_{p in C_gamma} P_p(Q_inc >= q_obs) + gamma,
    where C_gamma is the null-model space intersected with the Bonferroni product of per-arm
    Clopper-Pearson intervals at joint level 1-gamma (Berger & Boos 1994).  Candidate null
    parameters are kept in C_gamma by alternating projections between the model space and the
    box; the supremum is approximated by a stochastic search with common random numbers and
    is reported as the best value found.  If C_gamma is empty (no null parameter is compatible
    with the per-arm intervals, decided by a linear-programming feasibility check when the alternating
    projections do not converge) the function returns p = gamma and flags it."""
    rng = np.random.default_rng(seed)
    P, Bmat, sw = inconsistency_projector(net)
    if q_obs is None:
        q_obs = float(qinc_from_P(P, sw, net.phat)[0])
    Xc = net.design_consistency()
    Xw = sw[:, None] * Xc
    pinv = np.linalg.pinv(Xw.T @ Xw) @ Xw.T
    nint = net.n.astype(int)
    m = len(net.n)
    lo, hi = clopper_pearson_box(net, 1 - gamma / m)
    U = rng.uniform(size=(B, m))                       # common random numbers

    def project(p, iters=300):
        """alternating projections: model space (weighted fit) <-> box"""
        for _ in range(iters):
            beta = pinv @ (sw * p)
            pm = Xc @ beta
            pb = np.minimum(np.maximum(pm, lo), hi)
            if np.abs(pb - pm).max() < 1e-12:
                return pm, beta, True
            p = pb
        beta = pinv @ (sw * p)
        pm = Xc @ beta
        return pm, beta, bool(np.abs(np.minimum(np.maximum(pm, lo), hi) - pm).max() < 1e-7)

    def tail(p):
        p = np.clip(p, 1e-12, 1 - 1e-12)
        R = _binom.ppf(U, nint[None, :], p[None, :])
        return float(np.mean(qinc_from_P(P, sw, R / net.n) >= q_obs))

    p_fit = Xc @ (pinv @ (sw * net.phat))
    p0, beta0, feasible = project(p_fit)
    if not feasible:                                   # decide emptiness exactly, not by non-convergence
        p_lp = _lp_null_point(Xc, lo, hi)
        if p_lp is not None:
            p0, beta0, feasible = p_lp, None, True
    if not feasible:
        return dict(q_obs=q_obs, sup_tail=float("nan"), p_value=gamma, gamma=gamma,
                    p_at_fitted_null=float("nan"), feasible=False)
    best_p, best = p0.copy(), tail(p0)
    tail_fit = best
    scale = np.std(net.phat) + 1e-3
    for s in range(n_starts):
        p = best_p.copy()
        if s > 0:
            p, _, ok = project(best_p + rng.normal(scale=scale * 0.5, size=m))
            if not ok:
                p = best_p.copy()
        cur = tail(p)
        st = step * scale
        for it in range(n_iter):
            cand, _, ok = project(p + rng.normal(scale=st, size=m), iters=100)
            if ok:
                val = tail(cand)
                if val > cur:
                    p, cur = cand, val
                    continue
            st *= 0.9
        if cur > best:
            best, best_p = cur, p.copy()
        if verbose:
            print("start", s, "tail", cur, "best", best)
    return dict(q_obs=q_obs, sup_tail=best, p_value=min(1.0, best + gamma), gamma=gamma,
                p_at_fitted_null=tail_fit, feasible=True, best_p=best_p)


def hoeffding_inconsistency_test(net: Network):
    """Distribution-free exact-level test of consistency: Q_inc = sum_j zeta_j^2 with
    zeta_j = b_j' z independent-binomial linear statistics; Hoeffding bound for each
    coordinate and a union (Bonferroni) bound over the df_inc coordinates.  Valid for
    every nuisance value; conservative."""
    P, Bmat, sw = inconsistency_projector(net)
    z = sw * net.phat
    zeta = Bmat.T @ z
    ranges2 = (Bmat ** 2 * (net.w / net.n)[:, None]).sum(0)     # sum_i b_ji^2 w_i / n_i
    pj = 2 * np.exp(-2 * zeta ** 2 / ranges2)
    return dict(zeta=zeta, p_components=pj, p_value=float(min(1.0, pj.sum())), df=Bmat.shape[1])


def fiducial_predictive_p(net: Network, B: int = 4000, seed: int = 13):
    """Generalized (fiducial / posterior-predictive) p-value: draw per-arm risks from
    Beta(r+1/2, n-r+1/2), project them onto the null model space by the weighted fit,
    simulate binomial arms, recompute Q_inc.  Not exact; reported for comparison."""
    rng = np.random.default_rng(seed)
    P, Bmat, sw = inconsistency_projector(net)
    q_obs = float(qinc_from_P(P, sw, net.phat)[0])
    Xc = net.design_consistency()
    Xw = sw[:, None] * Xc
    Hc = Xw @ np.linalg.pinv(Xw.T @ Xw) @ Xw.T
    nint = net.n.astype(int)
    pstar = rng.beta(net.r + 0.5, net.n - net.r + 0.5, size=(B, len(net.n)))
    p0 = np.clip((pstar * sw[None, :]) @ Hc.T / sw[None, :], 0.0, 1.0)
    Y = rng.binomial(nint[None, :], p0) / net.n[None, :]
    return dict(q_obs=q_obs, p_value=float(np.mean(qinc_from_P(P, sw, Y) >= q_obs)))


def wald_inconsistency_stat(net: Network, Y, P=None, Bmat=None, sw=None, pinv=None, Xc=None):
    """Standardized (Wald-type) inconsistency statistic T = zeta' Sigma(p~)^{-1} zeta, with
    zeta = B' (sqrt(w) y) the df_inc inconsistency contrasts and Sigma(p~) their covariance
    evaluated at the null fit p~ of the same data (floored at half an event).  Y may be a
    vector or a (B, m) matrix; returns an array of statistics.  Vectorized over draws."""
    if P is None:
        P, Bmat, sw = inconsistency_projector(net)
    if Xc is None:
        Xc = net.design_consistency()
        Xw = sw[:, None] * Xc
        pinv = np.linalg.pinv(Xw.T @ Xw) @ Xw.T
    Y = np.atleast_2d(Y)
    Z = Y * sw[None, :]
    zeta = Z @ Bmat                                   # (B, df)
    beta = Z @ pinv.T                                 # (B, k)
    Pt = np.minimum(np.maximum(beta @ Xc.T, 0.5 / net.n[None, :]), 1.0)
    V = net.w[None, :] * Pt * (1 - Pt) / net.n[None, :]   # (B, m)
    S = np.einsum("bi,ij,ik->bjk", V, Bmat, Bmat)      # (B, df, df)
    sol = np.linalg.solve(S, zeta[:, :, None])[:, :, 0]
    return np.einsum("bj,bj->b", zeta, sol)


def berger_boos_sup_wald(net: Network, gamma: float = 0.001, B: int = 2000, seed: int = 11,
                         n_starts: int = 4, n_iter: int = 40, step: float = 0.5):
    """Berger-Boos supremum p-value for the standardized statistic T (see wald_inconsistency_stat).
    Same search as berger_boos_sup_p; the standardization makes the null tail far less
    sensitive to the nuisance risks, so the supremum is much closer to the plug-in value."""
    rng = np.random.default_rng(seed)
    P, Bmat, sw = inconsistency_projector(net)
    Xc = net.design_consistency()
    Xw = sw[:, None] * Xc
    pinv = np.linalg.pinv(Xw.T @ Xw) @ Xw.T
    t_obs = float(wald_inconsistency_stat(net, net.phat, P, Bmat, sw, pinv, Xc)[0])
    nint = net.n.astype(int)
    m = len(net.n)
    lo, hi = clopper_pearson_box(net, 1 - gamma / m)
    U = rng.uniform(size=(B, m))

    def project(p, iters=300):
        for _ in range(iters):
            pm = Xc @ (pinv @ (sw * p))
            pb = np.minimum(np.maximum(pm, lo), hi)
            if np.abs(pb - pm).max() < 1e-12:
                return pm, True
            p = pb
        pm = Xc @ (pinv @ (sw * p))
        return pm, bool(np.abs(np.minimum(np.maximum(pm, lo), hi) - pm).max() < 1e-7)

    def tail(p):
        p = np.clip(p, 1e-12, 1 - 1e-12)
        R = _binom.ppf(U, nint[None, :], p[None, :])
        return float(np.mean(wald_inconsistency_stat(net, R / net.n, P, Bmat, sw, pinv, Xc) >= t_obs))

    p_fit = Xc @ (pinv @ (sw * net.phat))
    p0, feasible = project(p_fit)
    if not feasible:                                   # decide emptiness exactly, not by non-convergence
        p_lp = _lp_null_point(Xc, lo, hi)
        if p_lp is not None:
            p0, feasible = p_lp, True
    if not feasible:
        return dict(t_obs=t_obs, sup_tail=float("nan"), p_value=gamma, feasible=False)
    best_p, best = p0.copy(), tail(p0)
    tail_fit = best
    scale = np.std(net.phat) + 1e-3
    for s in range(n_starts):
        p = best_p.copy()
        if s > 0:
            p, ok = project(best_p + rng.normal(scale=scale * 0.5, size=m))
            if not ok:
                p = best_p.copy()
        cur = tail(p)
        st = step * scale
        for it in range(n_iter):
            cand, ok = project(p + rng.normal(scale=st, size=m), iters=100)
            if ok:
                val = tail(cand)
                if val > cur:
                    p, cur = cand, val
                    continue
            st *= 0.9
        if cur > best:
            best, best_p = cur, p.copy()
    chi2_p = float(_chi2.sf(t_obs, Bmat.shape[1]))
    return dict(t_obs=t_obs, df=int(Bmat.shape[1]), sup_tail=best, p_value=min(1.0, best + gamma),
                p_at_fitted_null=tail_fit, chi2_p=chi2_p, feasible=True, best_p=best_p)


# =========================================================================== #
# Constrained binomial maximum-likelihood null fit (coherent plug-in generator)
# =========================================================================== #
from scipy.optimize import minimize as _minimize


def constrained_null_mle(net: Network, design: str = "consistency", y: np.ndarray | None = None):
    """Maximum-likelihood fit of the additive risk model under the stated null
    (consistency or design-specific), with 0 <= X beta <= 1 enforced, so that the
    fitted risks lie in the null mean space and in [0,1].  Used only to generate the
    plug-in binomial reference distribution; estimation never uses it."""
    X = net.design_consistency() if design == "consistency" else net.design_specific()
    r = net.r if y is None else y * net.n
    n = net.n
    Xw = np.sqrt(net.w)[:, None] * X
    beta0 = np.linalg.lstsq(Xw, np.sqrt(net.w) * (r / n), rcond=None)[0]
    eps = 1e-9

    def nll(b):
        p = np.clip(X @ b, eps, 1 - eps)
        return -np.sum(r * np.log(p) + (n - r) * np.log1p(-p))

    def grad(b):
        p = np.clip(X @ b, eps, 1 - eps)
        return -X.T @ (r / p - (n - r) / (1 - p))

    cons = [{"type": "ineq", "fun": lambda b: X @ b - eps, "jac": lambda b: X},
            {"type": "ineq", "fun": lambda b: 1 - eps - X @ b, "jac": lambda b: -X}]
    p_start = np.clip(X @ beta0, 0.5 / n, 1 - 0.5 / n)
    b_start = np.linalg.lstsq(X, p_start, rcond=None)[0]
    res = _minimize(nll, b_start, jac=grad, constraints=cons, method="SLSQP", options={"maxiter": 500, "ftol": 1e-12})
    res.fallback = False
    # SLSQP status 0 (converged) and 8 (line search stopped on an active bound, the usual termination when a
    # zero-event arm is fitted at risk zero) are accepted; any other status or an infeasible point is
    # replaced by the interior-point solution of the same concave problem when that has a higher likelihood.
    infeasible = (X @ res.x < -1e-8).any() or (X @ res.x > 1 + 1e-8).any()
    if res.status not in (0, 8) or infeasible:
        pb = _barrier_mle(X, r, n)
        if pb is not None and (infeasible or _nll_p(pb, r, n) < _nll_p(np.clip(X @ res.x, 0, 1), r, n)):
            res.fallback = True
            return np.clip(pb, 0.0, 1.0), res
    p = np.clip(X @ res.x, 0.0, 1.0)
    return p, res


def _nll_p(p, r, n):
    p = np.clip(p, 1e-300, 1 - 1e-16)
    return float(-np.sum(r * np.log(p) + (n - r) * np.log1p(-p)))


def _barrier_mle(X, r, n, mu_min=1e-12):
    """Interior-point (logarithmic barrier, damped Newton) maximizer of the binomial likelihood over
    0 <= X beta <= 1; returns the fitted risks, or None if no strictly feasible start is found."""
    phat = r / n
    pt = np.clip(phat, 0.002, 0.998)
    b = np.linalg.lstsq(X, pt, rcond=None)[0]
    for _ in range(50):
        if ((X @ b > 0) & (X @ b < 1)).all():
            break
        pt = 0.5 * pt + 0.5 * phat.mean()
        b = np.linalg.lstsq(X, pt, rcond=None)[0]
    else:
        return None

    def fbar(b, mu):
        p = X @ b
        if (p <= 0).any() or (p >= 1).any():
            return np.inf
        return _nll_p(p, r, n) - mu * np.sum(np.log(p) + np.log1p(-p))
    mu = 1.0
    while mu > mu_min:
        for _ in range(100):
            p = X @ b
            g = -X.T @ (r / p - (n - r) / (1 - p)) - mu * X.T @ (1 / p - 1 / (1 - p))
            h = r / p ** 2 + (n - r) / (1 - p) ** 2 + mu * (1 / p ** 2 + 1 / (1 - p) ** 2)
            step = np.linalg.lstsq(X.T @ (h[:, None] * X), -g, rcond=None)[0]
            t, f0 = 1.0, fbar(b, mu)
            while fbar(b + t * step, mu) > f0 + 1e-4 * t * (g @ step) and t > 1e-12:
                t *= 0.5
            b = b + t * step
            if abs(g @ step) < 1e-12:
                break
        mu *= 0.3
    return X @ b


def binomial_calibration_mle(net: Network, B: int = 4000, seed: int = 1) -> dict:
    """As binomial_calibration (design-fixed weights) but the null generator is the
    constrained binomial MLE under the consistency model, so the simulated mean lies
    exactly in the null mean space (no floor, no noncentral term)."""
    rng = np.random.default_rng(seed)
    p0, res = constrained_null_mle(net)
    nint = net.n.astype(int)
    obs = q_decomposition(net)
    draws = np.empty((B, 3))
    for b in range(B):
        yb = rng.binomial(nint, p0) / net.n
        q = q_decomposition(net, yb)
        draws[b] = (q["Q_net"], q["Q_inc"], q["Q_het"])
    o = np.array([obs["Q_net"], obs["Q_inc"], obs["Q_het"]])
    pv = (draws >= o[None, :]).mean(0)
    return dict(Q_net=o[0], Q_inc=o[1], Q_het=o[2], p_net=float(pv[0]), p_inc=float(pv[1]), p_het=float(pv[2]),
                draws=draws, p_null=p0, converged=bool(res.success), zero_arms_fitted=int((p0 <= 1e-9).sum()))


# =========================================================================== #
# Routes: within-study pairing, pooled study-labelled flow, bottleneck paths,
# identification check (added for the ACS revision)
# =========================================================================== #
from collections import defaultdict as _dd


def study_pairings(net: Network, a: str, b: str):
    """CSP canonical reduction: direct-first extraction, then lexicographic max-min
    pairing of residual supplies and demands within each study.  Returns
    (direct dict k->d, edges list of (u, v, k, flow))."""
    cur = net.currents(a, b)
    direct, edges = {}, []
    for k in net.studies:
        idx = net.rows[k]
        arms = list(net.arm_trt[idx])
        bk = cur[k].copy()
        s = {u: max(x, 0.0) for u, x in zip(arms, bk)}
        t = {u: max(-x, 0.0) for u, x in zip(arms, bk)}
        d = 0.0
        if a in arms and b in arms:
            d = min(s[a], t[b])
            s[a] -= d
            t[b] -= d
            if d > 1e-15:
                edges.append((a, b, k, d))
        direct[k] = d
        while True:
            S = [(u, x) for u, x in s.items() if x > 1e-15]
            D = [(v, x) for v, x in t.items() if x > 1e-15]
            if not S or not D:
                break
            best = max(((min(x, y), u, v) for u, x in S for v, y in D), key=lambda z: (z[0], -net.tix[z[1]], -net.tix[z[2]]))
            amt, u, v = best
            edges.append((u, v, k, amt))
            s[u] -= amt
            t[v] -= amt
    return direct, edges


def bottleneck_paths(edges, a, b, tol=1e-12):
    """Decompose a pooled acyclic study-labelled flow into a->b paths by repeatedly
    extracting the widest path (largest bottleneck).  Returns list of
    (weight, [(u, v, k), ...])."""
    flow = {(u, v, k): f for u, v, k, f in edges}
    paths = []
    while True:
        # widest path by DP over DAG (potentials guarantee acyclicity); use Dijkstra-like max-min
        best = {a: (float("inf"), None)}
        frontier = [a]
        visited = set()
        import heapq
        heap = [(-float("inf"), a)]
        while heap:
            negw, u = heapq.heappop(heap)
            if u in visited:
                continue
            visited.add(u)
            for (x, y, k), f in flow.items():
                if x == u and f > tol:
                    w = min(-negw, f)
                    if y not in best or w > best[y][0]:
                        best[y] = (w, (x, y, k))
                        heapq.heappush(heap, (-w, y))
        if b not in best or best[b][0] <= tol:
            break
        w = best[b][0]
        path, node = [], b
        while node != a:
            e = best[node][1]
            path.append(e)
            node = e[0]
        path = path[::-1]
        for e in path:
            flow[e] -= w
        paths.append((w, path))
    residual = sum(f for f in flow.values() if f > tol)
    return paths, residual


def route_identification(edges, a, b, tol=1e-12):
    """Proposition 2(ii)-(iii) of the paper: the study-labelled route allocation
    is unique iff every support vertex has min(N^-(v), N^+(v)) = 1, where N^- counts
    distinct source-to-v prefixes and N^+ distinct v-to-sink suffixes (capped at 2).
    Also returns the sharp interval [L, U] for each canonical route via the
    closed-form bounds U = min edge flow, L = [sum edge flows - sum internal vertex throughput]_+."""
    E = [(u, v, k, f) for u, v, k, f in edges if f > tol]
    verts = set([a, b]) | {u for u, _, _, _ in E} | {v for _, v, _, _ in E}
    out = _dd(list); inn = _dd(list)
    for u, v, k, f in E:
        out[u].append((v, k, f)); inn[v].append((u, k, f))
    # topological order by potential is implicit; use memoised counts capped at 2
    from functools import lru_cache
    @lru_cache(None)
    def Nminus(v):
        if v == a: return 1
        return min(2, sum(Nminus(u) for u, _, _ in inn[v]))
    @lru_cache(None)
    def Nplus(v):
        if v == b: return 1
        return min(2, sum(Nplus(w) for w, _, _ in out[v]))
    support = [v for v in verts if Nminus(v) > 0 and Nplus(v) > 0]
    unique = all(min(Nminus(v), Nplus(v)) == 1 for v in support)
    through = {v: sum(f for _, _, f in inn[v]) for v in verts}
    return dict(unique=unique, support=support, counts={v: (Nminus(v), Nplus(v)) for v in support}, throughput=through)


def route_table(net: Network, a: str, b: str):
    direct, edges = study_pairings(net, a, b)
    indirect_edges = [e for e in edges if not (e[0] == a and e[1] == b and len(e) == 4 and direct.get(e[2], 0) > 0 and e[3] == direct[e[2]])]
    paths, residual = bottleneck_paths(indirect_edges, a, b)
    ident = route_identification(indirect_edges, a, b)
    flow = {(u, v, k): f for u, v, k, f in indirect_edges}
    rows = []
    for w, path in paths:
        U = min(flow[e] for e in path)
        internal = [e[1] for e in path[:-1]]
        L = max(0.0, sum(flow[e] for e in path) - sum(ident["throughput"][v] for v in internal))
        rows.append(dict(weight=w, studies="/".join(str(e[2]) for e in path), path="\u2192".join([path[0][0]] + [e[1] for e in path]), lower=L, upper=U))
    return direct, rows, residual, ident


# --------------------------------------------------------------------------- #
# MOVER-Wilson interval for a fixed linear combination of independent proportions
# --------------------------------------------------------------------------- #
def wilson_interval(r, n, level: float = 0.95):
    """Wilson score interval for each binomial proportion r/n (vectorized)."""
    from scipy.stats import norm
    z = norm.ppf(0.5 + level / 2)
    r, n = np.asarray(r, float), np.asarray(n, float)
    p = r / n
    den = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return centre - half, centre + half


def mover_interval(c, r, n, level: float = 0.95):
    """MOVER (method of variance estimates recovery; Zou and Donner 2008) interval for theta = sum_i c_i p_i
    from independent binomial arms, built from the Wilson interval of each arm.  It generalizes Newcombe's
    hybrid score interval for a risk difference to the fixed coefficients of the network estimator, is
    defined at zero counts, and stays inside the range of the coefficients' feasible values."""
    c, r, n = np.asarray(c, float), np.asarray(r, float), np.asarray(n, float)
    p = r / n
    lo, hi = wilson_interval(r, n, level)
    pos, neg = c > 0, c < 0
    th = float(c @ p)
    dl = np.sqrt(np.sum((c[pos] * (p[pos] - lo[pos])) ** 2) + np.sum((c[neg] * (hi[neg] - p[neg])) ** 2))
    du = np.sqrt(np.sum((c[pos] * (hi[pos] - p[pos])) ** 2) + np.sum((c[neg] * (p[neg] - lo[neg])) ** 2))
    return th - dl, th + du


# --------------------------------------------------------------------------- #
# Nuisance-free e-value interval for a fixed linear combination of binomial risks
# --------------------------------------------------------------------------- #
EMIX_GRID = np.array([0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 0.90])


def emixture_interval(c, r, n, alpha: float = 0.05, tol: float = 1e-12):
    """Finite-sample (1-alpha) interval for psi = sum_i c_i p_i from independent Binomial(n_i, p_i) arms,
    valid for every risk vector (no mean model, no nuisance search).

    For fixed lambda with 1 + lambda a_i > 0 (a_i = c_i/n_i), E_lambda(psi0) = exp(-lambda psi0) prod_i
    (1 + lambda a_i)^R_i has expectation prod_i (1 + lambda a_i p_i)^n_i exp(-lambda psi) <= 1 at the true psi,
    because log(1+x) <= x; E^H_lambda(psi0) = exp{lambda (psihat - psi0) - lambda^2 V/8}, V = sum c_i^2/n_i, is
    an e-value by Hoeffding's lemma.  For each tail, half the mass is spread over the event-product e-values
    with lambda = +-g/b (g in EMIX_GRID, b = max|a_i|) and half is put on one Hoeffding e-value with
    lambda_H = +-sqrt(8 log(4/alpha)/V).  The interval keeps psi0 where both one-sided mixtures are <= 2/alpha
    (Markov's inequality and a union bound).  All tuning depends only on (c, n, alpha).
    The interval lies inside psihat -+ sqrt(V log(4/alpha)/2) (Hoeffding at alpha/2) and, with all counts
    zero, inside +-(b/0.9) log(32/alpha).
    """
    from scipy.special import logsumexp
    c, r, n = np.asarray(c, float), np.asarray(r, float), np.asarray(n, float)
    keep = c != 0
    c, r, n = c[keep], r[keep], n[keep]
    if c.size == 0:
        return 0.0, 0.0
    a = c / n
    psihat = float(c @ (r / n))
    V = float(np.sum(c ** 2 / n))
    b = float(np.max(np.abs(a)))
    J = len(EMIX_GRID)
    thr = np.log(2.0 / alpha)
    lo_sup, hi_sup = float(np.minimum(c, 0).sum()), float(np.maximum(c, 0).sum())

    def log_mix(psi0, s):
        lam = s * EMIX_GRID / b
        A = np.log1p(np.outer(lam, a)) @ r
        lh = s * np.sqrt(8 * np.log(4 / alpha) / V)
        terms = np.r_[A - lam * psi0 + np.log(0.5 / J), lh * (psihat - psi0) - lh * lh * V / 8 + np.log(0.5)]
        return float(logsumexp(terms))

    def bisect(f, left, right):          # f(left) > thr >= f(right) for the lower end, mirrored for the upper
        for _ in range(200):
            mid = 0.5 * (left + right)
            if f(mid) > thr:
                left = mid
            else:
                right = mid
            if abs(right - left) < tol:
                break
        return left, right

    lo = lo_sup
    if log_mix(lo_sup, +1) > thr:        # positive-lambda mixture decreases in psi0
        lo = bisect(lambda x: log_mix(x, +1), lo_sup, psihat)[0]
    hi = hi_sup
    if log_mix(hi_sup, -1) > thr:        # negative-lambda mixture increases in psi0
        l2, r2 = bisect(lambda x: log_mix(-x, -1), -hi_sup, -psihat)
        hi = -l2
    return float(lo), float(hi)
