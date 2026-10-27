"""Simulation study.
Part A: bias and interval coverage of the exact-binomial risk-scale estimator versus the
        inverse-variance risk-difference NMA with 0.5 continuity correction.
Part B: calibration of Q_inc and Q_het (type I error) and power under inconsistency,
        design-fixed weights with binomial calibration versus Pearson weights with the
        chi-square reference and with binomial calibration.
Writes ../results/sim_*.csv and ../results/sim_summary.json.
"""
import json, time
import numpy as np, pandas as pd
from scipy.stats import chi2
from rdcsp import Network, q_decomposition, wls_fit, pearson_weights, iv_rd_estimate

OUT = "../results"
rng = np.random.default_rng(2026)
Z = 1.959963984540054


def make_design(seed, nmin=40, nmax=160):
    r = np.random.default_rng(seed)
    designs = [("A", "B")] * 3 + [("A", "C")] * 2 + [("A", "D")] * 2 + [("B", "C"), ("C", "D"), ("A", "B", "C")]
    rows = []
    for i, d in enumerate(designs):
        for t in d:
            rows.append(dict(study=f"s{i+1}", treatment=t, events=0, n=int(r.integers(nmin, nmax + 1))))
    return pd.DataFrame(rows)


def true_probs(net, alpha, mu):
    return np.array([alpha[s] + mu[t] for s, t in zip(net.arm_study, net.arm_trt)])


def hat_matrices(net):
    sw = np.sqrt(net.w)
    def A(X):
        Xw = sw[:, None] * X
        H = Xw @ np.linalg.pinv(Xw.T @ Xw) @ Xw.T
        return (np.eye(len(sw)) - H) * sw[None, :]      # Q = ||A y||^2
    Ac, Ad = A(net.design_consistency()), A(net.design_specific())
    return Ac, Ad


def q_fixed(Ac, Ad, Y):
    """Y: (B, m) matrix of proportions -> Q_net, Q_het, Q_inc for each row."""
    Rc, Rd = Y @ Ac.T, Y @ Ad.T
    Qn, Qh = (Rc ** 2).sum(1), (Rd ** 2).sum(1)
    return Qn, Qh, Qn - Qh


# ------------------------------------------------------------------------- #
# Part A
# ------------------------------------------------------------------------- #
def part_a(reps=4000, B=2000, rate="low"):
    df = make_design(1)
    net = Network(df)
    nint = net.n.astype(int)
    r = np.random.default_rng(10 if rate == "low" else 11)
    lo, hi = (0.01, 0.04) if rate == "low" else (0.002, 0.012)
    alpha = {s: r.uniform(lo, hi) for s in net.studies}
    mu = {"A": 0.0, "B": -0.008, "C": 0.0, "D": 0.012} if rate == "low" else {"A": 0.0, "B": -0.0015, "C": 0.0, "D": 0.004}
    p = true_probs(net, alpha, mu)
    assert p.min() > 0
    targets = [("A", "D"), ("B", "D")]
    coefs = {t: net.coefficient_vector(*t) for t in targets}
    truth = {t: mu[t[1]] - mu[t[0]] for t in targets}
    exact_var = {t: float(np.sum(coefs[t] ** 2 * p * (1 - p) / net.n)) for t in targets}
    rec = {t: dict(est=[], cov_pct=[], cov_basic=[], cov_norm=[], cov_iv=[], w_pct=[], w_norm=[], w_iv=[], iv_est=[]) for t in targets}
    zero_cells = 0
    for i in range(reps):
        y = r.binomial(nint, p) / net.n
        zero_cells += int(np.any(y == 0))
        for t in targets:
            c = coefs[t]
            est = float(c @ y)
            se = np.sqrt(np.sum(c ** 2 * y * (1 - y) / net.n))
            sims = (r.binomial(nint, y, size=(B, len(c))) / net.n) @ c
            lo_, hi_ = np.quantile(sims, [0.025, 0.975])
            blo, bhi = 2 * est - hi_, 2 * est - lo_
            net.r = y * net.n; net.phat = y          # for the IV comparator
            iv, ivse = iv_rd_estimate(net, *t)
            d = rec[t]
            d["est"].append(est); d["iv_est"].append(iv)
            d["cov_pct"].append(lo_ <= truth[t] <= hi_)
            d["cov_basic"].append(blo <= truth[t] <= bhi)
            d["cov_norm"].append(est - Z * se <= truth[t] <= est + Z * se)
            d["cov_iv"].append(iv - Z * ivse <= truth[t] <= iv + Z * ivse)
            d["w_pct"].append(hi_ - lo_); d["w_norm"].append(2 * Z * se); d["w_iv"].append(2 * Z * ivse)
    rows = []
    for t in targets:
        d = {k: np.array(v) for k, v in rec[t].items()}
        rows.append(dict(rate=rate, target=f"{t[0]}:{t[1]}", truth=truth[t],
                         mean_est=d["est"].mean(), bias=d["est"].mean() - truth[t],
                         exact_sd=np.sqrt(exact_var[t]), empirical_sd=d["est"].std(ddof=1),
                         iv_bias=d["iv_est"].mean() - truth[t], iv_sd=d["iv_est"].std(ddof=1),
                         cov_percentile=d["cov_pct"].mean(), cov_basic=d["cov_basic"].mean(),
                         cov_exactvar_normal=d["cov_norm"].mean(), cov_iv_cc=d["cov_iv"].mean(),
                         width_percentile=d["w_pct"].mean(), width_normal=d["w_norm"].mean(), width_iv=d["w_iv"].mean(),
                         reps=reps, frac_datasets_with_zero_arm=zero_cells / reps,
                         mc_se_coverage=np.sqrt(0.95 * 0.05 / reps)))
    return rows, dict(event_rate_range=(lo, hi), mean_total_n=int(net.n.sum()), arms=int(len(net.n)))


# ------------------------------------------------------------------------- #
# Part B
# ------------------------------------------------------------------------- #
def part_b(reps=2000, Binner=400, rate="low", delta=0.0):
    df = make_design(1)
    net = Network(df)
    nint = net.n.astype(int)
    r = np.random.default_rng(20 + int(delta * 1000) + (0 if rate == "low" else 1))
    lo, hi = (0.01, 0.04) if rate == "low" else (0.002, 0.012)
    alpha = {s: r.uniform(lo, hi) for s in net.studies}
    mu = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}
    p = true_probs(net, alpha, mu)
    # inconsistency: shift the B:C design (study s8) by delta
    s8 = net.rows["s8"]
    p = p.copy()
    p[s8[1]] += delta
    Ac, Ad = hat_matrices(net)
    Xc = net.design_consistency()
    dfinc, dfhet = q_decomposition(net)["df_inc"], q_decomposition(net)["df_het"]
    rej = dict(fixed_binom_inc=0, fixed_binom_het=0, pearson_chi2_inc=0, pearson_chi2_het=0,
               pearson_binom_inc=0, pearson_binom_het=0)
    for i in range(reps):
        y = r.binomial(nint, p) / net.n
        # design-fixed Q, binomial calibration at the null fit (floored at half an event)
        Qn, Qh, Qi = q_fixed(Ac, Ad, y[None, :])
        fit = wls_fit(Xc, y, net.w)
        p0 = np.minimum(np.maximum(fit, 0.5 / net.n), 1.0)
        Yb = r.binomial(nint, p0, size=(Binner, len(y))) / net.n
        Qnb, Qhb, Qib = q_fixed(Ac, Ad, Yb)
        rej["fixed_binom_inc"] += np.mean(Qib >= Qi[0]) < 0.05
        rej["fixed_binom_het"] += np.mean(Qhb >= Qh[0]) < 0.05
        # Pearson weights at the null fit: chi-square reference
        wp = pearson_weights(net, fit)
        qp = q_decomposition(net, y, wp)
        rej["pearson_chi2_inc"] += (1 - chi2.cdf(qp["Q_inc"], dfinc)) < 0.05
        rej["pearson_chi2_het"] += (1 - chi2.cdf(qp["Q_het"], dfhet)) < 0.05
        # Pearson weights with binomial calibration (weights re-estimated per draw), smaller inner loop
        Qi_b, Qh_b = [], []
        for b in range(80):
            yb = Yb[b]
            fb = wls_fit(Xc, yb, net.w)
            qb = q_decomposition(net, yb, pearson_weights(net, fb))
            Qi_b.append(qb["Q_inc"]); Qh_b.append(qb["Q_het"])
        rej["pearson_binom_inc"] += np.mean(np.array(Qi_b) >= qp["Q_inc"]) < 0.05
        rej["pearson_binom_het"] += np.mean(np.array(Qh_b) >= qp["Q_het"]) < 0.05
    out = {k: v / reps for k, v in rej.items()}
    out.update(rate=rate, delta=delta, reps=reps, mc_se=np.sqrt(0.05 * 0.95 / reps))
    return out


if __name__ == "__main__":
    t0 = time.time()
    rowsA, metaA = [], {}
    for rate in ["low", "verylow"]:
        rows, meta = part_a(reps=4000, B=2000, rate=rate)
        rowsA += rows; metaA[rate] = meta
    pd.DataFrame(rowsA).to_csv(f"{OUT}/sim_partA.csv", index=False)
    print(pd.DataFrame(rowsA).round(4).to_string()); print("part A time", round(time.time() - t0), "s")
    t1 = time.time()
    rowsB = []
    for rate in ["low", "verylow"]:
        for delta in [0.0, 0.03, 0.06]:
            rowsB.append(part_b(reps=1500, Binner=400, rate=rate, delta=delta))
            print(rowsB[-1], round(time.time() - t1), "s")
    pd.DataFrame(rowsB).to_csv(f"{OUT}/sim_partB.csv", index=False)
    json.dump(dict(partA_meta=metaA, time_A=time.time() - t0, time_B=time.time() - t1), open(f"{OUT}/sim_summary.json", "w"), indent=1, default=float)
