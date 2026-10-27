"""Coverage of 95% intervals for the design-fixed estimator across designs (Section 3.9, Table 1).

Intervals: basic parametric bootstrap (1,000 draws), normal with the exact plug-in variance, MOVER-Wilson
(rdcsp.mover_interval), and the guaranteed Clopper-Pearson/LP interval.  Designs and true risks:
  synthetic ten-study network, risks 1-4% and 0.2-1.2% (run_simulations geometry and baselines), target A:D;
  cystic fibrosis network, constrained-MLE risks, TEZ/IVA vs triple;
  NMOSD star, constrained-MLE risks with zero arms at half an event, control vs rituximab (19 vs 19) and
  eculizumab vs rituximab (indirect);
  COPD network, constrained-MLE risks floored at half an event, placebo vs TIO-HH.
Usage: python3 interval_coverage.py [reps]  -> ../results/interval_coverage.csv, interval_coverage.tex
"""
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.stats import beta as _beta

from rdcsp import Network, constrained_null_mle, mover_interval
from run_simulations import make_design, true_probs

reps = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
Z = 1.959963984540054
t0 = time.time()


def cp_box(r, n, level):
    a = (1 - level) / 2
    lo = np.where(r > 0, _beta.ppf(a, r, n - r + 1), 0.0)
    hi = np.where(r < n, _beta.ppf(1 - a, r + 1, n - r), 1.0)
    return lo, hi


def lp_interval(net, X, g, r, n, gamma=0.05):
    lo, hi = cp_box(r, n, 1 - gamma / len(n))
    A = np.vstack([X, -X])
    b = np.r_[hi, -lo]
    res_lo = linprog(g, A_ub=A, b_ub=b, bounds=[(None, None)] * X.shape[1], method="highs")
    res_hi = linprog(-g, A_ub=A, b_ub=b, bounds=[(None, None)] * X.shape[1], method="highs")
    if not (res_lo.success and res_hi.success):
        return np.nan, np.nan
    return res_lo.fun, -res_hi.fun


def run(label, net, p_true, a, b, seed, with_lp=True):
    nint = net.n.astype(int)
    c = net.coefficient_vector(a, b)
    theta = float(c @ p_true)
    X = net.design_consistency()
    g = np.zeros(X.shape[1])
    tr = net.trts[1:]
    if b in tr:
        g[X.shape[1] - len(tr) + tr.index(b)] += 1
    if a in tr:
        g[X.shape[1] - len(tr) + tr.index(a)] -= 1
    rng = np.random.default_rng(seed)
    hit = dict(basic=0, normal=0, mover=0, lp=0)
    width = dict(basic=0.0, normal=0.0, mover=0.0, lp=0.0)
    n_lp = 0
    for i in range(reps):
        r = rng.binomial(nint, p_true)
        y = r / net.n
        th = float(c @ y)
        Ystar = rng.binomial(nint, y, size=(1000, len(y))) / net.n
        q_lo, q_hi = np.quantile(Ystar @ c, [0.025, 0.975])
        ints = {"basic": (2 * th - q_hi, 2 * th - q_lo)}
        v = float(np.sum(c ** 2 * y * (1 - y) / np.maximum(net.n - 1, 1)))
        ints["normal"] = (th - Z * np.sqrt(v), th + Z * np.sqrt(v))
        ints["mover"] = mover_interval(c, r, net.n)
        if with_lp and i < 500:
            ints["lp"] = lp_interval(net, X, g, r, net.n)
            n_lp += 1
        for k, (lo, hi) in ints.items():
            if np.isfinite(lo):
                hit[k] += lo <= theta <= hi
                width[k] += hi - lo
    out = dict(design=label, target=f"{a} vs {b}", theta=theta, reps=reps)
    for k in ["basic", "normal", "mover"]:
        out[f"cov_{k}"] = hit[k] / reps
        out[f"width_{k}"] = width[k] / reps
    out["cov_lp"] = hit["lp"] / n_lp if n_lp else np.nan
    out["width_lp"] = width["lp"] / n_lp if n_lp else np.nan
    print(out, flush=True)
    return out


rows = []
for rate, lohi, seed in [("low", (0.01, 0.04), 40), ("verylow", (0.002, 0.012), 41)]:
    df = make_design(1)
    net = Network(df)
    rg = np.random.default_rng(seed)
    alpha = {s: rg.uniform(*lohi) for s in net.studies}
    p = true_probs(net, alpha, {"A": 0.0, "B": 0.008, "C": 0.004, "D": 0.012 if rate == "low" else 0.004})
    rows.append(run(f"Synthetic, risks {'1-4' if rate == 'low' else '0.2-1.2'}%", net, p, "A", "D", 100 + seed))
for f, drop, tg, lab, seed in [("cf_sae_registry.csv", None, [("TEZ/IVA", "Triple")], "Cystic fibrosis", 200),
                               ("nmosd2025_relapse.csv", ["CHAMPION"], [("Control", "Rituximab"), ("Eculizumab", "Rituximab")], "NMOSD star", 300),
                               ("dong2013_copd_mortality.csv", None, [("Placebo", "TIO-HH")], "COPD", 400)]:
    d = pd.read_csv("../data/" + f)[["study", "treatment", "events", "n"]]
    if drop:
        d = d[~d.study.isin(drop)]
    net = Network(d)
    p = np.clip(constrained_null_mle(net, "consistency")[0], 0, 1)
    p = np.where(p < 0.5 / net.n, 0.5 / net.n, p)
    for j, (a, b) in enumerate(tg):
        rows.append(run(lab, net, p, a, b, seed + j, with_lp=(lab != "COPD")))
D = pd.DataFrame(rows)
D.to_csv("../results/interval_coverage.csv", index=False)
L = [r"\begin{tabular}{llrrrrrrrr}", r"\toprule",
     r" & & \multicolumn{2}{c}{Basic bootstrap} & \multicolumn{2}{c}{Normal, exact variance} & \multicolumn{2}{c}{MOVER-Wilson} & \multicolumn{2}{c}{Clopper--Pearson/LP} \\",
     r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}",
     r"Design & Comparison & Cov. & Width & Cov. & Width & Cov. & Width & Cov. & Width \\", r"\midrule"]
for _, r in D.iterrows():
    lp = (f"{r['cov_lp']:.3f} & {r['width_lp']:.3f}") if np.isfinite(r["cov_lp"]) else "-- & --"
    L.append(f"{r['design'].replace('%', chr(92) + '%')} & {r['target']} & {r['cov_basic']:.3f} & {r['width_basic']:.3f} & "
             f"{r['cov_normal']:.3f} & {r['width_normal']:.3f} & {r['cov_mover']:.3f} & {r['width_mover']:.3f} & {lp} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
open("../results/interval_coverage.tex", "w").write("\n".join(L) + "\n")
print(round(time.time() - t0), "s")
