"""Simulation Parts B and C rerun with the constrained-MLE null generator (replaces the floored
generator).  Usage: python3 run_sim_mle.py <rate> <delta> [reps] [n_supremum]  -> appends to ../results/sim_mle.csv
The Berger-Boos supremum test is run on the first n_supremum data sets of the cell (default 80 for delta=0, 40 otherwise;
the revision uses 400 and 200)."""
import sys, time, os
import numpy as np, pandas as pd
from scipy.stats import chi2
from rdcsp import (Network, q_decomposition, wls_fit, pearson_weights, inconsistency_projector, qinc_from_P,
                   qinc_exact_moments, satterthwaite_p, wald_inconsistency_stat, constrained_null_mle,
                   berger_boos_sup_wald)
from run_simulations import make_design, true_probs, hat_matrices, q_fixed

rate, delta = sys.argv[1], float(sys.argv[2]); reps = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
t0 = time.time()
df = make_design(1); net = Network(df); nint = net.n.astype(int)
r = np.random.default_rng(40 + int(delta * 1000) + (0 if rate == "low" else 1))
lo, hi = (0.01, 0.04) if rate == "low" else (0.002, 0.012)
alpha = {s: r.uniform(lo, hi) for s in net.studies}
p = true_probs(net, alpha, {"A": 0, "B": 0, "C": 0, "D": 0}).copy()
p[net.rows["s8"][1]] += delta
Ac, Ad = hat_matrices(net); P, Bm, sw = inconsistency_projector(net)
dfi, dfh = q_decomposition(net)["df_inc"], q_decomposition(net)["df_het"]
rej = dict(fixed_mle_inc=0, fixed_mle_het=0, satt_mle_inc=0, wald_plugin_mle=0, wald_chi2=0, wald_bb_mle=0, pearson_chi2_inc=0, n_converged=0)
nbb = 0; with_bb = int(sys.argv[4]) if len(sys.argv) > 4 else (80 if delta == 0 else 40)   # data sets given the supremum test
for i in range(reps):
    y = r.binomial(nint, p) / net.n
    net.r, net.phat = y * net.n, y
    Qn, Qh, Qi = q_fixed(Ac, Ad, y[None, :])
    p0, res = constrained_null_mle(net)
    rej["n_converged"] += int(res.success)
    Yb = r.binomial(nint, p0, size=(400, len(y))) / net.n
    Qnb, Qhb, Qib = q_fixed(Ac, Ad, Yb)
    rej["fixed_mle_inc"] += np.mean(Qib >= Qi[0]) < 0.05
    rej["fixed_mle_het"] += np.mean(Qhb >= Qh[0]) < 0.05
    mean, var = qinc_exact_moments(net, p0, P, sw)
    rej["satt_mle_inc"] += (satterthwaite_p(Qi[0], mean, var)[0] if mean > 0 and var > 0 else 1.0) < 0.05
    tobs = float(wald_inconsistency_stat(net, y, P, Bm, sw)[0])
    rej["wald_plugin_mle"] += np.mean(wald_inconsistency_stat(net, Yb, P, Bm, sw) >= tobs) < 0.05
    rej["wald_chi2"] += chi2.sf(tobs, dfi) < 0.05
    fit = wls_fit(net.design_consistency(), y, net.w)
    rej["pearson_chi2_inc"] += chi2.sf(q_decomposition(net, y, pearson_weights(net, fit))["Q_inc"], dfi) < 0.05
    if i < with_bb:
        try:
            rej["wald_bb_mle"] += berger_boos_sup_wald(net, B=800, n_starts=2, n_iter=20, seed=i)["p_value"] < 0.05; nbb += 1
        except ZeroDivisionError:
            pass
out = {k: v / reps for k, v in rej.items() if k not in ("wald_bb_mle", "n_converged")}
out["wald_bb_mle"] = rej["wald_bb_mle"] / nbb; out["frac_converged"] = rej["n_converged"] / reps
out.update(rate=rate, delta=delta, reps=reps, reps_bb=nbb, time_s=round(time.time() - t0))
f = "../results/sim_mle.csv"; old = pd.read_csv(f) if os.path.exists(f) else pd.DataFrame()
if len(old): old = old[~((old.rate == rate) & (old.delta == delta))]
pd.concat([old, pd.DataFrame([out])], ignore_index=True).to_csv(f, index=False); print(out)
