"""Design-based simulation on the real cystic fibrosis network (seven parallel-group trials).

Complements the synthetic simulation (run_simulations.py, run_sim_mle.py), whose geometry is fixed and
artificial, with the geometry, arm sizes and risk levels of an application.  True risks are the
constrained binomial maximum-likelihood fit of the observed SAE counts under the consistency model,
so the null holds exactly; for power, delta is added to the triple-combination arm of the head-to-head
trial VX17-445-103, which creates inconsistency in the loop placebo - TEZ/IVA - triple.

Recorded per cell (1,000 data sets):
  estimator: bias, empirical vs exact SD and coverage of the exact-variance normal interval and of the
             basic parametric-bootstrap interval (1,000 draws) for TEZ/IVA vs triple and placebo vs triple;
  tests at level 0.05: Q_inc with plug-in binomial calibration at the constrained-MLE null (400 draws),
             T_W with chi-square reference, T_W with plug-in calibration, Berger-Boos supremum on T_W
             (first n_sup data sets; B = 800, 2 starts x 20 iterations), Hoeffding bound.

Usage: python3 sim_cf_design.py <delta> [reps] [n_sup]   -> appends a row to ../results/sim_cf_design.csv
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import chi2

from rdcsp import (Network, q_decomposition, inconsistency_projector, qinc_from_P, wald_inconsistency_stat,
                   constrained_null_mle, berger_boos_sup_wald, hoeffding_inconsistency_test)

delta = float(sys.argv[1])
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
n_sup = int(sys.argv[3]) if len(sys.argv) > 3 else 200
t0 = time.time()

df = pd.read_csv("../data/cf_sae_registry.csv")[["study", "treatment", "events", "n"]]
net = Network(df)
nint = net.n.astype(int)
p_true, _ = constrained_null_mle(net, "consistency")
p_true = p_true.copy()
ix = int(np.where((net.arm_study == "VX17-445-103") & (net.arm_trt == "Triple"))[0][0])
p_true[ix] += delta

targets = [("TEZ/IVA", "Triple"), ("Placebo", "Triple")]
coef = {t: net.coefficient_vector(*t) for t in targets}
theta = {t: float(coef[t] @ p_true) for t in targets}           # projected target (= RD under the null)
P, Bm, sw = inconsistency_projector(net)
dfi = int(q_decomposition(net)["df_inc"])
rng = np.random.default_rng(700 + int(round(delta * 1000)))
Z = 1.959963984540054

est = {t: [] for t in targets}
cov_norm = {t: 0 for t in targets}
cov_basic = {t: 0 for t in targets}
rej = dict(qinc_plugin=0, tw_chi2=0, tw_plugin=0, tw_bb=0, hoeffding=0)
nbb = 0
for i in range(reps):
    y = rng.binomial(nint, p_true) / net.n
    net.r, net.phat = y * net.n, y
    # estimator and intervals
    Ystar = rng.binomial(nint, y, size=(1000, len(y))) / net.n
    for t in targets:
        c = coef[t]
        th = float(c @ y)
        est[t].append(th)
        v = float(np.sum(c ** 2 * y * (1 - y) / np.maximum(net.n - 1, 1)))
        cov_norm[t] += abs(th - theta[t]) <= Z * np.sqrt(v)
        q_lo, q_hi = np.quantile(Ystar @ c, [0.025, 0.975])
        cov_basic[t] += (2 * th - q_hi) <= theta[t] <= (2 * th - q_lo)
    # consistency tests
    q_obs = float(qinc_from_P(P, sw, y[None, :])[0])
    p0, _ = constrained_null_mle(net, "consistency")
    Yb = rng.binomial(nint, p0, size=(400, len(y))) / net.n
    rej["qinc_plugin"] += np.mean(qinc_from_P(P, sw, Yb) >= q_obs) < 0.05
    tobs = float(wald_inconsistency_stat(net, y[None, :], P, Bm, sw)[0])
    rej["tw_chi2"] += chi2.sf(tobs, dfi) < 0.05
    rej["tw_plugin"] += np.mean(wald_inconsistency_stat(net, Yb, P, Bm, sw) >= tobs) < 0.05
    rej["hoeffding"] += hoeffding_inconsistency_test(net)["p_value"] < 0.05
    if i < n_sup:
        try:
            rej["tw_bb"] += berger_boos_sup_wald(net, B=800, n_starts=2, n_iter=20, seed=i)["p_value"] < 0.05
            nbb += 1
        except ZeroDivisionError:
            pass

row = dict(delta=delta, reps=reps, reps_sup=nbb, time_s=round(time.time() - t0))
for t in targets:
    lab = f"{t[0]}:{t[1]}"
    e = np.array(est[t])
    c = coef[t]
    row[f"bias {lab}"] = float(e.mean() - theta[t])
    row[f"bias_mcse {lab}"] = float(e.std(ddof=1) / np.sqrt(reps))
    row[f"sd_emp {lab}"] = float(e.std(ddof=1))
    row[f"sd_exact {lab}"] = float(np.sqrt(np.sum(c ** 2 * p_true * (1 - p_true) / net.n)))
    row[f"cov_normal {lab}"] = cov_norm[t] / reps
    row[f"cov_basic {lab}"] = cov_basic[t] / reps
for k, v in rej.items():
    row[k] = v / (nbb if k == "tw_bb" else reps)
f = "../results/sim_cf_design.csv"
old = pd.read_csv(f) if os.path.exists(f) else pd.DataFrame()
if len(old):
    old = old[old.delta != delta]
pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(f, index=False)
print(row)
