"""Simulation on the COPD geometry when the risk-difference model is false (Section 4).

Two truths on the arm sizes of the 41 COPD trials:
  additive: the constrained maximum-likelihood fit of the observed deaths under model (1), so the
            risk-difference model holds exactly;
  logit:    a common odds ratio for each treatment, logit p_ku = a_k + b_u, with trial baselines a_k taken
            from each trial's pooled mortality (0.07% to 14%) and log odds ratios b = (ICS 0.05, LABA -0.10,
            LABA-ICS -0.20, TIO-HH -0.12, TIO-SMI 0.35) relative to placebo; trial-level risk differences then
            vary with baseline risk and model (1) is false.
For placebo vs TIO-HH (direct and indirect evidence) and placebo vs TIO-SMI (pendant), the estimator's
target is the projected value theta* = c'p of Theorem 1(iii).  Recorded: bias for theta*, coverage of the
basic bootstrap, exact-variance normal and MOVER-Wilson intervals for theta*, theta* against the risk difference at the
network's patient-weighted placebo risk and against the range of trial-level risk differences, and the
rejection rates of Q_het and Q_inc (plug-in calibration at the constrained-MLE null, 400 draws) and of T_W
with the chi-square reference, i.e. the power of the paper's diagnostics to detect the misspecification.

Usage: python3 sim_misspec.py <additive|logit> [reps]  -> ../results/sim_misspec_<truth>.csv
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import chi2

from rdcsp import (Network, q_decomposition, inconsistency_projector, qinc_from_P, wald_inconsistency_stat,
                   constrained_null_mle, mover_interval)
from run_simulations import hat_matrices, q_fixed

truth = sys.argv[1]
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
t0 = time.time()
df = pd.read_csv("../data/dong2013_copd_mortality.csv")[["study", "treatment", "events", "n"]]
net = Network(df)
nint = net.n.astype(int)
B_LOGOR = {"Placebo": 0.0, "ICS": 0.05, "LABA": -0.10, "LABA-ICS": -0.20, "TIO-HH": -0.12, "TIO-SMI": 0.35}

if truth == "additive":
    p_true = np.clip(constrained_null_mle(net, "consistency")[0], 0, 1)
else:
    pooled = df.groupby(df.study.astype(str)).apply(lambda g: (g.events.sum() + 0.5) / (g.n.sum() + 1.0), include_groups=False)
    b = np.array([B_LOGOR[t] for t in net.arm_trt])
    a = np.array([logit(pooled[s]) for s in net.arm_study])
    # centre each trial's log odds ratios so that its baseline matches its pooled mortality
    adj = pd.Series(b).groupby(net.arm_study).transform("mean").to_numpy()
    p_true = expit(a + b - adj)

targets = [("Placebo", "TIO-HH"), ("Placebo", "TIO-SMI")]
coef = {t: net.coefficient_vector(*t) for t in targets}
theta = {t: float(coef[t] @ p_true) for t in targets}
pl = net.arm_trt == "Placebo"
p_bar = float(np.sum(net.n[pl] * p_true[pl]) / np.sum(net.n[pl]))


def rd_at(pbase, t):
    if truth == "additive":
        return float("nan")
    return float(expit(logit(pbase) + B_LOGOR[t[1]]) - pbase)


trial_rd = {}
for t in targets:
    vals = []
    for k in net.studies:
        idx = net.rows[k]
        arms = list(net.arm_trt[idx])
        if t[0] in arms and t[1] in arms:
            vals.append(p_true[idx[arms.index(t[1])]] - p_true[idx[arms.index(t[0])]])
    trial_rd[t] = (min(vals), max(vals))

Ac, Ad = hat_matrices(net)
P, Bm, sw = inconsistency_projector(net)
q0 = q_decomposition(net)
dfi = int(q0["df_inc"])
rng = np.random.default_rng(900 if truth == "additive" else 901)
Z = 1.959963984540054
est = {t: [] for t in targets}
cov_b = {t: 0 for t in targets}
cov_n = {t: 0 for t in targets}
cov_m = {t: 0 for t in targets}
rej = dict(qhet=0, qinc=0, tw_chi2=0)
for i in range(reps):
    y = rng.binomial(nint, p_true) / net.n
    net.r, net.phat = y * net.n, y
    Ystar = rng.binomial(nint, y, size=(1000, len(y))) / net.n
    for t in targets:
        c = coef[t]
        th = float(c @ y)
        est[t].append(th)
        v = float(np.sum(c ** 2 * y * (1 - y) / np.maximum(net.n - 1, 1)))
        cov_n[t] += abs(th - theta[t]) <= Z * np.sqrt(v)
        lo, hi = np.quantile(Ystar @ c, [0.025, 0.975])
        cov_b[t] += (2 * th - hi) <= theta[t] <= (2 * th - lo)
        ml, mu = mover_interval(c, y * net.n, net.n)
        cov_m[t] += ml <= theta[t] <= mu
    Qn, Qh, Qi = q_fixed(Ac, Ad, y[None, :])
    p0, _ = constrained_null_mle(net, "consistency")
    Yb = rng.binomial(nint, np.clip(p0, 0, 1), size=(400, len(y))) / net.n
    Qnb, Qhb, Qib = q_fixed(Ac, Ad, Yb)
    rej["qhet"] += np.mean(Qhb >= Qh[0]) < 0.05
    rej["qinc"] += np.mean(Qib >= Qi[0]) < 0.05
    tobs = float(wald_inconsistency_stat(net, y[None, :], P, Bm, sw)[0])
    rej["tw_chi2"] += chi2.sf(tobs, dfi) < 0.05

row = dict(truth=truth, reps=reps, time_s=round(time.time() - t0), placebo_risk=p_bar)
for t in targets:
    lab = f"{t[0]}:{t[1]}"
    e = np.array(est[t])
    row[f"theta_star {lab}"] = theta[t]
    row[f"rd_at_pbar {lab}"] = rd_at(p_bar, t)
    row[f"trial_rd_min {lab}"], row[f"trial_rd_max {lab}"] = trial_rd[t]
    row[f"bias {lab}"] = float(e.mean() - theta[t])
    row[f"bias_mcse {lab}"] = float(e.std(ddof=1) / np.sqrt(reps))
    row[f"cov_basic {lab}"] = cov_b[t] / reps
    row[f"cov_normal {lab}"] = cov_n[t] / reps
    row[f"cov_mover {lab}"] = cov_m[t] / reps
for k, v in rej.items():
    row[k] = v / reps
f = f"../results/sim_misspec_{truth}.csv"
old = pd.DataFrame()
if len(old):
    old = old[old.truth != truth]
pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(f, index=False)
print(row)
