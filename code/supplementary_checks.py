"""Checks quoted in the supplement that are not produced elsewhere (Supplements S2, S4 and S10).

1. COPD sensitivity to risk-adjusted weights n_ku / {pi_u (1 - pi_u)}, pi_u the pooled event rate of
   treatment u (outcome-dependent, so outside the minimax statement): placebo estimates, direct masses and
   exact SEs against arm-size weights; and the n-1 variance estimator for placebo vs TIO-HH.
2. Exact coverage of the e-value interval (rdcsp.emixture_interval) by enumeration of every outcome for arm
   sizes (8, 11, 6) and coefficients (-1, 0.6, 0.4) on a grid of risk vectors.
3. Exact mean and variance of Q_inc and Q_het (Proposition 2) against 20,000 binomial simulations at the
   constrained maximum-likelihood null, COPD and liver networks.
Usage: python3 supplementary_checks.py -> ../results/supplementary_checks.json
"""
import itertools
import json

import numpy as np
import pandas as pd
from scipy.stats import binom

from rdcsp import Network, emixture_interval, constrained_null_mle, qinc_exact_moments, inconsistency_projector

out = {}
# 1. risk-adjusted weights, COPD
d = pd.read_csv("../data/dong2013_copd_mortality.csv")[["study", "treatment", "events", "n"]]
net = Network(d)
pool = d.groupby("treatment").apply(lambda g: g.events.sum() / g.n.sum(), include_groups=False)
w_alt = net.n / np.array([pool[t] * (1 - pool[t]) for t in net.arm_trt])
alt = Network(d, weights=w_alt)
rows = {}
for t in ["ICS", "LABA", "LABA-ICS", "TIO-HH", "TIO-SMI"]:
    r0, r1 = net.contributions("Placebo", t), alt.contributions("Placebo", t)
    rows[t] = dict(estimate_arm_size=r0.attrs["estimate"], estimate_risk_adjusted=r1.attrs["estimate"],
                   direct_mass_arm_size=1 - r0.attrs["q"], direct_mass_risk_adjusted=1 - r1.attrs["q"],
                   se_arm_size=float(np.sqrt(net.exact_variance("Placebo", t))),
                   se_risk_adjusted=float(np.sqrt(alt.exact_variance("Placebo", t))))
out["copd_risk_adjusted_weights"] = rows
out["copd_risk_adjusted_max_abs_estimate_change"] = max(abs(v["estimate_arm_size"] - v["estimate_risk_adjusted"]) for v in rows.values())
c = net.coefficient_vector("Placebo", "TIO-HH")
p = net.r / net.n
out["copd_se_n"] = float(np.sqrt(np.sum(c ** 2 * p * (1 - p) / net.n)))
out["copd_se_n_minus_1"] = float(np.sqrt(np.sum(c ** 2 * p * (1 - p) / np.maximum(net.n - 1, 1))))

# 2. exact coverage of the e-value interval
n = np.array([8, 11, 6])
cc = np.array([-1.0, 0.6, 0.4])
outcomes = list(itertools.product(*[range(k + 1) for k in n]))
ints = {r: emixture_interval(cc, np.array(r, float), n.astype(float)) for r in outcomes}
grid = [0.0, 0.02, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
worst, argworst = 1.0, None
for pv in itertools.product(grid, repeat=3):
    pv = np.array(pv)
    psi = float(cc @ pv)
    cov = sum(np.prod(binom.pmf(r, n, pv)) for r in outcomes if ints[r][0] - 1e-12 <= psi <= ints[r][1] + 1e-12)
    if cov < worst:
        worst, argworst = cov, pv.tolist()
out["emixture_exact_coverage"] = dict(arm_sizes=n.tolist(), coefficients=cc.tolist(), risk_grid=grid,
                                      n_risk_vectors=len(grid) ** 3, min_coverage=worst, at=argworst)

# 3. exact moments of Q against simulation
rng = np.random.default_rng(77)
for name, f in [("copd", "dong2013_copd_mortality.csv"), ("liver", "gurusamy2011_liver_transplant_mortality.csv")]:
    g = Network(pd.read_csv("../data/" + f)[["study", "treatment", "events", "n"]])
    p0 = np.clip(constrained_null_mle(g)[0], 0, 1)
    P, _, sw = inconsistency_projector(g)
    Xd = sw[:, None] * g.design_specific()
    Ahet = np.eye(len(sw)) - Xd @ np.linalg.pinv(Xd.T @ Xd) @ Xd.T
    res = {}
    Y = rng.binomial(g.n.astype(int)[None, :], p0[None, :], size=(20000, len(p0))) / g.n
    Z = Y * sw[None, :]
    for lab, A in [("Q_inc", P), ("Q_het", Ahet)]:
        mean, var = qinc_exact_moments(g, p0, P=A, sw=sw)
        q = np.einsum("bi,ij,bj->b", Z, A, Z)
        res[lab] = dict(exact_mean=mean, sim_mean=float(q.mean()), exact_var=var, sim_var=float(q.var()),
                        mc_se_mean=float(q.std() / np.sqrt(len(q))))
    out[f"Q_moments_{name}"] = res
json.dump(out, open("../results/supplementary_checks.json", "w"), indent=1)
print(json.dumps(out, indent=1))
