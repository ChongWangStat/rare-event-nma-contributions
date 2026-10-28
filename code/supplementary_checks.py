"""Checks quoted in the supplement that are not produced elsewhere (Supplements S2, S4 and S10).

1. COPD sensitivity to risk-adjusted weights n_ku / {pi_u (1 - pi_u)}, pi_u the pooled event rate of
   treatment u (outcome-dependent, so outside the minimax statement): placebo estimates, direct masses and
   exact SEs against arm-size weights; and the n-1 variance estimator for placebo vs TIO-HH.
2. Exact coverage of the e-value interval (rdcsp.emixture_interval) by enumeration of every outcome for arm
   sizes (8, 11, 6) and coefficients (-1, 0.6, 0.4) on a grid of risk vectors.
3. Exact mean and variance of Q_inc and Q_het (Proposition 3) against 20,000 binomial simulations at the
   constrained maximum-likelihood null, COPD and liver networks.
4. Proposition 2 (identification of routes) by brute force: every a-b route of the indirect flow is enumerated,
   (ii) is compared with rank(A) = M for all 77 targets of the six networks, and for the COPD network the
   closed-form bounds (iii) of the eight largest canonical routes of every target are compared with linear
   programs over Pi(f), and the study visit masses (i) and the indirect sum are checked at random vertices
   of Pi(f).
Usage: python3 supplementary_checks.py -> ../results/supplementary_checks.json
"""
import itertools
import json

import numpy as np
import pandas as pd
from scipy.stats import binom

from rdcsp import (Network, emixture_interval, constrained_null_mle, qinc_exact_moments, inconsistency_projector,
                   study_pairings, bottleneck_paths, route_identification)
from scipy.optimize import linprog
from collections import defaultdict

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

# 4. Proposition 2 by brute force
def indirect_flow(net, a, b):
    direct, edges = study_pairings(net, a, b)
    ind = [e for e in edges if not (e[0] == a and e[1] == b and direct.get(e[2], 0) > 0 and e[3] == direct[e[2]])]
    return direct, [e for e in ind if e[3] > 1e-12]


def all_routes(edges, a, b):
    out_ = defaultdict(list)
    for e in edges:
        out_[e[0]].append(e)
    res = []
    def dfs(u, path):
        if u == b:
            res.append(tuple(path))
            return
        for e in out_[u]:
            dfs(e[1], path + [e])
    dfs(a, [])
    return res


nets = {"COPD": Network(pd.read_csv("../data/dong2013_copd_mortality.csv")[["study", "treatment", "events", "n"]]),
        "liver": Network(pd.read_csv("../data/gurusamy2011_liver_transplant_mortality.csv")[["study", "treatment", "events", "n"]]),
        "COVID-19": Network(pd.read_csv("../data/covid19_wholenetwork.csv")[["study", "treatment", "events", "n"]]),
        "cystic fibrosis": Network(pd.read_csv("../data/cf_sae_registry.csv")[["study", "treatment", "events", "n"]]),
        "NMOSD": Network(pd.read_csv("../data/nmosd2025_relapse.csv").query("study != 'CHAMPION'")[["study", "treatment", "events", "n"]]),
        "SMA safety": Network(pd.read_csv("../data/sma_nusinersen_pneumonia.csv")[["study", "treatment", "events", "n"]])}
rng4 = np.random.default_rng(2026)
crit = dict(targets=0, with_indirect=0, unique=0, agree=0, max_routes=0)
lp = dict(routes_checked=0, max_abs_bound_gap=0.0, max_abs_visit_gap=0.0, max_abs_indirect_gap=0.0)
for name, nt in nets.items():
    trts = sorted(set(nt.arm_trt))
    phat = nt.r / nt.n
    for a, b in itertools.combinations(trts, 2):
        crit["targets"] += 1
        direct, E = indirect_flow(nt, a, b)
        if not E:
            continue
        crit["with_indirect"] += 1
        R = all_routes(E, a, b)
        crit["max_routes"] = max(crit["max_routes"], len(R))
        eidx = {(u, v, k): i for i, (u, v, k, f) in enumerate(E)}
        A = np.zeros((len(E), len(R)))
        for j, r in enumerate(R):
            for (u, v, k, f) in r:
                A[eidx[(u, v, k)], j] = 1.0
        f = np.array([e[3] for e in E])
        unique = route_identification(E, a, b)["unique"]
        crit["unique"] += int(unique)
        crit["agree"] += int(unique == (np.linalg.matrix_rank(A) == len(R)))
        if name != "COPD":
            continue
        ident = route_identification(E, a, b)
        paths, _ = bottleneck_paths(E, a, b)
        flow = {(u, v, k): w for u, v, k, w in E}
        for w, path in sorted(paths, key=lambda z: -z[0])[:8]:
            j = R.index(tuple((u, v, k, flow[(u, v, k)]) for u, v, k in path))
            U = min(flow[e] for e in path)
            L = max(0.0, sum(flow[e] for e in path) - sum(ident["throughput"][e[1]] for e in path[:-1]))
            cvec = np.zeros(len(R)); cvec[j] = 1.0
            lo = linprog(cvec, A_eq=A, b_eq=f, bounds=(0, None), method="highs").fun
            hi = -linprog(-cvec, A_eq=A, b_eq=f, bounds=(0, None), method="highs").fun
            lp["routes_checked"] += 1
            lp["max_abs_bound_gap"] = max(lp["max_abs_bound_gap"], abs(lo - L), abs(hi - U))
        con = nt.contributions(a, b).set_index("study")
        idx_of = {k: nt.rows[k] for k in nt.studies}
        y = []
        for (u, v, k, fe) in E:
            ii = list(idx_of[k]); arms = list(nt.arm_trt[ii])
            y.append(phat[ii[arms.index(v)]] - phat[ii[arms.index(u)]])
        delta = A.T @ np.array(y)
        studies_e = [e[2] for e in E]
        for _ in range(5):
            pi = linprog(rng4.standard_normal(len(R)), A_eq=A, b_eq=f, bounds=(0, None), method="highs").x
            lp["max_abs_indirect_gap"] = max(lp["max_abs_indirect_gap"], abs(pi @ delta - con["C_I"].sum()))
            for k in set(studies_e):
                visit = sum(pi[j] for j, r in enumerate(R) if any(e[2] == k for e in r))
                lp["max_abs_visit_gap"] = max(lp["max_abs_visit_gap"], abs(visit - (con.loc[k, "W"] - con.loc[k, "d"])))
out["route_identification_check"] = dict(criterion=crit, copd_lp=lp)

json.dump(out, open("../results/supplementary_checks.json", "w"), indent=1)
print(json.dumps(out, indent=1))
