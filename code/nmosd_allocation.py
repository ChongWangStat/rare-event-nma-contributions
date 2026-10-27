"""A funder's allocation decision in the NMOSD network (Section 5.3; Theorem 7(iii)).

Question: with a fixed budget of N patients for new randomized trials in AQP4-positive NMOSD, which
trials should be funded, and how large, so that the evidence comparing the five biologics with one
another (the ten active-versus-active comparisons, all of them currently wholly indirect) becomes as
precise as possible?

With arm-size weights, the minimax worst-case variance of comparison a:b is rho(1-rho) r_ab
(Theorem 2), so the decision needs no outcomes.  A new balanced two-arm trial of N_j patients on u:v adds
conductance g_j = N_j/4 on that edge; r_ab is convex in the vector of added conductances (Theorem 7(iii)),
so both objectives below are convex programs with a unique optimal value:
the objective is the average of r_ab over the ten active comparisons (the largest r_ab is reported too).
Because the unrestricted optimum spreads patients over many small trials, the practical decision is also
solved exactly over one or two funded trials (all candidate sets enumerated, splits on a 10-patient grid).
Candidates: all 15 comparisons among control and the five biologics (ravulizumab, single-arm only, is
disconnected under study-specific baselines and is not a candidate).  Budgets: 200 and 400 patients.

Usage: python3 nmosd_allocation.py  -> ../results/nmosd_allocation.json, nmosd_allocation.tex
"""
import json
import os
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from rdcsp import Network

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results")
df = pd.read_csv(os.path.join(HERE, "..", "data", "nmosd2025_relapse.csv"))
net = Network(df[df.study != "CHAMPION"][["study", "treatment", "events", "n"]])
T = net.trts
active = [t for t in T if t != "Control"]
targets = list(combinations(active, 2))                   # 10 active-versus-active comparisons
cands = list(combinations(T, 2))                          # 15 candidate trials
D = np.array([net.e(u, v) for u, v in cands])             # rows d_j
E = np.array([net.e(a, b) for a, b in targets])
L0 = net.L


def resistances(N):
    g = np.maximum(N, 0) / 4.0
    Lp = np.linalg.pinv(L0 + (D.T * g) @ D)
    return np.einsum("ti,ij,tj->t", E, Lp, E), Lp


def mean_obj(N):
    r, Lp = resistances(N)
    # gradient: d r_t / d N_j = -(1/4) (d_j' Lp e_t)^2
    G = (D @ Lp @ E.T) ** 2                               # (cands, targets)
    return r.mean(), -G.mean(axis=1) / 4.0


def solve_mean(budget):
    x0 = np.full(len(cands), budget / len(cands))
    res = minimize(lambda x: mean_obj(x)[0], x0, jac=lambda x: mean_obj(x)[1], method="SLSQP",
                   bounds=[(0, budget)] * len(cands), constraints=[{"type": "eq", "fun": lambda x: x.sum() - budget}],
                   options={"ftol": 1e-14, "maxiter": 1000})
    return res.x


def _unused_solve_worst(budget):
    # variables (N_1..N_15, s): minimize s subject to r_t(N) <= s
    n = len(cands)
    x0 = np.r_[np.full(n, budget / n), resistances(np.full(n, budget / n))[0].max()]

    def cons_r(x):
        return x[-1] - resistances(x[:-1])[0]

    res = minimize(lambda x: x[-1], x0, method="SLSQP", bounds=[(0, budget)] * n + [(0, None)],
                   constraints=[{"type": "eq", "fun": lambda x: x[:-1].sum() - budget}, {"type": "ineq", "fun": cons_r}],
                   options={"ftol": 1e-14, "maxiter": 2000})
    return res.x[:-1]


r0 = resistances(np.zeros(len(cands)))[0]
out = dict(targets=[f"{a}:{b}" for a, b in targets], r_before=r0.tolist(), mean_before=float(r0.mean()), max_before=float(r0.max()))
h2h = [c for c in cands if "Control" not in c]


def plan_vec(spec):
    N = np.zeros(len(cands))
    for c, n in spec.items():
        N[cands.index(c)] = n
    return N


def best_k(budget, pool, k, step=10):
    """Best set of k balanced two-arm trials from `pool` (exhaustive; split on a grid of `step` patients)."""
    best = None
    for combo in combinations(pool, k):
        if k == 1:
            splits = [(budget,)]
        else:
            splits = [(n, budget - n) for n in range(step, budget, step)]
        for s in splits:
            val = resistances(plan_vec(dict(zip(combo, s))))[0].mean()
            if best is None or val < best[0]:
                best = (val, dict(zip(combo, s)))
    return best[1]


rows = []
for budget in [200, 400]:
    plans = {
        "unrestricted optimum (bound)": dict(zip(cands, solve_mean(budget))),
        "best single trial": best_k(budget, cands, 1),
        "best single head-to-head trial": best_k(budget, h2h, 1),
        "best two trials": best_k(budget, cands, 2),
        "best two head-to-head trials": best_k(budget, h2h, 2),
        "eculizumab vs inebilizumab only": {("Eculizumab", "Inebilizumab"): budget},
        "equal split over the 10 head-to-head comparisons": {c: budget / 10 for c in h2h},
    }
    for name, spec in plans.items():
        N = plan_vec(spec)
        r = resistances(N)[0]
        alloc = {f"{u} vs {v}": round(float(x)) for (u, v), x in zip(cands, N) if x > 0.5}
        rows.append(dict(budget=budget, plan=name, allocation=alloc, mean_r=float(r.mean()), max_r=float(r.max()),
                         mean_reduction=float(1 - r.mean() / r0.mean()), max_reduction=float(1 - r.max() / r0.max()),
                         worst_target=out["targets"][int(np.argmax(r))]))
out["plans"] = rows
json.dump(out, open(os.path.join(OUT, "nmosd_allocation.json"), "w"), indent=1)

# LaTeX table for the supplement
L = [r"\begin{tabular}{rlp{6.4cm}rr}", r"\toprule",
     r"Budget & Plan & Trials funded (patients) & Mean $r$ & Largest $r$ \\", r"\midrule"]
for r in rows:
    al = "; ".join(f"{k} ({v})" for k, v in sorted(r["allocation"].items(), key=lambda z: -z[1]))
    if r["plan"].startswith("equal split"):
        al = f"all 10 head-to-head comparisons, {round(r['budget'] / 10)} patients each"
    if r["plan"].startswith("unrestricted"):
        al = f"{len(r['allocation'])} trials of {min(r['allocation'].values())}--{max(r['allocation'].values())} patients"
    pct = lambda x: "0\\%" if round(100 * x) == 0 else f"$-{100 * x:.0f}$\\%"
    L.append(f"{r['budget']} & {r['plan']} & {al} & {pct(r['mean_reduction'])} & {pct(r['max_reduction'])} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(OUT, "nmosd_allocation.tex"), "w").write("\n".join(L) + "\n")
for r in rows:
    print(r["budget"], r["plan"][:40].ljust(40), r["allocation"] if len(r["allocation"]) < 4 else f"{len(r['allocation'])} trials",
          f"mean -{100 * r['mean_reduction']:.0f}%  worst -{100 * r['max_reduction']:.0f}%  ({r['worst_target']})")
print("before: mean r", round(out["mean_before"], 4), "max r", round(out["max_before"], 4))
