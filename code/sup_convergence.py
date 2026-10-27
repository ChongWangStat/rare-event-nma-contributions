"""Convergence of the Berger-Boos supremum p-value for the standardized consistency
statistic T_W across search budgets and random seeds (Supplementary Table S4).

The supremum over the confidence set C_gamma is found by a stochastic search with common
random numbers.  A reader needs to know that the reported value is not an artefact of the
budget: here each network is searched at three budgets (B common random numbers x starts x
iterations) and, at the largest budget, with three seeds.  Because the search maximizes a Monte Carlo
estimate, the maximum is biased upward by simulation noise; the tail at the best point found is therefore
re-estimated with B_FRESH independent draws (its Monte Carlo SE is reported), and p_BB is reported from that
re-estimate.  The plug-in tail at the fitted null and the chi-square p-value are shown for reference.

Usage:  python3 sup_convergence.py NAME   (NAME in gurusamy, covid, dong, cf)
        python3 sup_convergence.py table  (collects ../results/supconv_*.json into a LaTeX table)
"""
import json
import sys
import time

import numpy as np
import pandas as pd

from scipy.stats import binom as _binom

from rdcsp import Network, berger_boos_sup_wald, inconsistency_projector, wald_inconsistency_stat

B_FRESH = 20000   # independent draws used to re-estimate the tail at the best point found

NETS = {
    "gurusamy": ("gurusamy2011_liver_transplant_mortality.csv", "Liver transplant"),
    "covid": ("covid19_wholenetwork.csv", "COVID-19"),
    "dong": ("dong2013_copd_mortality.csv", "COPD"),
    "cf": ("cf_sae_registry.csv", "Cystic fibrosis"),
}
BUDGETS = [(1000, 2, 20), (2000, 4, 40), (4000, 8, 80)]
SEEDS_LARGEST = [11, 23, 37]
OUT = "../results"


def load(name):
    df = pd.read_csv(f"../data/{NETS[name][0]}")
    return Network(df[["study", "treatment", "events", "n"]])


def fresh_tail(net, p, t_obs, seed):
    """Tail probability P_p(T_W >= t_obs) re-estimated with independent draws, and its MC SE."""
    P, Bmat, sw = inconsistency_projector(net)
    Xc = net.design_consistency()
    Xw = sw[:, None] * Xc
    pinv = np.linalg.pinv(Xw.T @ Xw) @ Xw.T
    rng = np.random.default_rng(seed)
    hits = 0
    for chunk in range(B_FRESH // 5000):
        U = rng.uniform(size=(5000, len(net.n)))
        R = _binom.ppf(U, net.n.astype(int)[None, :], np.clip(p, 1e-12, 1 - 1e-12)[None, :])
        hits += int(np.sum(wald_inconsistency_stat(net, R / net.n, P, Bmat, sw, pinv, Xc) >= t_obs))
    tail = hits / B_FRESH
    return tail, float(np.sqrt(max(tail * (1 - tail), 1e-12) / B_FRESH))


def run(name):
    """Resumable: rows already in ../results/supconv_NAME.json are kept and skipped."""
    import os
    net = load(name)
    f = f"{OUT}/supconv_{name}.json"
    rows = json.load(open(f)) if os.path.exists(f) else []
    done = {(r["B"], r["seed"]) for r in rows}
    for B, starts, iters in BUDGETS:
        seeds = SEEDS_LARGEST if (B, starts, iters) == BUDGETS[-1] else [11]
        for seed in seeds:
            if (B, seed) in done:
                continue
            t0 = time.time()
            r = berger_boos_sup_wald(net, B=B, n_starts=starts, n_iter=iters, seed=seed)
            ft, fse = fresh_tail(net, r["best_p"], r["t_obs"], seed=1000 + seed)
            rows.append(dict(network=name, B=B, starts=starts, iters=iters, seed=seed,
                             t_obs=r["t_obs"], df=r["df"], chi2_p=r["chi2_p"],
                             tail_fitted=r["p_at_fitted_null"], sup_tail_search=r["sup_tail"],
                             p_bb_search=r["p_value"], tail_fresh=ft, tail_fresh_se=fse,
                             p_bb=ft + 0.001, seconds=time.time() - t0))
            print(rows[-1], flush=True)
            json.dump(rows, open(f, "w"), indent=1)
    rows.sort(key=lambda r: (r["B"], r["seed"]))
    json.dump(rows, open(f, "w"), indent=1)


def table():
    lines = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
             r"Network & Budget ($B$, starts$\times$iterations) & Seed & $T_W$ (df) & Tail at fitted null & Largest tail in search & $p_{\mathrm{BB}}$ (SE) & $\chi^2$ p \\",
             r"\midrule"]
    import os
    avail = [n for n in NETS if os.path.exists(f"{OUT}/supconv_{n}.json")]
    for name in avail:
        rows = json.load(open(f"{OUT}/supconv_{name}.json"))
        for i, r in enumerate(rows):
            lab = NETS[name][1] if i == 0 else ""
            tw = f"{r['t_obs']:.2f} ({r['df']})" if i == 0 else ""
            chi = f"{r['chi2_p']:.3f}" if i == 0 else ""
            lines.append(f"{lab} & {r['B']}, {r['starts']}$\\times${r['iters']} & {r['seed']} & {tw} & "
                         f"{r['tail_fitted']:.3f} & {r['sup_tail_search']:.3f} & {r['p_bb']:.3f} ({r['tail_fresh_se']:.3f}) & {chi} \\\\")
        lines.append(r"\addlinespace")
    lines[-1] = r"\bottomrule"
    lines.append("\\end{tabular}")
    open(f"{OUT}/supconv.tex", "w").write("\n".join(lines) + "\n")
    summ = {}
    for name in avail:
        rows = json.load(open(f"{OUT}/supconv_{name}.json"))
        p = [r["p_bb"] for r in rows]
        big = [r["p_bb"] for r in rows if r["B"] == BUDGETS[-1][0]]
        best = max([r for r in rows if r["B"] == BUDGETS[-1][0]], key=lambda r: r["p_bb"])
        summ[name] = dict(p_min=min(p), p_max=max(p), largest_budget_range=[min(big), max(big)],
                          reported=best["p_bb"], reported_se=best["tail_fresh_se"], t_obs=best["t_obs"],
                          df=best["df"], chi2_p=best["chi2_p"], tail_fitted=best["tail_fitted"])
    json.dump(summ, open(f"{OUT}/supconv_summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "table":
        table()
    else:
        run(sys.argv[1])
