"""Precision shares eta_k for the COPD comparison placebo vs tiotropium HandiHaler (Dong et al. 2013).

eta_k = (sum over study-k arms of c_{k,arm}^2 / n_arm) / r_ab, where c are the arm coefficients of
the linear estimator and r_ab = e' L^+ e; the shares sum to one.  The table lists the eight
studies with the largest shares, with direct weight d_k and throughput tau_k (= W) alongside.
Also prints the check eta_k = d_k for two-arm direct studies, and the plug-in (n) versus
unbiased (n-1) standard errors for the two headline mortality targets.

  -> results/dong_precision_shares.tex

Usage:  python3 dong_precision_shares.py [OUTDIR]     (OUTDIR defaults to ../results)
Run time: a few seconds.
"""
import os
import sys

import numpy as np
import pandas as pd

from rdcsp import Network

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "results")
os.makedirs(OUT, exist_ok=True)


def shares(net, a, b):
    e = net.e(a, b)
    R = float(e @ net.Lp @ e)
    cur = net.currents(a, b)
    res = net.contributions(a, b)
    rows = []
    for k in net.studies:
        idx = net.rows[k]
        c = -cur[k]
        E = float((c ** 2 / net.n[idx]).sum())
        rows.append((k, E / R))
    s = pd.DataFrame(rows, columns=['study', 'eta'])
    return res.merge(s, on='study'), R


if __name__ == "__main__":
    dong = Network(pd.read_csv(os.path.join(DATA, 'dong2013_copd_mortality.csv')))
    res, R = shares(dong, 'Placebo', 'TIO-HH')
    print('sum eta', res.eta.sum(), 'r_ab', R)
    top = res.sort_values('eta', ascending=False).head(8)
    lines = ["\\begin{tabular}{llrrr}", "\\toprule", "Study & Arms & $d_k$ & $\\tau_k$ & $\\eta_k$ \\\\", "\\midrule"]
    for _, r in top.iterrows():
        lines.append(f"{r['study']} & {r['arms']} & {r['d']:.3f} & {r['W']:.3f} & {r['eta']:.3f} \\\\")
    rest = res.drop(top.index)
    lines.append(f"Other {len(rest)} studies & & {rest['d'].sum():.3f} & {rest['W'].sum():.3f} & {rest['eta'].sum():.3f} \\\\")
    lines += ["\\midrule", f"Sum & & {res['d'].sum():.3f} & {res['W'].sum():.3f} & {res['eta'].sum():.3f} \\\\",
              "\\bottomrule", "\\end{tabular}"]
    open(os.path.join(OUT, 'dong_precision_shares.tex'), 'w').write("\n".join(lines) + "\n")
    print(top[['study', 'arms', 'd', 'W', 'eta']].round(3).to_string())
    # two-arm direct studies: eta_k equals d_k
    two = res[(res.d > 0) & (res.arms.str.count('/') == 1)]
    print('max |eta-d| two-arm direct:', (two.eta - two.d).abs().max())
    # unbiased variance (n-1) vs plug-in (n) for the two headline targets
    gur = Network(pd.read_csv(os.path.join(DATA, 'gurusamy2011_liver_transplant_mortality.csv')))
    for net, a, b in [(dong, 'Placebo', 'TIO-HH'), (gur, 'Control/Placebo', 'Aprotinin')]:
        c = net.coefficient_vector(a, b)
        v1 = (c ** 2 * net.phat * (1 - net.phat) / net.n).sum()
        v2 = (c ** 2 * net.phat * (1 - net.phat) / (net.n - 1)).sum()
        print(a, b, 'SE plug-in n', np.sqrt(v1).round(5), 'SE unbiased n-1', np.sqrt(v2).round(5))
    print("wrote", os.path.abspath(os.path.join(OUT, 'dong_precision_shares.tex')))
