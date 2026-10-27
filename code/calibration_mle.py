"""Q calibration with the constrained binomial maximum-likelihood null generator on the three
mortality/AE networks (COPD, liver transplant, COVID-19): plug-in p-values for Q_net, Q_inc and
Q_het, exact null expectations at the constrained MLE, convergence flag, smallest fitted risk,
the noncentral term of Q_inc at the fitted null (zero up to rounding) and the Monte Carlo means.
Quoted in the COPD application ("At the constrained maximum-likelihood null the exact
expectations are 1.130, 0.896 and 0.234, and the plug-in binomial p-values are 0.10, 0.12 and 0.15").

  -> results/calibration_mle.json

Usage:  python3 calibration_mle.py [OUTDIR]     (OUTDIR defaults to ../results)
Run time: a few minutes (12,000 COPD and 20,000 other binomial draws).
"""
import json
import os
import sys

import pandas as pd

from rdcsp import Network, binomial_calibration_mle, null_expectation_Q, inconsistency_projector

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "results")
os.makedirs(OUT, exist_ok=True)

if __name__ == "__main__":
    out = {}
    for nm, f in [('dong', 'dong2013_copd_mortality'), ('gur', 'gurusamy2011_liver_transplant_mortality'),
                  ('covid', 'covid19_wholenetwork')]:
        net = Network(pd.read_csv(os.path.join(DATA, f'{f}.csv')))
        cal = binomial_calibration_mle(net, B=20000 if nm != 'dong' else 12000, seed=5)
        ex = null_expectation_Q(net, cal['p_null'])
        P, Bm, sw = inconsistency_projector(net)
        m = P @ (sw * cal['p_null'])
        out[nm] = dict(p_net=cal['p_net'], p_inc=cal['p_inc'], p_het=cal['p_het'], E0_net=ex['E_Q_net'],
                       E0_inc=ex['E_Q_inc'], E0_het=ex['E_Q_het'], converged=cal['converged'],
                       min_p0=float(cal['p_null'].min()), noncentral_inc=float(m @ m),
                       mc_mean=cal['draws'].mean(0).tolist())
        print(nm, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in out[nm].items()}, flush=True)
    json.dump(out, open(os.path.join(OUT, 'calibration_mle.json'), 'w'), indent=1)
