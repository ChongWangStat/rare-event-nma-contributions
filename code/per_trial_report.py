"""Per-trial report for one comparison of a network meta-analysis of a binary outcome (the reporting
recommendation of the Discussion).

Input: a CSV with columns study, treatment, events, n (one row per arm; arm sizes as analysed).
Output (printed and written to CSV): for every study, the direct weight d_k, the throughput tau_k, the
precision share eta_k, the direct and indirect contributions C^D_k and C^I_k, the contribution C_k with its
exact plug-in standard error (n-1 denominators), and the estimate with its exact SE, the MOVER-Wilson interval
and the nuisance-free e-value interval.  Contributions sum to the estimate exactly; d, tau and eta depend on
the arm sizes only.  Externally controlled single-arm trials need the shared-baseline model and are not
handled here; the network must be connected.

Usage: python3 per_trial_report.py data.csv TREATMENT_A TREATMENT_B [out.csv]
       (the target is the risk under B minus the risk under A)
Example: python3 per_trial_report.py ../data/dong2013_copd_mortality.csv Placebo TIO-HH
"""
import sys

import numpy as np
import pandas as pd

from rdcsp import Network, mover_interval, emixture_interval

if len(sys.argv) < 4:
    sys.exit(__doc__)
path, a, b = sys.argv[1:4]
dest = sys.argv[4] if len(sys.argv) > 4 else None
df = pd.read_csv(path)[["study", "treatment", "events", "n"]]
net = Network(df)
c = net.coefficient_vector(a, b)
p = net.r / net.n
r_ab = float(np.sum(c ** 2 / net.n))
res = net.contributions(a, b)
eta, se = [], []
for k in res.study:
    idx = net.rows[k]
    eta.append(float(np.sum(c[idx] ** 2 / net.n[idx]) / r_ab))
    se.append(float(np.sqrt(np.sum(c[idx] ** 2 * p[idx] * (1 - p[idx]) / np.maximum(net.n[idx] - 1, 1)))))
rep = pd.DataFrame(dict(study=res.study, arms=res.arms, d=res.d, tau=res.W, eta=eta,
                        C_direct=res.C_D, C_indirect=res.C_I, C=res.C, SE_C=se))
rep = rep.sort_values("tau", ascending=False)
est = float(c @ p)
se_est = float(np.sqrt(np.sum(c ** 2 * p * (1 - p) / np.maximum(net.n - 1, 1))))
mv, ev = mover_interval(c, net.r, net.n), emixture_interval(c, net.r, net.n)
pd.set_option("display.width", 160)
print(rep.round(4).to_string(index=False))
print(f"\n{b} minus {a}: estimate {est:.4f} (exact SE {se_est:.4f}); sum of contributions {rep.C.sum():.4f}")
print(f"direct mass {res.d.sum():.3f}, indirect mass {res.attrs['q']:.3f}")
print(f"95% MOVER-Wilson interval [{mv[0]:.4f}, {mv[1]:.4f}]; 95% e-value interval (valid for every risk vector) "
      f"[{ev[0]:.4f}, {ev[1]:.4f}]")
if dest:
    rep.to_csv(dest, index=False)
