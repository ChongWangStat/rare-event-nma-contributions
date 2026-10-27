"""Check of the constrained binomial maximum-likelihood null fit (Supplement S3): the SLSQP solution used by
rdcsp.constrained_null_mle is compared with an interior-point solution (logarithmic barrier, damped Newton)
of the same concave problem on the four networks with loops.  SLSQP often reports a boundary termination
(status 8) because optimal fitted risks of zero-event arms sit on the constraint 0 <= X beta.
Usage: python3 check_constrained_mle.py"""
import numpy as np
import pandas as pd

from rdcsp import Network, constrained_null_mle

for f in ["dong2013_copd_mortality.csv", "gurusamy2011_liver_transplant_mortality.csv", "cf_sae_registry.csv",
          "covid19_wholenetwork.csv"]:
    net = Network(pd.read_csv("../data/" + f)[["study", "treatment", "events", "n"]])
    X, r, n = net.design_consistency(), net.r, net.n

    def nll(b):
        p = np.clip(X @ b, 1e-300, 1 - 1e-16)
        return -np.sum(r * np.log(p) + (n - r) * np.log1p(-p))

    p1, res = constrained_null_mle(net)
    b1 = np.linalg.lstsq(X, p1, rcond=None)[0]
    # strictly feasible start
    pt = np.clip(net.phat, 0.002, 0.998)
    b = np.linalg.lstsq(X, pt, rcond=None)[0]
    for _ in range(50):
        p = X @ b
        if ((p > 0) & (p < 1)).all():
            break
        pt = 0.5 * pt + 0.5 * net.phat.mean()
        b = np.linalg.lstsq(X, pt, rcond=None)[0]

    def fbar(b, mu):
        p = X @ b
        if (p <= 0).any() or (p >= 1).any():
            return np.inf
        return nll(b) - mu * np.sum(np.log(p) + np.log1p(-p))

    mu = 1.0
    while mu > 1e-12:
        for _ in range(100):
            p = X @ b
            g = -X.T @ (r / p - (n - r) / (1 - p)) - mu * X.T @ (1 / p - 1 / (1 - p))
            h = r / p ** 2 + (n - r) / (1 - p) ** 2 + mu * (1 / p ** 2 + 1 / (1 - p) ** 2)
            step = np.linalg.lstsq(X.T @ (h[:, None] * X), -g, rcond=None)[0]
            s, f0 = 1.0, fbar(b, mu)
            while fbar(b + s * step, mu) > f0 + 1e-4 * s * (g @ step) and s > 1e-12:
                s *= 0.5
            b = b + s * step
            if abs(g @ step) < 1e-12:
                break
        mu *= 0.3
    print(f"{f:48s} SLSQP status {res.status}  -loglik SLSQP {nll(b1):.4f}  barrier {nll(b):.4f}  "
          f"max |fitted risk difference| {np.abs(X @ b - p1).max():.2e}")
