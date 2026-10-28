"""Check of the constrained binomial maximum-likelihood null fit (Supplement S3): the SLSQP solution used by
rdcsp.constrained_null_mle is compared with an interior-point solution (logarithmic barrier, damped Newton)
of the same concave problem on the four networks with loops and on 200 simulated data sets from each null
setting of the synthetic simulation (risks 1-4% and 0.2-1.2%).  SLSQP often reports a boundary termination
(status 8) because optimal fitted risks of zero-event arms sit on the constraint 0 <= X beta.
Usage: python3 check_constrained_mle.py"""
import numpy as np
import pandas as pd

from rdcsp import Network, constrained_null_mle

from rdcsp import _barrier_mle, _nll_p


def barrier_mle(net, r):
    pb = _barrier_mle(net.design_consistency(), r, net.n)
    return pb, _nll_p(pb, r, net.n)


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


# simulated data sets from the null settings of the synthetic simulation
from run_simulations import make_design, true_probs
net = Network(make_design(1))
X, nint = net.design_consistency(), net.n.astype(int)
for rate, (lo_, hi_), mu_ in [("low", (0.01, 0.04), {"A": 0.0, "B": -0.008, "C": 0.0, "D": 0.012}),
                              ("verylow", (0.002, 0.012), {"A": 0.0, "B": -0.0015, "C": 0.0, "D": 0.004})]:
    rr = np.random.default_rng(10 if rate == "low" else 11)
    alpha = {s_: rr.uniform(lo_, hi_) for s_ in net.studies}
    ptrue = true_probs(net, alpha, mu_)
    rng = np.random.default_rng(500)
    dmax, llmax, ok, fb = 0.0, 0.0, 0, 0
    for i in range(200):
        r = rng.binomial(nint, ptrue).astype(float)
        p1, res = constrained_null_mle(net, y=r / net.n)
        ok += bool(res.success)
        fb += bool(res.fallback)
        pb, llb = barrier_mle(net, r)
        p1c = np.clip(p1, 1e-300, 1 - 1e-16)
        ll1 = -np.sum(r * np.log(p1c) + (net.n - r) * np.log1p(-p1c))
        dmax = max(dmax, np.abs(pb - p1).max())
        llmax = max(llmax, ll1 - llb)
    print(f"synthetic {rate:8s}: SLSQP reports success in {ok}/200, interior-point fallback used in {fb}/200; "
          f"max |fitted risk difference| {dmax:.2e}; max excess -loglik {llmax:.2e}")
