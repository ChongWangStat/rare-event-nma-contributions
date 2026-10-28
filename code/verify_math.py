"""Numerical checks for the propositions in the mathematical note on rare-event
study contributions (risk-difference scale, exact binomial arms).
Run from code/: python3 verify_math.py  -> ../results/verify_math.json"""
import json
import numpy as np, pandas as pd
from itertools import combinations
from rdcsp import Network, add_planned_trial, q_decomposition, null_expectation_Q

out = {}
rng = np.random.default_rng(2026)

dong = Network(pd.read_csv("../data/dong2013_copd_mortality.csv"))
gur = Network(pd.read_csv("../data/gurusamy2011_liver_transplant_mortality.csv"))
cov = Network(pd.read_csv("../data/covid19_wholenetwork.csv"))

# 1. Reduced (basic contrasts) and full (all pairwise contrasts, V^+) representations agree
def full_contrast_gls(net, a, b):
    rows, Vblocks, ys = [], [], []
    for k in net.studies:
        idx = net.rows[k]
        m = len(idx)
        if m < 2:
            continue
        pairs = list(combinations(range(m), 2))
        C = np.zeros((len(pairs), m))
        for r, (i, j) in enumerate(pairs):
            C[r, j], C[r, i] = 1.0, -1.0
        S = np.diag(1.0 / net.w[idx])
        Vblocks.append(C @ S @ C.T)
        ys.append(C @ net.phat[idx])
        X = np.zeros((len(pairs), net.T))
        for r, (i, j) in enumerate(pairs):
            X[r, net.tix[net.arm_trt[idx[j]]]] += 1
            X[r, net.tix[net.arm_trt[idx[i]]]] -= 1
        rows.append(X)
    X = np.vstack(rows)[:, 1:]
    y = np.concatenate(ys)
    Vp = np.zeros((len(y), len(y)))
    pos = 0
    for Vk in Vblocks:
        m = Vk.shape[0]
        Vp[pos:pos + m, pos:pos + m] = np.linalg.pinv(Vk)
        pos += m
    beta = np.linalg.pinv(X.T @ Vp @ X) @ X.T @ Vp @ y
    mu = np.concatenate([[0.0], beta])
    return mu[net.tix[b]] - mu[net.tix[a]]

out["full_vs_reduced_max_diff"] = max(abs(full_contrast_gls(n_, a, b) - n_.estimate(a, b))
                                      for n_, a, b in [(dong, "Placebo", "TIO-HH"), (gur, "Control/Placebo", "Aprotinin"), (cov, "A", "E")])

# 2. Maximum principle and closed-form direct weights
def check_closed_forms(net, a, b):
    e = net.e(a, b)
    phi = net.Lp @ e
    R = float(e @ phi)
    res = net.contributions(a, b)
    rec = dict(R_ab=R, source_attains_max=bool(phi[net.tix[a]] >= phi.max() - 1e-12), sink_attains_min=bool(phi[net.tix[b]] <= phi.min() + 1e-12))
    two_arm_err, lower_bound_ok, sum_g = 0.0, True, 0.0
    for k in net.studies:
        idx = net.rows[k]
        arms = list(net.arm_trt[idx])
        if a in arms and b in arms:
            w = net.w[idx]
            ia, ib = arms.index(a), arms.index(b)
            g_ab = w[ia] * w[ib] / w.sum()
            d = float(res.loc[res["study"] == k, "d"].iloc[0])
            sum_g += g_ab
            if len(idx) == 2:
                two_arm_err = max(two_arm_err, abs(d - g_ab * R))
            else:
                lower_bound_ok &= d >= g_ab * R - 1e-12
    rec.update(two_arm_closed_form_max_err=two_arm_err, multiarm_lower_bound_holds=bool(lower_bound_ok),
               direct_mass=float(res["d"].sum()), lower_bound_direct_mass=R * sum_g)
    return rec

out["closed_forms"] = {f"{a}:{b}": check_closed_forms(n_, a, b) for n_, a, b in
                       [(dong, "Placebo", "TIO-HH"), (dong, "LABA", "TIO-HH"), (gur, "Control/Placebo", "Aprotinin"), (cov, "A", "E"), (cov, "B", "E")]}

# 3. Rank-one design-stage update (Theorem on planned trials)
a2, b2 = "Placebo", "TIO-SMI"
e2 = dong.e(a2, b2)
G_old = 1.0 / float(e2 @ dong.Lp @ e2)
ds = []
for n in [200, 500, 1000, 2000]:
    new = add_planned_trial(dong, {a2: n, b2: n})
    c = new.contributions(a2, b2)
    d_num = float(c.loc[c["study"] == "PLANNED", "d"].iloc[0])
    g = n / 2.0
    ds.append(dict(n=n, numeric=d_num, closed_form=g / (g + G_old)))
# throughput of the planned trial for another target via the Woodbury formula
a3, b3 = "Placebo", "TIO-HH"
e3 = dong.e(a3, b3)
dvec = dong.e(a2, b2)
g = 1000.0
R_uv = float(dvec @ dong.Lp @ dvec)
W_formula = g * abs(float(dvec @ dong.Lp @ e3)) / (1 + g * R_uv)
new = add_planned_trial(dong, {a2: 2000, b2: 2000})
c3 = new.contributions(a3, b3)
W_num = float(c3.loc[c3["study"] == "PLANNED", "W"].iloc[0])
out["design_stage"] = dict(direct_weight=ds, throughput_other_target_formula=W_formula, throughput_other_target_numeric=W_num)
# updated potentials of existing studies after adding the trial
phi_old = dong.Lp @ e3
phi_new_formula = phi_old - g * (dong.Lp @ dvec) * float(dvec @ phi_old) / (1 + g * R_uv)
phi_new_numeric = new.Lp @ new.e(a3, b3)
out["design_stage"]["potential_update_max_err"] = float(np.abs(phi_new_formula - phi_new_numeric).max())

# 4. Exact unbiasedness under non-normal random effects on the risk difference
c = gur.coefficient_vector("Control/Placebo", "Aprotinin")
mu = {t: 0.0 for t in gur.trts}
mu["Aprotinin"] = -0.03
p0 = np.array([0.08 + mu[t] for t in gur.arm_trt])
tau = 0.01
ests = []
nint = gur.n.astype(int)
for _ in range(200000):
    dev = (rng.exponential(1.0, size=len(p0)) - 1.0) * tau / np.sqrt(2)   # skewed, mean zero
    p = np.clip(p0 + dev, 1e-4, 1 - 1e-4)
    ests.append(c @ (rng.binomial(nint, p) / gur.n))
ests = np.array(ests)
out["random_effects"] = dict(true=-0.03, mean_estimate=float(ests.mean()), mc_se=float(ests.std() / np.sqrt(len(ests))),
                             var_formula=float((c ** 2 * (p0 * (1 - p0) - tau ** 2 / 2) / gur.n).sum() + tau ** 2 / 2 * (c ** 2).sum()),
                             var_empirical=float(ests.var()))

# 5. Hoeffding finite-sample interval (exact, distribution-free) on the sparse network
cover, width = 0, []
half = np.sqrt((c ** 2 / gur.n).sum() * np.log(2 / 0.05) / 2)
for _ in range(20000):
    est = c @ (rng.binomial(nint, p0) / gur.n)
    cover += abs(est + 0.03) <= half
out["hoeffding"] = dict(coverage=cover / 20000, halfwidth=float(half), exact_sd=float(np.sqrt((c ** 2 * p0 * (1 - p0) / gur.n).sum())))

# 6. Efficiency bound (Kantorovich) for design-fixed weights vs optimal inverse-variance weights
def efficiency(net, a, b, p):
    e = net.e(a, b)
    cw = net.coefficient_vector(a, b)
    var_w = float((cw ** 2 * p * (1 - p) / net.n).sum())
    opt = Network(net.df, weights=net.n / (p * (1 - p)))
    copt = opt.coefficient_vector(a, b)
    var_opt = float((copt ** 2 * p * (1 - p) / net.n).sum())
    lam = p * (1 - p)
    bound = 4 * lam.min() * lam.max() / (lam.min() + lam.max()) ** 2
    return dict(var_w=var_w, var_opt=var_opt, efficiency=var_opt / var_w, kantorovich_bound=bound)
p_d = np.clip(q_decomposition(dong)["fit_c"], 0.5 / dong.n, 1)
out["efficiency_dong_placebo_tiohh"] = efficiency(dong, "Placebo", "TIO-HH", p_d)

# 7. Q identities and exact null expectations (design-fixed weights)
for name, net in [("dong", dong), ("gur", gur), ("covid", cov)]:
    q = q_decomposition(net)
    p0n = np.clip(q["fit_c"], 0.5 / net.n, 1)
    ex = null_expectation_Q(net, p0n)
    B = 20000
    draws = np.empty((B, 3))
    for b_ in range(B):
        yb = rng.binomial(net.n.astype(int), p0n) / net.n
        qb = q_decomposition(net, yb)
        draws[b_] = (qb["Q_net"], qb["Q_het"], qb["Q_inc"])
    out[f"Q_{name}"] = dict(additivity=q["Q_net"] - q["Q_het"] - q["Q_inc"], E0=ex, mc_mean=dict(zip(["net", "het", "inc"], draws.mean(0).tolist())))

# 8. Algebraic identities for every target of every network (rdcsp.verify_identities)
from rdcsp import verify_identities
nets = {"COPD": dong, "liver": gur, "COVID-19": cov,
        "cystic fibrosis": Network(pd.read_csv("../data/cf_sae_registry.csv")[["study", "treatment", "events", "n"]]),
        "NMOSD": Network(pd.read_csv("../data/nmosd2025_relapse.csv").query("study != 'CHAMPION'")[["study", "treatment", "events", "n"]]),
        "SMA safety": Network(pd.read_csv("../data/sma_nusinersen_pneumonia.csv")[["study", "treatment", "events", "n"]])}
worst, n_targets = {}, 0
for name, net_ in nets.items():
    for a_, b_ in combinations(net_.trts, 2):
        n_targets += 1
        for key, val in verify_identities(net_, a_, b_).items():
            worst[key] = max(worst.get(key, 0.0), abs(float(val)))
out["identities_all_targets"] = dict(n_targets=n_targets, max_abs=worst)

json.dump(out, open("../results/verify_math.json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
