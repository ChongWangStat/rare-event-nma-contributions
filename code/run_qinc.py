"""Consistency testing with Q_inc: plug-in binomial calibration, exact two-moment
(Satterthwaite) reference, Berger-Boos exact unconditional supremum p-value,
Hoeffding distribution-free test, and the fiducial predictive p-value.
Part 1: the three networks.  Part 2: simulated levels and power.
Writes ../results/qinc_networks.json, ../results/qinc_sim.csv"""
import json, time, sys
import numpy as np, pandas as pd
from rdcsp import (Network, q_decomposition, null_fit_probabilities, inconsistency_projector, qinc_from_P,
                   qinc_exact_moments, satterthwaite_p, berger_boos_sup_p, hoeffding_inconsistency_test,
                   fiducial_predictive_p, wls_fit, wald_inconsistency_stat, berger_boos_sup_wald)
from rdcsp import constrained_null_mle
from scipy.stats import chi2 as _chi2
from run_simulations import make_design, true_probs

OUT = "../results"
t0 = time.time()
res = {}
NETS = [] if len(sys.argv) > 1 else [("gurusamy", "gurusamy2011_liver_transplant_mortality"), ("covid", "covid19_wholenetwork"),
                                     ("cf", "cf_sae_registry"), ("dong", "dong2013_copd_mortality")]
for name, f in NETS:  # [("gurusamy", "gurusamy2011_liver_transplant_mortality"), ("covid", "covid19_wholenetwork"), ("dong", "dong2013_copd_mortality")]:
    net = Network(pd.read_csv(f"../data/{f}.csv"))
    P, Bm, sw = inconsistency_projector(net)
    q = q_decomposition(net)
    # null generator: constrained binomial maximum likelihood under the consistency model (fitted risks in
    # the null mean space and in [0, 1]); the T_W supremum is computed in sup_convergence.py
    p0 = constrained_null_mle(net, "consistency")[0]
    rng = np.random.default_rng(5)
    Y = rng.binomial(net.n.astype(int), p0, size=(20000, len(p0))) / net.n
    qs = qinc_from_P(P, sw, Y)
    mean, var = qinc_exact_moments(net, p0, P, sw)
    sp, c, nu = satterthwaite_p(q["Q_inc"], mean, var)
    hoef = hoeffding_inconsistency_test(net)
    fid = fiducial_predictive_p(net, B=4000)
    bb = berger_boos_sup_p(net, B=2000 if name != "dong" else 1000, n_starts=6 if name != "dong" else 3,
                           n_iter=60 if name != "dong" else 40, seed=11)
    Tw = wald_inconsistency_stat(net, Y)
    t_obs = float(wald_inconsistency_stat(net, net.phat[None, :])[0])
    wald_plugin_p = float(np.mean(Tw >= t_obs))
    res[name] = dict(Q_inc=q["Q_inc"], df_inc=int(q["df_inc"]), plugin_p=float(np.mean(qs >= q["Q_inc"])),
                     exact_mean=mean, exact_var=var, satterthwaite_p=sp, satt_c=c, satt_nu=nu,
                     hoeffding_p=hoef["p_value"], hoeffding_components=hoef["p_components"].tolist(),
                     fiducial_p=fid["p_value"], bb_p=bb["p_value"], bb_sup_tail=bb["sup_tail"],
                     bb_tail_at_fitted=bb["p_at_fitted_null"],
                     wald_T=t_obs, wald_chi2_p=float(_chi2.sf(t_obs, Bm.shape[1])),
                     wald_plugin_p=wald_plugin_p, null_generator="constrained MLE", time_s=time.time() - t0)
    print(name, {k: v for k, v in res[name].items() if k != "hoeffding_components"}, flush=True)
if NETS:
    json.dump(res, open(f"{OUT}/qinc_networks.json", "w"), indent=1, default=float)

# ---------------- Part 2: simulated level and power ---------------------------------- #
def run_cell(rate, delta, reps, with_bb, seed):
    df = make_design(1)
    net = Network(df)
    nint = net.n.astype(int)
    r = np.random.default_rng(seed)
    lo, hi = (0.01, 0.04) if rate == "low" else (0.002, 0.012)
    alpha = {s: r.uniform(lo, hi) for s in net.studies}
    mu = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}
    p = true_probs(net, alpha, mu).copy()
    s8 = net.rows["s8"]
    p[s8[1]] += delta
    P, Bm, sw = inconsistency_projector(net)
    Xc = net.design_consistency()
    rej = dict(plugin=0, satterthwaite=0, hoeffding=0, fiducial=0, bb=0, wald_plugin=0, wald_chi2=0, wald_bb=0)
    nbb = 0
    from scipy.stats import chi2 as _c2
    dfi = Bm.shape[1]
    for i in range(reps):
        y = r.binomial(nint, p) / net.n
        net.r, net.phat = y * net.n, y
        qobs = float(qinc_from_P(P, sw, y)[0])
        fit = wls_fit(Xc, y, net.w)
        p0 = np.minimum(np.maximum(fit, 0.5 / net.n), 1.0)
        Yb = r.binomial(nint, p0, size=(400, len(y))) / net.n
        rej["plugin"] += np.mean(qinc_from_P(P, sw, Yb) >= qobs) < 0.05
        mean, var = qinc_exact_moments(net, p0, P, sw)
        rej["satterthwaite"] += satterthwaite_p(qobs, mean, var)[0] < 0.05
        rej["hoeffding"] += hoeffding_inconsistency_test(net)["p_value"] < 0.05
        rej["fiducial"] += fiducial_predictive_p(net, B=300, seed=i)["p_value"] < 0.05
        tobs = float(wald_inconsistency_stat(net, y, P, Bm, sw)[0])
        rej["wald_plugin"] += np.mean(wald_inconsistency_stat(net, Yb, P, Bm, sw) >= tobs) < 0.05
        rej["wald_chi2"] += _c2.sf(tobs, dfi) < 0.05
        if with_bb and i < with_bb:
            bb = berger_boos_sup_p(net, q_obs=qobs, B=800, n_starts=2, n_iter=20, seed=i)
            rej["bb"] += bb["p_value"] < 0.05
            bbw = berger_boos_sup_wald(net, B=800, n_starts=2, n_iter=20, seed=i)
            rej["wald_bb"] += bbw["p_value"] < 0.05
            nbb += 1
    out = {k: v / reps for k, v in rej.items() if k not in ("bb", "wald_bb")}
    out["bb"] = rej["bb"] / nbb if nbb else np.nan
    out["wald_bb"] = rej["wald_bb"] / nbb if nbb else np.nan
    out.update(rate=rate, delta=delta, reps=reps, reps_bb=nbb)
    return out

if len(sys.argv) > 1:
    rate, delta = sys.argv[1], float(sys.argv[2])
    row = run_cell(rate, delta, reps=1000, with_bb=120 if delta == 0.0 else 60, seed=100)
    print(row, round(time.time() - t0), "s", flush=True)
    import os
    f = f"{OUT}/qinc_sim.csv"
    df = pd.read_csv(f) if os.path.exists(f) else pd.DataFrame()
    df = pd.concat([df[(df.get("rate") != rate) | (df.get("delta") != delta)] if len(df) else df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(f, index=False)
