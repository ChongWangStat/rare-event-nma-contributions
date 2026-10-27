"""Analyses added in the 27 October 2026 revision (numbers quoted in Sections 3.7, 5.1, 5.3, 5.4).

  1. Leave-one-out importance on the design scale (variance proportional to sum c^2/n, i.e. the
     effective resistance r_ab), against the direct weights               -> loo_design
  2. Effective number of direct trials, (sum d)^2 / sum d^2                -> effective_trials
  3. Allocation of a future trial (COPD placebo vs TIO-HH, N = 1000; NMOSD eculizumab vs
     inebilizumab, N = 200)                                               -> allocation
  4. Design-based precision of the network relative to the best single trial,
     r_single / r_ab with r_single = 1/n_a + 1/n_b                       -> design_precision
  5. Efficiency of arm-size weights against the oracle weights n/{p(1-p)} at the
     constrained-MLE fitted risks (floored at 0.001)                      -> efficiency
  6. Binomial maximum likelihood under the additive model (constrained) against the
     design-fixed estimator, COPD placebo comparisons                     -> mle_comparison
  7. TIOSPIR: the network estimate of TIO-SMI vs TIO-HH, the design-stage weight TIOSPIR would
     have received (Theorem 5) and the estimate after adding it            -> tiospir
  8. Cystic fibrosis: EXPAND (crossover) excluded, and exposure-adjusted analysis with
     person-time weights                                                  -> cf_sensitivity
  9. Registry: anticipated/realized enrolment and weight agreement on the non-trivial rows
                                                                           -> registry
 10. Significance and weight: rank correlations and shares                -> q2

Usage: python3 revision_analyses.py [OUTDIR]   -> OUTDIR/revision_analyses.json (default ../results)
Run time: about one minute.
"""
import json
import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from rdcsp import (Network, add_planned_trial, constrained_null_mle, q_decomposition,
                   wald_inconsistency_stat, inconsistency_projector, study_significance,
                   berger_boos_sup_wald, binomial_interval)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "results")
R = {}


def load(f, drop=None):
    df = pd.read_csv(os.path.join(DATA, f))[["study", "treatment", "events", "n"]]
    if drop:
        df = df[~df.study.isin(drop)]
    return Network(df)


def rab(net, a, b):
    e = net.e(a, b)
    return float(e @ net.Lp @ e)


dong = load("dong2013_copd_mortality.csv")
cf = load("cf_sae_registry.csv")
nm = load("nmosd2025_relapse.csv", ["CHAMPION"])
gur = load("gurusamy2011_liver_transplant_mortality.csv")

# ---------------------------------------------------------------- 1. LOO on the design scale
a, b = "Placebo", "TIO-HH"
r_full = rab(dong, a, b)
res = dong.contributions(a, b)
loo = []
for k in dong.studies:
    sub = Network(dong.df[dong.df.study != k])
    if not sub.connected or a not in sub.trts or b not in sub.trts:
        loo.append(dict(study=k, imp_design=np.nan))
        continue
    loo.append(dict(study=k, imp_design=1 - r_full / rab(sub, a, b)))
loo = pd.DataFrame(loo).merge(res[["study", "d", "W"]], on="study")
top = loo.sort_values("imp_design", ascending=False).head(5)
R["loo_design"] = dict(
    top5=top.round(4).to_dict("records"),
    corr_d=float(loo[["imp_design", "d"]].corr().iloc[0, 1]),
    corr_W=float(loo[["imp_design", "W"]].corr().iloc[0, 1]),
    study3=float(loo.loc[loo.study == "3", "imp_design"].iloc[0]),
    study12=float(loo.loc[loo.study == "12", "imp_design"].iloc[0]),
    sum_importance=float(loo.imp_design.sum()), min_importance=float(loo.imp_design.min()))

# ---------------------------------------------------------------- 2. effective number of direct trials
eff = {}
for x, y in [("Placebo", "TIO-HH"), ("Placebo", "TIO-SMI")]:
    r = dong.contributions(x, y)
    d = r.d.to_numpy()
    eff[f"{x}:{y}"] = dict(n_direct=int((d > 0).sum()), direct_mass=float(d.sum()),
                          eff_trials=float(d.sum() ** 2 / np.sum(d ** 2)),
                          n_zero_throughput=int((r.W < 1e-9).sum()), n_studies=len(d))
R["effective_trials"] = eff

# ---------------------------------------------------------------- 3. allocation
def reduction(net, tgt, u, v, g):
    e = net.e(*tgt)
    dvec = net.e(u, v)
    phi = net.Lp @ e
    ruv = float(dvec @ net.Lp @ dvec)
    return g * float(dvec @ phi) ** 2 / (1 + g * ruv) / float(e @ phi)


alloc = []
for u, v in combinations(dong.trts, 2):
    alloc.append(dict(trial=f"{u}:{v}", reduction=reduction(dong, (a, b), u, v, 1000 / 4)))
alloc = sorted(alloc, key=lambda z: -z["reduction"])
red_bal = reduction(dong, (a, b), a, b, 1000 / 4)
red_21 = reduction(dong, (a, b), a, b, (2000 / 3) * (1000 / 3) / 1000)
tg = ("Eculizumab", "Inebilizumab")
nm_rank = sorted([dict(trial=f"{u}:{v}", reduction=reduction(nm, tg, u, v, 200 / 4))
                  for u, v in combinations(nm.trts, 2)], key=lambda z: -z["reduction"])
R["allocation"] = dict(copd_N1000=alloc, copd_loss_2to1=1 - red_21 / red_bal,
                       nmosd_N200=nm_rank)

# ---------------------------------------------------------------- 4. design-based precision gain
dp = []
for name, net in [("COPD", dong), ("Cystic fibrosis", cf), ("NMOSD", nm), ("Liver", gur)]:
    sig = study_significance(net)
    for x, y in combinations(net.trts, 2):
        r = net.contributions(x, y)
        direct = r[r.d > 0]
        if len(direct) == 0:
            continue
        rs = []
        nonsig_w = 0.0
        for _, row in direct.iterrows():
            s = sig[(sig.study == row.study) & (((sig.a == x) & (sig.b == y)) | ((sig.a == y) & (sig.b == x)))].iloc[0]
            rs.append(1 / s.n_a + 1 / s.n_b)
            if s.fisher_p >= 0.05:
                nonsig_w += row.d
        dp.append(dict(network=name, target=f"{x}:{y}", n_direct=len(direct),
                       gain=min(rs) / rab(net, x, y), share_nonsig=nonsig_w / direct.d.sum(),
                       d_max=float(direct.d.max())))
dp = pd.DataFrame(dp)
dp.to_csv(os.path.join(OUT, "design_precision.csv"), index=False)
R["design_precision"] = {
    n: dict(targets=len(g), median_gain=float(g.gain.median()),
            median_gain_allnonsig=float(g[g.share_nonsig > 0.999].gain.median()) if (g.share_nonsig > 0.999).any() else None,
            n_allnonsig=int((g.share_nonsig > 0.999).sum()))
    for n, g in dp.groupby("network")}
R["design_precision"]["COPD_Placebo:TIO-HH"] = float(dp[(dp.network == "COPD") & (dp.target == "Placebo:TIO-HH")].gain.iloc[0])

# ---------------------------------------------------------------- 5. efficiency of arm-size weights
def oracle_var(net, x, y, p):
    w = net.n / (p * (1 - p))
    o = Network(net.df, weights=w)
    c = o.coefficient_vector(x, y)
    return float(np.sum(c ** 2 * p * (1 - p) / net.n))


effic = {}
for name, net in [("COPD", dong), ("Cystic fibrosis", cf), ("Liver", gur), ("NMOSD", nm)]:
    pooled = net.df.groupby("study").apply(lambda g: (g.events.sum() + 0.5) / (g.n.sum() + 1.0), include_groups=False)
    configs = {"study_pooled": pooled.loc[net.df.study].to_numpy(),
               "mle_floor_0.005": np.clip(constrained_null_mle(net, "consistency")[0], 0.005, 0.995)}
    effic[name] = {}
    for cname, p in configs.items():
        vals = [oracle_var(net, x, y, p) / net.exact_variance(x, y, p) for x, y in combinations(net.trts, 2)]
        effic[name][cname] = dict(min=float(np.min(vals)), median=float(np.median(vals)), max=float(np.max(vals)),
                                  n_targets=len(vals), risk_range=[float(p.min()), float(p.max())])
R["efficiency"] = effic

# ---------------------------------------------------------------- 6. binomial MLE (additive model) vs design-fixed
p_mle, res_mle = constrained_null_mle(dong, "consistency")
X = dong.design_consistency()
beta = np.linalg.lstsq(X, p_mle, rcond=None)[0]
ntrt = len(dong.trts) - 1
mu = np.concatenate([[0.0], beta[-ntrt:]])          # reference = first treatment (alphabetical)
mle = []
for t in ["ICS", "LABA", "LABA-ICS", "TIO-HH", "TIO-SMI"]:
    est_mle = mu[dong.tix[t]] - mu[dong.tix["Placebo"]]
    est_df = dong.estimate("Placebo", t)
    mle.append(dict(target=f"Placebo:{t}", design_fixed=est_df, mle=float(est_mle), diff=float(est_mle - est_df)))
zero_trials = [k for k, g in dong.df.groupby("study") if g.events.sum() == 0]
R["mle_comparison"] = dict(rows=mle, max_abs_diff=float(max(abs(m["diff"]) for m in mle)),
                           boundary_arms=int((p_mle < 1e-6).sum()), zero_event_trials=zero_trials,
                           zero_event_trial_treatments={k: sorted(dong.df[dong.df.study == k].treatment) for k in zero_trials},
                           nll=float(-np.sum(dong.r * np.log(np.clip(p_mle, 1e-300, 1)) + (dong.n - dong.r) * np.log1p(-np.clip(p_mle, 0, 1 - 1e-12)))))

# ---------------------------------------------------------------- 7. TIOSPIR (Wise et al. 2013, NEJM)
x, y = "TIO-HH", "TIO-SMI"
est = dong.estimate(x, y)
se = np.sqrt(dong.exact_variance(x, y))
ci = binomial_interval(dong, x, y, B=20000, seed=5)
G = 1 / rab(dong, x, y)
tios = {"TIO-SMI": (423, 5711), "TIO-HH": (439, 5694)}
g = 5711 * 5694 / (5711 + 5694)
d_pred = g / (g + G)
df2 = pd.concat([dong.df, pd.DataFrame([dict(study="TIOSPIR", treatment=t, events=r, n=n) for t, (r, n) in tios.items()])],
                ignore_index=True)
net2 = Network(df2)
r2 = net2.contributions(x, y)
R["tiospir"] = dict(network_est=est, network_se=float(se), network_interval=[float(v) for v in ci] if np.ndim(ci) else ci,
                    direct_mass_before=float(dong.contributions(x, y).d.sum()),
                    tiospir_rd=423 / 5711 - 439 / 5694, d_predicted=float(d_pred),
                    d_realized=float(r2.loc[r2.study == "TIOSPIR", "d"].iloc[0]),
                    est_after=float(r2.attrs["estimate"]), se_after=float(np.sqrt(net2.exact_variance(x, y))),
                    placebo_tiosmi_after=float(net2.estimate("Placebo", "TIO-SMI")))

# ---------------------------------------------------------------- 8. cystic fibrosis sensitivity
def cf_targets(net):
    out = {}
    for x, y in [("TEZ/IVA", "Triple"), ("Placebo", "Triple"), ("Placebo", "TEZ/IVA")]:
        r = net.contributions(x, y)
        c = net.coefficient_vector(x, y)
        E = {k: float(np.sum(c[net.rows[k]] ** 2 / net.w[net.rows[k]])) for k in net.studies}
        rr = rab(net, x, y)
        out[f"{x}:{y}"] = dict(est=r.attrs["estimate"], direct=1 - r.attrs["q"],
                               d={k: float(v) for k, v in zip(r.study, r.d) if v > 0},
                               eta={k: E[k] / rr for k in net.studies if E[k] / rr > 0.004})
    return out


# main analysis: the seven parallel-group trials (data/cf_sae_registry.csv); sensitivity: EXPAND, a
# crossover trial, added with its treatment periods entered as if they were independent arms
cfx = load("cf_sae_registry_with_expand.csv")
qd = q_decomposition(cfx)
tw = berger_boos_sup_wald(cfx, B=4000, n_starts=8, n_iter=80, seed=11)
cf_main = cf_targets(cf)
for k in cf_main:
    cf_main[k]["se"] = float(np.sqrt(cf.exact_variance(*k.split(":"))))
cf_withexp = cf_targets(cfx)
for k in cf_withexp:
    cf_withexp[k]["se"] = float(np.sqrt(cfx.exact_variance(*k.split(":"))))
weeks = {"STRIVE": 52, "ENVISION": 52, "TRAFFIC": 28, "TRANSPORT": 28, "EVOLVE": 28, "EXPAND": 8,
         "VX17-445-102": 28, "VX17-445-103": 8}      # adverse-event observation window (registry)
dfe = pd.read_csv(os.path.join(DATA, "cf_sae_registry.csv"))[["study", "treatment", "events", "n"]]   # parallel-group trials
pt = dfe.n * dfe.study.map(weeks) / 52.0            # patient-years
cfe = Network(dfe, weights=pt.to_numpy())
cfe_t = cf_targets(cfe)
for k in cfe_t:  # rate per patient-year: re-express the estimate with person-time denominators
    x, y = k.split(":")
    rate = (dfe.events / pt).to_numpy()
    cfe_t[k]["est_rate_per_py"] = float(cfe.estimate(x, y, phat=rate))
R["cf_sensitivity"] = dict(
    participants_main=int(dfe.n.sum()), events_main=int(dfe.events.sum()),
    main=cf_main, with_expand=cf_withexp,
    with_expand_Q=dict(Q_inc=qd["Q_inc"], df_inc=int(qd["df_inc"]), Q_het=qd["Q_het"], df_het=int(qd["df_het"])),
    with_expand_TW=dict(t=tw["t_obs"], df=tw["df"], chi2_p=tw["chi2_p"], p_bb=tw["p_value"]),
    exposure_weights=cfe_t, weeks=weeks)

# ---------------------------------------------------------------- 9. registry agreement
hist = pd.read_csv(os.path.join(DATA, "registry_history.csv"))
ratio = (hist.first_enrollment / hist.last_enrollment)
fd = pd.to_datetime(hist.first_date)
sd = pd.to_datetime(hist.start_date.astype(str).str.slice(0, 10), format="mixed")
same_month_only = hist.start_date.astype(str).str.len() <= 7          # start date given to the month only
# month-only start dates are read as the first of the month: a record dated earlier is before the start
before = (fd < sd) & ~(same_month_only & (fd.dt.to_period("M") == sd.dt.to_period("M")))
reg = pd.read_csv(os.path.join(OUT, "registry_design_check.csv"))
nontriv = reg[(reg.study != "RIN-1") & (reg.d_pub > 0) & (reg.d_pub < 0.999)]
dir_rows = reg[reg.d_pub > 0]
R["registry"] = dict(
    n_trials_history=int(len(hist)), anticipated_over_realized=[float(ratio.min()), float(ratio.max())],
    realized_over_anticipated=[float((1 / ratio).min()), float((1 / ratio).max())],
    direct_rows=int(len(dir_rows)), direct_rows_trials=int(dir_rows.study.nunique()),
    corr_all=float(np.corrcoef(dir_rows.d_pub, dir_rows.d_anticipated)[0, 1]),
    mad_all=float(np.mean(np.abs(dir_rows.d_pub - dir_rows.d_anticipated))),
    nontrivial_rows=int(len(nontriv)),
    corr_nontrivial=float(np.corrcoef(nontriv.d_pub, nontriv.d_anticipated)[0, 1]),
    mad_nontrivial=float(np.mean(np.abs(nontriv.d_pub - nontriv.d_anticipated))),
    max_abs_nontrivial=float(np.max(np.abs(nontriv.d_pub - nontriv.d_anticipated))),
    max_abs_actual_vs_pub=float(np.max(np.abs(reg.d_pub - reg.d_actual))),
    n_first_record_before_start=int(before.sum()),
    max_days_after_start=int(max(0, ((fd - sd).dt.days[~same_month_only & ~before]).max())),
    n_same_month=int((same_month_only & (fd.dt.to_period("M") == pd.PeriodIndex(hist.start_date.astype(str).str.slice(0, 7), freq="M"))).sum()))

# ---------------------------------------------------------------- 10. significance and weight
s = pd.read_csv(os.path.join(OUT, "significance_size_weight.csv"))
q2 = {}
for name, g2 in list(s.groupby("network")) + [("all", s)]:
    q2[name] = dict(pairs=len(g2), frac_pairs_nonsig=float((g2.p >= 0.05).mean()),
                    share_weight_nonsig=float(g2[g2.p >= 0.05].d.sum() / g2.d.sum()),
                    spearman_p_d=float(spearmanr(g2.p, g2.d).correlation) if len(g2) > 2 else None)
pt_hh = s[(s.network == "COPD") & (s.target == "Placebo:TIO-HH")]
q2["COPD_Placebo:TIO-HH_p_range"] = [float(pt_hh.p.min()), float(pt_hh.p.max())]
R["q2"] = q2

# ---------------------------------------------------------------- 11. moment estimator, COPD (Proposition 2)
from rdcsp import null_expectation_Q
qd_dong = q_decomposition(dong)
e0 = null_expectation_Q(dong, np.clip(constrained_null_mle(dong, "consistency")[0], 0, 1))
sw_ = np.sqrt(dong.w)
Xw_ = sw_[:, None] * dong.design_specific()
Ad_ = np.eye(len(sw_)) - Xw_ @ np.linalg.pinv(Xw_.T @ Xw_) @ Xw_.T
tr_ = float(np.sum(np.diag(Ad_) * dong.w * (1 - 1 / dong.n)))   # Proposition 2 with E{p(1-p)} = p_bar(1-p_bar) - sigma^2
s2_ = max(qd_dong["Q_het"] - e0["E_Q_het"], 0) / tr_
c_ = dong.coefficient_vector("Placebo", "TIO-HH")
v_ = float((c_ ** 2 * dong.phat * (1 - dong.phat) / np.maximum(dong.n - 1, 1)).sum())
R["moment_copd"] = dict(Q_het=qd_dong["Q_het"], E0_het=e0["E_Q_het"], E0_net=e0["E_Q_net"], E0_inc=e0["E_Q_inc"],
                        trace=tr_, sigma2=s2_, tau=float(np.sqrt(2 * s2_)), se_common=float(np.sqrt(v_)),
                        se_re=float(np.sqrt(v_ + s2_ * np.sum(c_ ** 2))))

json.dump(R, open(os.path.join(OUT, "revision_analyses.json"), "w"), indent=1, default=float)
print(json.dumps(R, indent=1, default=float)[:20000])
