"""Applications: COPD mortality (Dong 2013), liver-transplant mortality (Gurusamy 2011),
and the COVID-19 network used in the CSP papers.  Writes CSV/LaTeX tables and figures
to ../results.  Run time: a few minutes (binomial calibration draws)."""
import json, time
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2
from rdcsp import mover_interval
from rdcsp import (Network, q_decomposition, pearson_weights, binomial_calibration, null_expectation_Q,
                   berger_boos_envelope, binomial_interval, study_significance,
                   throughput_table, add_planned_trial, iv_rd_estimate, verify_identities)

OUT = "../results"
summary = {}


def fmt(x, d=4):
    x = 0.0 if abs(x) < 0.5 * 10 ** (-d) else x          # no "-0.0000"
    return f"{x:.{d}f}"


# ------------------------------------------------------------------------- #
# 1. COPD mortality network (Dong et al. 2013): dense network, rare events
# ------------------------------------------------------------------------- #
t0 = time.time()
dong = Network(pd.read_csv("../data/dong2013_copd_mortality.csv"))
summary["dong"] = dict(studies=len(dong.studies), treatments=dong.trts,
                       arms=int(len(dong.n)), total_events=int(dong.r.sum()), total_n=int(dong.n.sum()),
                       multi_arm=int(sum(len(dong.rows[k]) > 2 for k in dong.studies)),
                       zero_event_studies=int(sum(dong.r[dong.rows[k]].sum() == 0 for k in dong.studies)),
                       zero_arms=int((dong.r == 0).sum()),
                       event_rate=float(dong.r.sum() / dong.n.sum()))
summary["dong"]["identities"] = {k: float(v) for k, v in verify_identities(dong, "Placebo", "TIO-HH").items()}

# all targets versus placebo
rows = []
for t in dong.trts:
    if t == "Placebo":
        continue
    res = dong.contributions("Placebo", t)
    ci = binomial_interval(dong, "Placebo", t, B=20000, seed=11)
    iv, ivse = iv_rd_estimate(dong, "Placebo", t)
    rows.append(dict(target=f"Placebo:{t}", estimate=res.attrs["estimate"], exact_se=ci["exact_se"],
                     pct_lo=float(mover_interval(dong.coefficient_vector("Placebo", t), dong.r, dong.n)[0]),   # MOVER-Wilson 95% interval
                     pct_hi=float(mover_interval(dong.coefficient_vector("Placebo", t), dong.r, dong.n)[1]),
                     basic_lo=ci["basic"][0], basic_hi=ci["basic"][1],
                     direct_mass=1 - res.attrs["q"], indirect_mass=res.attrs["q"],
                     n_direct_studies=int((res["d"] > 0).sum()),
                     iv_cc_estimate=iv, iv_cc_se=ivse))
tab = pd.DataFrame(rows)
tab.to_csv(f"{OUT}/dong_targets.csv", index=False)
with open(f"{OUT}/dong_targets.tex", "w") as f:
    f.write("\\begin{tabular}{lrrrrrr}\n\\toprule\nTarget & Estimate & Exact SE & 95\\% interval & Direct mass & Direct studies & IV+cc estimate (SE) \\\\\n\\midrule\n")
    for r in rows:
        f.write(f"{r['target']} & {fmt(r['estimate'])} & {fmt(r['exact_se'])} & [{fmt(r['pct_lo'])}, {fmt(r['pct_hi'])}] & {fmt(r['direct_mass'],3)} & {r['n_direct_studies']} & {fmt(r['iv_cc_estimate'])} ({fmt(r['iv_cc_se'])}) \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# study table for Placebo:TIO-HH
res = dong.contributions("Placebo", "TIO-HH")
res.to_csv(f"{OUT}/dong_placebo_tiohh_studies.csv", index=False)
top = res.sort_values("W", ascending=False)
with open(f"{OUT}/dong_placebo_tiohh_studies.tex", "w") as f:
    f.write("\\begin{tabular}{llrrrrr}\n\\toprule\nStudy & Arms & $d_k$ & $\\tau_k-d_k$ & Study RD & $C^D_k$ & $C_k$ \\\\\n\\midrule\n")
    for _, r in top.head(12).iterrows():
        de = "--" if np.isnan(r["direct_est"]) else fmt(r["direct_est"])
        f.write(f"{r['study']} & {r['arms']} & {fmt(r['d'],3)} & {fmt(r['WI'],3)} & {de} & {fmt(r['C_D'],5)} & {fmt(r['C'],5)} \\\\\n")
    rest = top.iloc[12:]
    f.write(f"\\multicolumn{{2}}{{l}}{{Other {len(rest)} studies}} & {fmt(rest['d'].sum(),3)} & {fmt(rest['WI'].sum(),3)} & -- & {fmt(rest['C_D'].sum(),5)} & {fmt(rest['C'].sum(),5)} \\\\\n")
    f.write(f"\\midrule\nSum & & {fmt(res['d'].sum(),3)} & & & {fmt(res['C_D'].sum(),5)} & {fmt(res['C'].sum(),5)} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
summary["dong"]["placebo_tiohh"] = dict(estimate=res.attrs["estimate"], q=res.attrs["q"],
                                        top_study=str(top.iloc[0]["study"]), top_d=float(top.iloc[0]["d"]))

# significance vs contribution: each study's own placebo comparison
sig = study_significance(dong)
sig = sig[(sig["a"] == "Placebo") | (sig["b"] == "Placebo")].copy()
sig["other"] = np.where(sig["a"] == "Placebo", sig["b"], sig["a"])
dlist = []
for _, r in sig.iterrows():
    c = dong.contributions("Placebo", r["other"])
    dlist.append(float(c.loc[c["study"] == r["study"], "d"].iloc[0]))
sig["direct_weight"] = dlist
sig["n_total"] = sig["n_a"] + sig["n_b"]
sig["significant"] = sig["fisher_p"] < 0.05
sig.to_csv(f"{OUT}/dong_significance_vs_contribution.csv", index=False)
cross = sig.groupby("significant").agg(studies=("study", "count"), total_direct_weight=("direct_weight", "sum"),
                                       median_direct_weight=("direct_weight", "median"),
                                       max_direct_weight=("direct_weight", "max"))
cross.to_csv(f"{OUT}/dong_significance_crosstab.csv")
summary["dong"]["signif_crosstab"] = cross.to_dict()
summary["dong"]["largest_nonsig_direct"] = sig[~sig["significant"]].sort_values("direct_weight", ascending=False).head(3)[["study", "other", "r_a", "n_a", "r_b", "n_b", "fisher_p", "direct_weight"]].to_dict("records")
summary["dong"]["signif_studies"] = sig[sig["significant"]][["study", "other", "r_a", "n_a", "r_b", "n_b", "fisher_p", "direct_weight"]].to_dict("records")

# comparative information: throughput across all targets
thr = throughput_table(dong)
thr.to_csv(f"{OUT}/dong_throughput.csv")
summary["dong"]["throughput_corr_with_n"] = float(np.corrcoef(thr["max_throughput"], thr["total_n"])[0, 1])
big_small = thr.sort_values("total_n", ascending=False).head(10)[["total_n", "max_throughput"]]
summary["dong"]["largest_studies_throughput"] = big_small.round(4).to_dict("index")

# Q decomposition with binomial calibration (Pearson weights)
cal = binomial_calibration(dong, B=20000, seed=5)
env = berger_boos_envelope(dong, B=1000, M=10, seed=6)
ex0 = null_expectation_Q(dong, cal["p_null"])
calP = binomial_calibration(dong, B=4000, seed=5, pearson=True)
summary["dong"]["Q"] = dict(Q_net=cal["Q_net"], Q_inc=cal["Q_inc"], Q_het=cal["Q_het"],
                            df_net=int(cal["df_net"]), df_inc=int(cal["df_inc"]), df_het=int(cal["df_het"]),
                            chi2_p_net=float(1 - chi2.cdf(cal["Q_net"], cal["df_net"])),
                            chi2_p_inc=float(1 - chi2.cdf(cal["Q_inc"], cal["df_inc"])),
                            chi2_p_het=float(1 - chi2.cdf(cal["Q_het"], cal["df_het"])),
                            binom_p_net=cal["p_net"], binom_p_inc=cal["p_inc"], binom_p_het=cal["p_het"],
                            binom_mean_net=float(cal["draws"][:, 0].mean()), binom_mean_inc=float(cal["draws"][:, 1].mean()),
                            binom_mean_het=float(cal["draws"][:, 2].mean()),
                            envelope_p_net=env["p_net"], envelope_p_inc=env["p_inc"], envelope_p_het=env["p_het"],
                            E0_net=ex0["E_Q_net"], E0_inc=ex0["E_Q_inc"], E0_het=ex0["E_Q_het"],
                            null_fit_min=float(cal["p_null"].min()), null_fit_floored=int((q_decomposition(dong)["fit_c"] < 0.5 / dong.n).sum()),
                            null_fit_negative=int((q_decomposition(dong)["fit_c"] < 0).sum()),
                            pearson=dict(Q_net=calP["Q_net"], Q_inc=calP["Q_inc"], binom_mean_net=float(calP["draws"][:, 0].mean()),
                                         binom_median_net=float(np.median(calP["draws"][:, 0])), binom_p99_net=float(np.quantile(calP["draws"][:, 0], 0.99)),
                                         chi2_p_net=float(1 - chi2.cdf(calP["Q_net"], calP["df_net"])), binom_p_net=calP["p_net"],
                                         chi2_p_inc=float(1 - chi2.cdf(calP["Q_inc"], calP["df_inc"])), binom_p_inc=calP["p_inc"]))
np.save(f"{OUT}/dong_Q_draws.npy", cal["draws"])

# design-stage contribution: a planned placebo-controlled TIO-SMI trial of various sizes
plan = []
for n in [200, 500, 1000, 2000]:
    new = add_planned_trial(dong, {"Placebo": n, "TIO-SMI": n})
    c = new.contributions("Placebo", "TIO-SMI")
    c2 = new.contributions("Placebo", "TIO-HH")
    plan.append(dict(n_per_arm=n, direct_weight_TIOSMI=float(c.loc[c["study"] == "PLANNED", "d"].iloc[0]),
                     throughput_TIOHH=float(c2.loc[c2["study"] == "PLANNED", "W"].iloc[0])))
pd.DataFrame(plan).to_csv(f"{OUT}/dong_design_stage.csv", index=False)
summary["dong"]["design_stage"] = plan
summary["dong"]["time_s"] = time.time() - t0

# Figure 1: contributions for Placebo:TIO-HH
import figstyle as fs
with plt.rc_context(fs.RC):
    fig, ax = plt.subplots(figsize=(fs.width(0.9), 5.4))
    fs.margins(fig, left=1.62, right=0.15, bottom=0.45, top=0.06)
    d = res.sort_values("C")
    yy = np.arange(len(d))
    ax.barh(yy, d["C_D"], height=0.72, color=fs.BLUE, label="direct part $C^D_k$", zorder=2)
    ax.barh(yy, d["C_I"], left=d["C_D"], height=0.72, color=fs.VERMILION, label="indirect part $C^I_k$", zorder=2)
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{s} ({a})" for s, a in zip(d["study"], d["arms"])], fontsize=7)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.set_ylim(-0.7, len(d) - 0.3)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=fs.GRID, lw=0.6, zorder=0)
    ax.axvline(0, color=fs.INK, lw=0.7, zorder=3)
    ax.set_xlabel("additive contribution to the estimated risk difference (TIO-HH \u2212 placebo)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), handlelength=1.4, handleheight=0.9)
    ax.text(0.0, 0.905, f"contributions sum to {res.attrs['estimate']:.5f}".replace("-", "−"), transform=ax.transAxes,
            ha="left", va="top", fontsize=8)
    fig.savefig(f"{OUT}/figures/fig_dong_contributions.pdf")
    plt.close(fig)

# Figure 2: binomial reference vs chi-square for Q_inc
fig, ax = plt.subplots(figsize=(6, 3.6))
ax.hist(cal["draws"][:, 1], bins=60, density=True, color="#bbbbbb", label="binomial reference (simulated under the null fit)")
ax.axvline(ex0["E_Q_inc"], color="k", ls="--", label=f"exact null mean = {ex0['E_Q_inc']:.3f}")
ax.axvline(cal["Q_inc"], color="r", label=f"observed $Q_{{inc}}$ = {cal['Q_inc']:.3f} (p = {cal['p_inc']:.2f})")
ax.set_xlabel("$Q_{inc}$ with design-fixed weights $w_{ku}=n_{ku}$")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{OUT}/figures/fig_dong_Qinc.pdf")
plt.close(fig)

# ------------------------------------------------------------------------- #
# 2. Liver-transplant mortality (Gurusamy 2011): sparse, small trials
# ------------------------------------------------------------------------- #
gur = Network(pd.read_csv("../data/gurusamy2011_liver_transplant_mortality.csv"))
summary["gur"] = dict(studies=len(gur.studies), treatments=gur.trts, arms=int(len(gur.n)),
                      total_events=int(gur.r.sum()), total_n=int(gur.n.sum()),
                      multi_arm=int(sum(len(gur.rows[k]) > 2 for k in gur.studies)),
                      zero_arms=int((gur.r == 0).sum()),
                      zero_event_studies=int(sum(gur.r[gur.rows[k]].sum() == 0 for k in gur.studies)),
                      median_n_per_arm=float(np.median(gur.n)))
res_g = gur.contributions("Control/Placebo", "Aprotinin")
res_g.to_csv(f"{OUT}/gur_control_aprotinin_studies.csv", index=False)
ci_g = binomial_interval(gur, "Control/Placebo", "Aprotinin", B=20000, seed=12)
cal_g = binomial_calibration(gur, B=4000, seed=7)
ex_g = null_expectation_Q(gur, cal_g["p_null"])
summary["gur"]["control_aprotinin"] = dict(estimate=res_g.attrs["estimate"], q=res_g.attrs["q"],
                                           exact_se=ci_g["exact_se"], percentile=ci_g["percentile"], basic=ci_g["basic"], normal=ci_g["normal"],
                                           mover=list(mover_interval(gur.coefficient_vector("Control/Placebo", "Aprotinin"), gur.r, gur.n)))
summary["gur"]["Q"] = dict(Q_net=cal_g["Q_net"], Q_inc=cal_g["Q_inc"], Q_het=cal_g["Q_het"],
                           df_net=int(cal_g["df_net"]), df_inc=int(cal_g["df_inc"]), df_het=int(cal_g["df_het"]),
                           chi2_p_inc=float(1 - chi2.cdf(cal_g["Q_inc"], cal_g["df_inc"])), binom_p_inc=cal_g["p_inc"],
                           chi2_p_het=float(1 - chi2.cdf(cal_g["Q_het"], cal_g["df_het"])), binom_p_het=cal_g["p_het"],
                           binom_mean_inc=float(cal_g["draws"][:, 1].mean()), binom_mean_het=float(cal_g["draws"][:, 2].mean()),
                           E0_inc=ex_g["E_Q_inc"], E0_het=ex_g["E_Q_het"], null_fit_min=float(cal_g["p_null"].min()))
# zero-event studies and their coefficients
zs = []
for k in gur.studies:
    idx = gur.rows[k]
    if gur.r[idx].sum() == 0:
        zs.append(dict(study=k, arms="/".join(gur.arm_trt[idx]), n=int(gur.n[idx].sum()),
                       W=float(res_g.loc[res_g["study"] == k, "W"].iloc[0]),
                       d=float(res_g.loc[res_g["study"] == k, "d"].iloc[0])))
summary["gur"]["zero_event_studies_detail"] = zs
with open(f"{OUT}/gur_control_aprotinin_studies.tex", "w") as f:
    f.write("\\begin{tabular}{llrrrrr}\n\\toprule\nStudy & Arms & $d_k$ & $\\tau_k-d_k$ & Study RD & $C^D_k$ & $C_k$ \\\\\n\\midrule\n")
    for _, r in res_g.sort_values("W", ascending=False).iterrows():
        de = "--" if np.isnan(r["direct_est"]) else fmt(r["direct_est"])
        f.write(f"{r['study']} & {r['arms'].replace('Control/Placebo','Control')} & {fmt(r['d'],3)} & {fmt(r['WI'],3)} & {de} & {fmt(r['C_D'],5)} & {fmt(r['C'],5)} \\\\\n")
    f.write(f"\\midrule\nSum & & {fmt(res_g['d'].sum(),3)} & & & {fmt(res_g['C_D'].sum(),5)} & {fmt(res_g['C'].sum(),5)} \\\\\n\\bottomrule\n" + "\\end{tabular}\n")

# ------------------------------------------------------------------------- #
# 3. COVID-19 network: risk scale versus the published logit-scale weights
# ------------------------------------------------------------------------- #
cov = Network(pd.read_csv("../data/covid19_wholenetwork.csv"))
res_c = cov.contributions("A", "E")
res_c.to_csv(f"{OUT}/covid_AE_studies.csv", index=False)
logit_published = {"SOLIDARITY": 0.639151, "ACTT-1": 0.140517, "RECOVERY": 0.0, "CoDEX": 0.0, "REMAP-CAP": 0.0}
summary["covid"] = dict(estimate=res_c.attrs["estimate"], q=res_c.attrs["q"],
                        logit_q_published=0.220332,
                        direct=[dict(study=r["study"], d_risk=float(r["d"]), d_logit=logit_published[r["study"]],
                                     W_risk=float(r["W"])) for _, r in res_c.iterrows()])
cal_c = binomial_calibration(cov, B=4000, seed=8)
ex_c = null_expectation_Q(cov, cal_c["p_null"])
summary["covid"]["Q"] = dict(Q_net=cal_c["Q_net"], Q_inc=cal_c["Q_inc"], Q_het=cal_c["Q_het"],
                             df_net=int(cal_c["df_net"]), df_inc=int(cal_c["df_inc"]), df_het=int(cal_c["df_het"]),
                             chi2_p_inc=float(1 - chi2.cdf(cal_c["Q_inc"], cal_c["df_inc"])), binom_p_inc=cal_c["p_inc"],
                             E0_inc=ex_c["E_Q_inc"], binom_p_het=cal_c["p_het"], E0_het=ex_c["E_Q_het"])

with open(f"{OUT}/summary.json", "w") as f:
    json.dump(summary, f, indent=1, default=float)
print(json.dumps(summary, indent=1, default=float)[:6000])
