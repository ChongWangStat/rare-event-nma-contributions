"""LaTeX macros for numbers quoted in the main text that come from several results files, so that the
text cannot drift from the results.  Usage: python3 make_text_macros.py -> ../results/text_macros.tex"""
import json

OUT = "../results"
R = json.load(open(f"{OUT}/revision_analyses.json"))
S = json.load(open(f"{OUT}/significance_size_weight.json"))
I = json.load(open(f"{OUT}/informativeness.json"))

m = {}
m["SigPairs"] = str(S["all"]["pairs"])
m["SigRhoCOPD"] = f"{S['COPD']['spearman_p_d']:.2f}".replace("-", "$-$")
m["SigRhoCF"] = f"{S['Cystic fibrosis']['spearman_p_d']:.2f}".replace("-", "$-$")
m["SigRhoAll"] = f"{S['all']['spearman_p_d']:.2f}".replace("-", "$-$")
m["SigRhoSize"] = f"{S['all']['spearman_n_d']:.2f}"
m["InfGainCOPD"] = f"{I['COPD']['median_precision_gain']:.1f}"
m["InfGainCF"] = f"{I['Cystic fibrosis']['median_precision_gain']:.1f}"
m["TauDong"] = f"{R['moment_copd']['tau']:.4f}"
m["SeReDong"] = f"{R['moment_copd']['se_re']:.4f}"
reg = R["registry"]
m["RegNontriv"] = str(reg["nontrivial_rows"])
m["RegCorr"] = f"{reg['corr_nontrivial']:.2f}"
m["RegMAD"] = f"{reg['mad_nontrivial']:.3f}"
m["RegMax"] = f"{reg['max_abs_nontrivial']:.3f}"
m["CfExpoD"] = f"{R['cf_sensitivity']['exposure_weights']['TEZ/IVA:Triple']['direct']:.2f}"
mc = R["moment_copd"]
m["EzeroHet"] = f"{mc['E0_het']:.3f}"
m["TraceHet"] = f"{mc['trace']:,.0f}".replace(",", "{,}")
we = R["cf_sensitivity"]
m["CfExpQinc"] = f"{we['with_expand_Q']['Q_inc']:.3f}"
m["CfExpTW"] = f"{we['with_expand_TW']['t']:.2f}"
m["CfExpChi"] = f"{we['with_expand_TW']['chi2_p']:.2f}"
m["CfExpBB"] = f"{we['with_expand_TW']['p_bb']:.2f}"
m["CfParticipants"] = f"{we['participants_main']:,}".replace(",", "{,}")
m["CfEvents"] = str(we["events_main"])
ef = R["efficiency"]
m["EffCOPD"] = f"{100 * ef['COPD']['study_pooled']['median']:.0f}"
m["EffCF"] = f"{100 * ef['Cystic fibrosis']['study_pooled']['median']:.0f}"
m["EffLiver"] = f"{100 * ef['Liver']['study_pooled']['median']:.0f}"
m["EffRiskLo"] = f"{100 * ef['COPD']['study_pooled']['risk_range'][0]:.2f}"
m["EffRiskHi"] = f"{100 * ef['COPD']['study_pooled']['risk_range'][1]:.0f}"
sma = json.load(open(f"{OUT}/sma_summary.json"))
m["TauSma"] = f"{sma['tau']:.2f}"
m["SeReSma"] = f"{sma['se_re']:.2f}"
m["RegBefore"] = str(reg["n_first_record_before_start"])
m["RegDaysAfter"] = str(reg["max_days_after_start"])
summ = json.load(open(f"{OUT}/summary.json"))
gb = summ["gur"]["control_aprotinin"]["basic"]
m["GurBasicLo"] = f"{gb[0]:.3f}".replace("-", "-")
m["GurBasicHi"] = f"{gb[1]:.3f}"
import os
import pandas as pd
# TIOSPIR weight from its first registration
m["TiospirReg"] = f"{R['tiospir']['d_from_registration']:.2f}"
m["TiospirRegN"] = f"{R['tiospir']['anticipated_enrollment']:,}".replace(",", "{,}")
# NMOSD allocation decision
A = json.load(open(f"{OUT}/nmosd_allocation.json"))
plan = {(p["budget"], p["plan"]): p for p in A["plans"]}
pc = lambda x: f"{100 * x:.0f}"
m["AllocTwoMean"] = pc(plan[(400, "best two trials")]["mean_reduction"])
m["AllocTwoMax"] = pc(plan[(400, "best two trials")]["max_reduction"])
m["AllocEcuIneMean"] = pc(plan[(400, "eculizumab vs inebilizumab only")]["mean_reduction"])
m["AllocBound"] = pc(plan[(400, "unrestricted optimum (bound)")]["mean_reduction"])
m["AllocSingle"] = pc(plan[(400, "best single trial")]["mean_reduction"])
m["AllocSingleHH"] = pc(plan[(400, "best single head-to-head trial")]["mean_reduction"])
m["AllocTwoMeanSmall"] = pc(plan[(200, "best two trials")]["mean_reduction"])
# MOVER-based numbers quoted in the applications
from rdcsp import Network, mover_interval
IJ = json.load(open(f"{OUT}/informativeness.json"))
m["InfNetMid"] = str(IJ["COPD"]["n_network_excludes_mid"])
m["InfBestMid"] = str(IJ["COPD"]["n_best_trial_excludes_mid"])
gm = summ["gur"]["control_aprotinin"]["mover"]
m["GurMoverLo"] = f"{gm[0]:.3f}"
m["GurMoverHi"] = f"{gm[1]:.3f}"
_d = Network(pd.read_csv("../data/dong2013_copd_mortality.csv"))
_hs = mover_interval(_d.coefficient_vector("TIO-HH", "TIO-SMI"), _d.r, _d.n)
m["HhSmiLo"], m["HhSmiHi"] = f"{_hs[0]:.3f}", f"{_hs[1]:.3f}"
_th = mover_interval(_d.coefficient_vector("Placebo", "TIO-HH"), _d.r, _d.n)
m["PlaHhLo"], m["PlaHhHi"] = f"{_th[0]:.3f}", f"{_th[1]:.3f}"
_ts = mover_interval(_d.coefficient_vector("Placebo", "TIO-SMI"), _d.r, _d.n)
m["PlaSmiLo"], m["PlaSmiHi"] = f"{_ts[0]:.3f}", f"{_ts[1]:.3f}"
# interval coverage
C = pd.read_csv(f"{OUT}/interval_coverage.csv")
m["MoverCovMin"] = f"{C.cov_mover.min():.3f}"
m["MoverCovMax"] = f"{C.cov_mover.max():.3f}"
m["BasicCovMin"] = f"{min(C.cov_basic.min(), C.cov_normal.min()):.2f}"
# misspecification
if os.path.exists(f"{OUT}/sim_misspec_logit.csv"):
    ML = pd.read_csv(f"{OUT}/sim_misspec_logit.csv").iloc[0]
    MA = pd.read_csv(f"{OUT}/sim_misspec_additive.csv").iloc[0]
    m["MisCovMin"] = f"{min(ML['cov_mover Placebo:TIO-HH'], ML['cov_mover Placebo:TIO-SMI']):.3f}"
    m["MisCovMax"] = f"{max(ML['cov_mover Placebo:TIO-HH'], ML['cov_mover Placebo:TIO-SMI']):.3f}"
    m["MisPowerHet"] = f"{ML['qhet']:.2f}"
    m["MisPowerInc"] = f"{ML['qinc']:.2f}"
    m["MisLevelHet"] = f"{MA['qhet']:.3f}"
    m["MisLevelInc"] = f"{MA['qinc']:.3f}"
    m["MisThetaSMI"] = f"{ML['theta_star Placebo:TIO-SMI']:.3f}"
    m["MisRdBarSMI"] = f"{ML['rd_at_pbar Placebo:TIO-SMI']:.3f}"
    m["MisPbar"] = f"{100 * ML['placebo_risk']:.1f}"
with open(f"{OUT}/text_macros.tex", "w") as f:
    for k, v in m.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(m)
