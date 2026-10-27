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
with open(f"{OUT}/text_macros.tex", "w") as f:
    for k, v in m.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(m)
