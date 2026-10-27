"""Supplementary tables for the interval-coverage study and the misspecification simulation.
Usage: python3 make_robustness_tables.py  -> ../results/misspec.tex (interval_coverage.tex is written by
interval_coverage.py)"""
import pandas as pd

OUT = "../results"
rows = [pd.read_csv(f"{OUT}/sim_misspec_{t}.csv") for t in ["additive", "logit"]]
M = pd.concat(rows, ignore_index=True)
lab = {"additive": "Additive (model holds)", "logit": "Common odds ratio"}
L = [r"\begin{tabular}{llrrrrrrr}", r"\toprule",
     r"Truth & Comparison & $\theta^*$ & RD at mean placebo risk & Trial RDs & Bias (MC SE) & Cov.\ MOVER & Cov.\ basic & Cov.\ normal \\",
     r"\midrule"]
for _, r in M.iterrows():
    for t in ["Placebo:TIO-HH", "Placebo:TIO-SMI"]:
        rd = r[f"rd_at_pbar {t}"]
        rd_s = "--" if pd.isna(rd) else f"{rd:.4f}"
        L.append(f"{lab[r['truth']] if t.endswith('HH') else ''} & {t.replace(':', ' vs ')} & {r[f'theta_star {t}']:.4f} & {rd_s} & "
                 f"{r[f'trial_rd_min {t}']:.4f} to {r[f'trial_rd_max {t}']:.4f} & {r[f'bias {t}']:+.5f} ({r[f'bias_mcse {t}']:.5f}) & "
                 f"{r[f'cov_mover {t}']:.3f} & {r[f'cov_basic {t}']:.3f} & {r[f'cov_normal {t}']:.3f} \\\\")
L += [r"\midrule", r"Truth & \multicolumn{8}{l}{Rejection rate at level 0.05: $Q_{\mathrm{het}}$ plug-in / $Q_{\mathrm{inc}}$ plug-in / $T_W$ $\chi^2$} \\", r"\midrule"]
for _, r in M.iterrows():
    L.append(f"{lab[r['truth']]} & \\multicolumn{{8}}{{l}}{{{r['qhet']:.3f} / {r['qinc']:.3f} / {r['tw_chi2']:.3f}  ({int(r['reps'])} data sets)}} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
open(f"{OUT}/misspec.tex", "w").write("\n".join(L) + "\n")
print("\n".join(L))
