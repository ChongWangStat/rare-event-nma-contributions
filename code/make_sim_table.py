"""Table 1 of the main text: estimator (Panel A) and consistency tests (Panel B), for the synthetic
ten-study network (run_simulations.py Part A, run_sim_mle.py) and for the design-based simulation on the
cystic fibrosis network (sim_cf_design.py).  Monte Carlo standard errors are given in the caption.

Usage: python3 make_sim_table.py  -> ../results/sim_main.tex
"""
import pandas as pd

OUT = "../results"
A = pd.read_csv(f"{OUT}/sim_partA.csv")
M = pd.read_csv(f"{OUT}/sim_mle.csv")
C = pd.read_csv(f"{OUT}/sim_cf_design.csv")


def f(x, d=3):
    return f"{x:.{d}f}"


def sg(x, d=4):
    s = f"{x:+.{d}f}"
    return s[1:] if float(s) == 0 else s


lines = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
         r"\multicolumn{8}{l}{\emph{A. Estimator}} \\",
         r"Setting & Target & Bias & Exact SD & Empirical SD & Cov.\ basic & Cov.\ normal & IV+cc bias / cov. \\",
         r"\midrule"]
lab = {"low": "Synthetic, risks 1--4\\%", "verylow": "Synthetic, risks 0.2--1.2\\%"}
for _, r in A.iterrows():
    lines.append(f"{lab[r['rate']]} & {r['target']} & {sg(r['bias'])} & {f(r['exact_sd'], 4)} & {f(r['empirical_sd'], 4)} & "
                 f"{f(r['cov_basic'])} & {f(r['cov_exactvar_normal'])} & {sg(r['iv_bias'])} / {f(r['cov_iv_cc'])} \\\\")
c0 = C[C.delta == 0.0].iloc[0]
for t in ["TEZ/IVA:Triple", "Placebo:Triple"]:
    lines.append(f"Cystic fibrosis design & {t} & {sg(c0['bias ' + t])} & {f(c0['sd_exact ' + t], 4)} & {f(c0['sd_emp ' + t], 4)} & "
                 f"{f(c0['cov_basic ' + t])} & {f(c0['cov_normal ' + t])} & -- \\\\")
lines += [r"\midrule",
          r"\multicolumn{8}{l}{\emph{B. Consistency tests: rejection rate at level 0.05}} \\",
          r"Setting & $\delta$ & $Q_{\mathrm{inc}}$ plug-in & $T_W$ $\chi^2$ & $T_W$ plug-in & $T_W$ BB sup & (data sets) & \\",
          r"\midrule"]
for _, r in M.sort_values(["delta", "rate"], ascending=[True, True]).iterrows():
    lines.append(f"{lab[r['rate']]} & {r['delta']:.2f} & {f(r['fixed_mle_inc'])} & {f(r['wald_chi2'])} & {f(r['wald_plugin_mle'])} & "
                 f"{f(r['wald_bb_mle'])} & ({int(r['reps_bb'])}) & \\\\")
for _, r in C.sort_values("delta").iterrows():
    lines.append(f"Cystic fibrosis design & {r['delta']:.2f} & {f(r['qinc_plugin'])} & {f(r['tw_chi2'])} & {f(r['tw_plugin'])} & "
                 f"{f(r['tw_bb'])} & ({int(r['reps_sup'])}) & \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(f"{OUT}/sim_main.tex", "w").write("\n".join(lines) + "\n")

# Supplementary table: all recorded quantities of the synthetic estimator study, with Monte Carlo SEs
cols = [("True RD", "truth", 4), ("Mean estimate", "mean_est", 4), ("Bias", "bias", 5), ("Exact SD (true risks)", "exact_sd", 4),
        ("Empirical SD", "empirical_sd", 4), ("IV+cc bias", "iv_bias", 4), ("IV+cc empirical SD", "iv_sd", 4),
        ("Coverage, percentile", "cov_percentile", 3), ("Coverage, basic", "cov_basic", 3),
        ("Coverage, exact-variance normal", "cov_exactvar_normal", 3), ("Coverage, IV+cc normal", "cov_iv_cc", 3),
        ("Mean width, percentile", "width_percentile", 4), ("Mean width, normal", "width_normal", 4), ("Mean width, IV+cc", "width_iv", 4)]
hdr = " & ".join(f"{'low' if r['rate'] == 'low' else 'very low'}, {r['target']}" for _, r in A.iterrows())
S = [r"\begin{tabular}{l" + "r" * len(A) + "}", r"\toprule", " & " + hdr + r" \\", r"\midrule"]
for name, col, d in cols:
    S.append(name + " & " + " & ".join(f"{v:.{d}f}" for v in A[col]) + r" \\")
S.append("MC SE of bias & " + " & ".join(f"{sd / (r ** 0.5):.5f}" for sd, r in zip(A["empirical_sd"], A["reps"])) + r" \\")
S += [r"\bottomrule", r"\end{tabular}"]
open(f"{OUT}/sim_partA_supp.tex", "w").write("\n".join(S) + "\n")
print("\n".join(lines))
