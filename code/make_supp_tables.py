"""Supplementary tables built from revision_analyses.json and the registry files:
supp_mle.tex (binomial MLE vs design-fixed, COPD), supp_eff.tex (efficiency of arm-size weights),
supp_cfsens.tex (cystic fibrosis sensitivity analyses), supp_primary.tex (registered primary endpoints).
Usage: python3 make_supp_tables.py"""
import json

import pandas as pd

OUT = "../results"
R = json.load(open(f"{OUT}/revision_analyses.json"))

# --- MLE comparison
L = [r"\begin{tabular}{lrrr}", r"\toprule", r"Comparison & Design-fixed & Binomial MLE & Difference \\", r"\midrule"]
for r in R["mle_comparison"]["rows"]:
    z = lambda x: "0" if abs(x) < 5e-6 else f"{x:+.5f}"      # the MLE is exactly zero for three comparisons
    L.append(f"{r['target'].replace(':', ' vs ')} & {z(r['design_fixed'])} & {z(r['mle'])} & {z(r['diff'])} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
open(f"{OUT}/supp_mle.tex", "w").write("\n".join(L) + "\n")

# --- efficiency
L = [r"\begin{tabular}{llrrrr}", r"\toprule",
     r"Network & Risk configuration & Arm risks & Min & Median & Max \\", r"\midrule"]
lab = {"study_pooled": "study-level pooled rate", "mle_floor_0.005": "constrained MLE, floored at 0.005"}
for net, v in R["efficiency"].items():
    for i, (k, e) in enumerate(v.items()):
        L.append(f"{net if i == 0 else ''} & {lab[k]} & {e['risk_range'][0]:.4f}--{e['risk_range'][1]:.3f} & "
                 f"{e['min']:.2f} & {e['median']:.2f} & {e['max']:.2f} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
open(f"{OUT}/supp_eff.tex", "w").write("\n".join(L) + "\n")

# --- cystic fibrosis sensitivity
cs = R["cf_sensitivity"]
L = [r"\begin{tabular}{llrrl}", r"\toprule",
     r"Analysis & Comparison & Estimate & Direct mass & Direct weights $d_k$ \\", r"\midrule"]
names = {"main": "Main (7 parallel-group trials)", "with_expand": "Adding EXPAND (periods as arms)",
         "exposure_weights": "Person-time weights (Remark 1)"}
for key in ["main", "with_expand", "exposure_weights"]:
    for i, (t, v) in enumerate(cs[key].items()):
        dd = ", ".join(f"{k} {x:.2f}" for k, x in v["d"].items())
        est = v["est"] if key != "exposure_weights" else v["est"]
        L.append(f"{names[key] if i == 0 else ''} & {t.replace(':', ' vs ')} & {est:+.4f} & {v['direct']:.2f} & {dd} \\\\")
    L.append(r"\addlinespace")
L[-1] = r"\bottomrule"
L.append(r"\end{tabular}")
open(f"{OUT}/supp_cfsens.tex", "w").write("\n".join(L) + "\n")

# --- registered primary endpoints (cystic fibrosis)
P = pd.read_csv("../data/registry_primary_endpoints.csv")
L = [r"\begin{tabular}{lllr}", r"\toprule", r"Trial & Comparison & Estimate (ppFEV$_1$) & $p$ \\", r"\midrule"]
for _, r in P.iterrows():
    comp = str(r["comparison"]).replace("&", "and").replace("%", r"\%")
    L.append(f"{r['study']} & {comp[:60]} & {r['estimate']} & {str(r['p_value']).replace('<', '$<$')} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
open(f"{OUT}/supp_primary.tex", "w").write("\n".join(L) + "\n")
print("wrote supp_mle.tex supp_eff.tex supp_cfsens.tex supp_primary.tex")
