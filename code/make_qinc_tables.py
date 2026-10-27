"""Tables and macros for the consistency-testing results.

qinc_networks.tex (Supplementary Table): Q_inc with plug-in calibration at the constrained-MLE null,
two-moment approximation and Berger-Boos supremum; T_W with chi-square reference, plug-in calibration and
Berger-Boos supremum (largest search budget, tail re-estimated with independent draws; from
sup_convergence.py); Hoeffding bound.  qinc_macros.tex: values quoted in the main text.
qinc_sim.tex: the earlier floored-generator simulation (kept for the record only).
"""
import json

import pandas as pd

OUT = "../results"
r = json.load(open(f"{OUT}/qinc_networks.json"))
sc = json.load(open(f"{OUT}/supconv_summary.json"))
MISSING = dict(reported=float("nan"), largest_budget_range=[0, 0], p_min=0, p_max=0)
for _n in ["dong", "gurusamy", "cf", "covid"]:
    sc.setdefault(_n, MISSING)   # provisional only; the final run has all four
NETS = [("dong", "COPD"), ("gurusamy", "Liver transplant"), ("cf", "Cystic fibrosis"), ("covid", "COVID-19")]


def f3(x):
    return f"{x:.3f}"


with open(f"{OUT}/qinc_networks.tex", "w") as f:
    f.write("\\begin{tabular}{lrrrrrrrrrr}\n\\toprule\n & & \\multicolumn{4}{c}{$Q_{\\mathrm{inc}}$} & \\multicolumn{4}{c}{$T_W$} & \\\\\n"
            "\\cmidrule(lr){3-6}\\cmidrule(lr){7-10}\n")
    f.write("Network & df & value & plug-in & two-moment & BB sup & value & $\\chi^2$ & plug-in & BB sup & Hoeffding \\\\\n\\midrule\n")
    for name, lab in NETS:
        d = r[name]
        f.write(f"{lab} & {d['df_inc']} & {d['Q_inc']:.3f} & {f3(d['plugin_p'])} & {f3(d['satterthwaite_p'])} & {f3(d['bb_p'])} & "
                f"{d['wald_T']:.2f} & {f3(d['wald_chi2_p'])} & {f3(d['wald_plugin_p'])} & {f3(sc[name]['reported'])} & "
                f"{f3(d['hoeffding_p'])} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

s = pd.read_csv(f"{OUT}/qinc_sim.csv")
with open(f"{OUT}/qinc_sim.tex", "w") as f:
    f.write("\\begin{tabular}{lrrrrrrrrr}\n\\toprule\n & & \\multicolumn{3}{c}{$Q_{\\mathrm{inc}}$} & \\multicolumn{3}{c}{$T_W$} & & \\\\\n"
            "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\n")
    f.write("Rates & $\\delta$ & plug-in & two-moment & BB sup & plug-in & $\\chi^2$ & BB sup & Hoeffding & fiducial \\\\\n\\midrule\n")
    for _, x in s.iterrows():
        f.write(f"{x['rate']} & {x['delta']:.2f} & {x['plugin']:.3f} & {x['satterthwaite']:.3f} & {x['bb']:.3f} & "
                f"{x['wald_plugin']:.3f} & {x['wald_chi2']:.3f} & {x['wald_bb']:.3f} & {x['hoeffding']:.3f} & {x['fiducial']:.3f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

macros = {}
for name, key in [("dong", "Dong"), ("gurusamy", "Gur"), ("covid", "Covid"), ("cf", "Cf")]:
    macros[f"WT{key}"] = f"{r[name]['wald_T']:.2f}"
    macros[f"Wchi{key}"] = f3(r[name]["wald_chi2_p"])
    macros[f"Wplug{key}"] = f3(r[name]["wald_plugin_p"])
    macros[f"Wbb{key}"] = f3(sc[name]["reported"])
    macros[f"Qplug{key}"] = f3(r[name]["plugin_p"])
    macros[f"Qbb{key}"] = f3(r[name]["bb_p"])
    macros[f"Qsatt{key}"] = f3(r[name]["satterthwaite_p"])
spread = max(max(v["largest_budget_range"]) - min(v["largest_budget_range"]) for v in sc.values())
spread_all = max(v["p_max"] - v["p_min"] for v in sc.values())
macros["SupconvSpread"] = f"{spread_all:.3f}"
with open(f"{OUT}/qinc_macros.tex", "w") as f:
    for k, v in macros.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(open(f"{OUT}/qinc_networks.tex").read())
print(macros, "largest-budget spread", spread)
