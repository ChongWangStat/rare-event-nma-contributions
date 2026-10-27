"""Turn simulation CSVs into LaTeX tables for the manuscript."""
import pandas as pd, numpy as np
OUT="../results"
a=pd.read_csv(f"{OUT}/sim_partA.csv")
with open(f"{OUT}/sim_partA.tex","w") as f:
    f.write("\\begin{tabular}{llrrrrrrrr}\n\\toprule\nRates & Target & True RD & Bias & Exact SD & Emp.\\ SD & IV+cc bias & Cov.\\ basic & Cov.\\ normal & Cov.\\ IV+cc \\\\\n\\midrule\n")
    for _,r in a.iterrows():
        f.write(f"{r['rate']} & {r['target']} & {r['truth']:.4f} & {r['bias']:+.5f} & {r['exact_sd']:.4f} & {r['empirical_sd']:.4f} & {r['iv_bias']:+.4f} & {r['cov_basic']:.3f} & {r['cov_exactvar_normal']:.3f} & {r['cov_iv_cc']:.3f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
b=pd.read_csv(f"{OUT}/sim_partB.csv")
with open(f"{OUT}/sim_partB.tex","w") as f:
    f.write("\\begin{tabular}{lrrrrrrr}\n\\toprule\n & & \\multicolumn{3}{c}{$Q_{\\mathrm{inc}}$} & \\multicolumn{3}{c}{$Q_{\\mathrm{het}}$} \\\\\n\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\nRates & $\\delta$ & Fixed, binomial & Pearson, $\\chi^2$ & Pearson, binomial & Fixed, binomial & Pearson, $\\chi^2$ & Pearson, binomial \\\\\n\\midrule\n")
    for _,r in b.iterrows():
        f.write(f"{r['rate']} & {r['delta']:.2f} & {r['fixed_binom_inc']:.3f} & {r['pearson_chi2_inc']:.3f} & {r['pearson_binom_inc']:.3f} & {r['fixed_binom_het']:.3f} & {r['pearson_chi2_het']:.3f} & {r['pearson_binom_het']:.3f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
print(open(f"{OUT}/sim_partA.tex").read()); print(open(f"{OUT}/sim_partB.tex").read())
