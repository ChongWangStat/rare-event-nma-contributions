"""LaTeX table of the consistency-test simulation with the constrained-MLE null generator
(Supplementary Table S3) from results/sim_mle.csv, which run_sim_mle.py fills one cell at a time.

  ../results/sim_mle.csv  ->  results/sim_mle.tex

Usage:  python3 make_sim_mle_table.py [OUTDIR]  (reads sim_mle.csv from ../results; OUTDIR defaults to ../results)
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
OUT = sys.argv[1] if len(sys.argv) > 1 else RESULTS
os.makedirs(OUT, exist_ok=True)

if __name__ == "__main__":
    d = pd.read_csv(os.path.join(RESULTS, 'sim_mle.csv')).sort_values(['rate', 'delta'])
    pd.set_option('display.width', 200)
    print(d.round(3).to_string())
    with open(os.path.join(OUT, 'sim_mle.tex'), 'w') as f:
        f.write("\\begin{tabular}{lrrrrrrrr}\n\\toprule\n & & \\multicolumn{2}{c}{$Q_{\\mathrm{inc}}$} & "
                "\\multicolumn{3}{c}{$T_W$} & $Q_{\\mathrm{het}}$ & Pearson \\\\\n\\cmidrule(lr){3-4}\\cmidrule(lr){5-7}\n"
                "Rates & $\\delta$ & plug-in & two-moment & plug-in & $\\chi^2$ & BB sup$^{\\dagger}$ & plug-in & $\\chi^2$ \\\\\n"
                "\\midrule\n")
        for _, x in d.iterrows():
            f.write(f"{x['rate']} & {x['delta']:.2f} & {x['fixed_mle_inc']:.3f} & {x['satt_mle_inc']:.3f} & "
                    f"{x['wald_plugin_mle']:.3f} & {x['wald_chi2']:.3f} & {x['wald_bb_mle']:.3f} & "
                    f"{x['fixed_mle_het']:.3f} & {x['pearson_chi2_inc']:.3f} \\\\\n")
        f.write("\\bottomrule\n" + "\\end{tabular}\n")
