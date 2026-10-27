# Exact study contributions in network meta-analysis of rare events

This repository holds the code and data for the paper "What Does Each Trial Contribute? Exact Study Contributions in Network Meta-Analysis of Rare Events". The scripts regenerate every table, figure and number in the paper and its supplement.

The method works with binomial arms on the risk-difference scale, using design-fixed (arm-size) weights. The network estimator is then linear in the observed proportions. Each trial's contribution is a fixed coefficient vector times its own arm proportions, the coefficients depend only on the design, and the estimator is exactly unbiased with an exact binomial law. Cochran's Q splits exactly into heterogeneity and inconsistency, and consistency is tested by plug-in binomial calibration and a Berger–Boos unconditional test.

## Quick start

```
pip install -r requirements.txt          # numpy 2.4.4, scipy 1.17.1, pandas 3.0.5, matplotlib 3.11.2
bash code/run_all.sh                     # about two hours on two cores; see the script for per-step times
```

`run_all.sh` writes everything to `results/` (CSV, JSON, LaTeX tables, PDF figures, logs) and the LaTeX macros `results/qinc_macros.tex` and `results/text_macros.tex`. The manuscript reads from these macros every number that combines several results files. All Monte Carlo steps use fixed seeds. With the pinned library versions, the stored results are reproduced exactly. With other versions of numpy and scipy, deterministic outputs agree to floating-point rounding, and a few Monte Carlo p-values move by a handful of draws.

To use the method on your own arm-level data:

```python
import pandas as pd
from rdcsp import Network          # run from code/
net = Network(pd.read_csv("my_network.csv"))      # columns: study, treatment, events, n
res = net.contributions("Placebo", "Drug")        # one row per study: d (direct weight), W (throughput), C_D, C_I, C
res.attrs["estimate"], res.attrs["q"]             # risk difference Drug - Placebo; indirect mass
```

## Code (`code/`)

| Script | What it does | Main outputs |
|---|---|---|
| `rdcsp.py` | The method: `Network` (Laplacians, currents, contributions, exact variance), Q decomposition, exact moments, constrained-MLE null, plug-in calibration, Berger–Boos supremum tests, Hoeffding bounds, intervals, routes and identification, planned trials, comparators | library |
| `run_applications.py` | COPD, liver transplant and COVID-19 networks | `dong_targets.tex`, `gur_control_aprotinin_studies.tex`, `fig_dong_contributions.pdf`, `summary.json` |
| `add_route_outputs.py` | COPD routes, identification, flow figure, p-value vs weight figure, SE(C_k) table | `dong_routes.tex`, `dong_identification.tex`, `fig_dong_flow.pdf`, `fig_dong_pvalue_weight.pdf`, `dong_placebo_tiohh_studies_se.tex` |
| `extras.py` | Competing methods (shortest-path contributions, leave-one-out importance, Mantel–Haenszel exclusions), continuity-correction table, allocation | `cc_instability.tex`, `extras.json` |
| `dong_precision_shares.py` | Precision shares, COPD | `dong_precision_shares.tex` |
| `informativeness.py` | Design-based precision of the network vs the best single trial | `informativeness.csv/.json`, `fig_informativeness.pdf` |
| `fig_significance.py` | Significance, size and weight across four networks | `significance_size_weight.csv/.json`, `fig_significance_size_weight.pdf` |
| `run_rare_disease.py` | NMOSD, SMA safety, SMA efficacy (external controls), cystic fibrosis | `nmosd_table.tex`, `nmosd_summary.json`, `sma_*.json`, `cf_summary.json`, `cf_identification.csv` |
| `registry_checks.py` | Weights from registered enrolment and from the enrolment anticipated at first registration | `nmosd_registry_check.csv`, `registry_design_check.csv/.tex` |
| `registry_primary.py` | Registered primary-endpoint results (needs network access; output included) | `data/registry_primary_endpoints.csv` |
| `revision_analyses.py` | Leave-one-out on the design scale, effective number of trials, allocation, design precision, efficiency, binomial MLE, TIOSPIR, cystic fibrosis sensitivity, registry agreement, moment estimator | `revision_analyses.json`, `design_precision.csv` |
| `calibration_mle.py` | Plug-in calibration of Q at the constrained-MLE null | `calibration_mle.json` |
| `run_qinc.py` | Consistency tests on the networks (no arguments) | `qinc_networks.json` |
| `sup_convergence.py` | Berger–Boos supremum for T_W at three budgets and three seeds, tail re-estimated with fresh draws | `supconv_*.json`, `supconv.tex`, `supconv_summary.json` |
| `run_simulations.py`, `make_tables.py` | Synthetic simulation, estimator | `sim_partA.csv`, `sim_partB.csv` |
| `run_sim_mle.py`, `make_sim_mle_table.py` | Synthetic simulation, consistency tests with the constrained-MLE null | `sim_mle.csv/.tex` |
| `sim_cf_design.py` | Design-based simulation on the cystic fibrosis network | `sim_cf_design.csv` |
| `make_sim_table.py` | Table 1 and Supplementary Table S(simA) | `sim_main.tex`, `sim_partA_supp.tex` |
| `make_qinc_tables.py`, `make_text_macros.py`, `make_supp_tables.py` | LaTeX tables and macros for the manuscript and supplement | `qinc_networks.tex`, `qinc_macros.tex`, `text_macros.tex`, `supp_*.tex` |
| `verify_math.py` | Numerical checks of every identity and closed form | `verify_math.json` |
| `figstyle.py` | Shared figure style | — |
| `run_all.sh` | Runs everything in dependency order | — |

## Data (`data/`)

Every arm-level file has the columns `study, treatment, events, n` (number with the event, number randomized). Some files add `source` or `outcome`.

| File | Network | Source |
|---|---|---|
| `dong2013_copd_mortality.csv` | COPD, death during follow-up; 41 trials, 6 treatments | R package `netmeta`, data set `Dong2013` |
| `gurusamy2011_liver_transplant_mortality.csv` | Liver transplantation, death; 14 trials, 7 treatments | `netmeta`, data set `Gurusamy2011` (label "Aprotonin" corrected to "Aprotinin") |
| `nmosd2025_relapse.csv` | NMOSD, relapse; 6 randomized trials and 1 single-arm trial | trial reports summarized by John et al. (2025), *J Neurol* 272:563 |
| `sma_nusinersen_pneumonia.csv` | SMA, pneumonia; 3 nusinersen vs sham trials | Abbas et al. (2022), *Medicina* 58:213 |
| `sma_efficacy_registry.csv` | SMA, motor-function responders; CHERISH, SUNFISH, ENDEAR, STR1VE, FIREFISH | ClinicalTrials.gov posted results |
| `cf_sae_registry.csv` | Cystic fibrosis, ≥1 serious adverse event; 7 parallel-group trials (main analysis) | ClinicalTrials.gov adverse-event modules |
| `cf_sae_registry_with_expand.csv` | The same plus the crossover trial EXPAND (sensitivity analysis only) | ClinicalTrials.gov |
| `covid19_wholenetwork.csv` | COVID-19, death; 5 trials, treatments A–E (standard care, corticosteroids, lopinavir–ritonavir, hydroxychloroquine, remdesivir) | Wang et al., arXiv:2604.21994 |
| `registry_history.csv` | First anticipated and last actual enrolment of 19 registered trials, registered allocation ratio | ClinicalTrials.gov record histories, read by hand (the history endpoint is not scriptable) |
| `registry_primary_endpoints.csv` | Registered primary-endpoint analyses of the cystic fibrosis trials | ClinicalTrials.gov API v2 (`registry_primary.py`) |

ClinicalTrials.gov pages were accessed between 25 and 27 October 2026.

## License

The code is released under the MIT License (see `LICENSE`). The data files are derived from the public sources listed above.
