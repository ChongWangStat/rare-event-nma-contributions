#!/usr/bin/env bash
# Regenerate every table, figure and number in results/ from data/, in dependency order.
# Run from anywhere:  bash code/run_all.sh      (scripts are run from inside code/)
# Each step's stdout is kept in results/<script>.log; numbers quoted in the text but not stored
# in a results file are printed there.
# Long Monte Carlo steps are marked LONG with approximate run times (2-core machine); the
# total is roughly two hours, dominated by sup_convergence.py dong and the simulations.
# Every Monte Carlo step uses fixed seeds.  The stored results were produced with Python 3.13,
# numpy 2.4.4, scipy 1.17.1, pandas 3.0.5 (requirements.txt).  With numpy 2.5.3 / scipy 1.18.1
# a few Monte Carlo p-values move by a handful of draws; deterministic outputs agree to
# floating-point rounding (< 1e-13).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
R=../results
mkdir -p "$R/figures"

# --- Optional: refresh registry-derived data (needs network access; overwrites data/) ---------
# $PY registry_primary.py                       # data/registry_primary_endpoints.csv (CT.gov API v2)
# data/registry_history.csv was read by hand from the ClinicalTrials.gov record-history pages
# (api/int/studies/{NCT}/history); that endpoint is not reachable from a script (HTTP 403).

# --- Applications: COPD, liver transplant, COVID-19 ------------------------------------------
$PY run_applications.py      | tee "$R/run_applications.log"      # ~2 min; tables, figures, summary.json
$PY add_route_outputs.py     | tee "$R/add_route_outputs.log"     # needs dong_significance_vs_contribution.csv
$PY extras.py                | tee "$R/extras.log"                # cc_instability, loo, evidence concentration, allocation
$PY informativeness.py       | tee "$R/informativeness.log"
$PY fig_significance.py      | tee "$R/fig_significance.log"
$PY verify_math.py           | tee "$R/verify_math.log"           # verify_math.json
$PY dong_precision_shares.py | tee "$R/dong_precision_shares.log" # dong_precision_shares.tex
$PY calibration_mle.py       | tee "$R/calibration_mle.log"       # ~3 min; calibration_mle.json

# --- Rare-disease networks and registry checks ---------------------------------------------
$PY run_rare_disease.py      | tee "$R/run_rare_disease.log"      # ~2-3 min; nmosd_summary.json, nmosd_table.tex,
                                                                  #   sma_summary.json, sma_efficacy_summary.json,
                                                                  #   cf_summary.json, cf_identification.csv
$PY registry_checks.py       | tee "$R/registry_checks.log"       # nmosd_registry_check.csv,
                                                                  #   registry_design_check.csv/.tex
$PY revision_analyses.py     | tee "$R/revision_analyses.log"     # ~2 min; revision_analyses.json, design_precision.csv
                                                                  #   (needs registry_design_check.csv, significance_size_weight.csv)

# --- Simulation study (estimator bias/coverage, Q calibration) ------------------------------
# LONG: ~14 min (sim_summary.json: Part A 436 s, Part B 398 s)
$PY run_simulations.py       | tee "$R/run_simulations.log"       # sim_partA.csv, sim_partB.csv, sim_summary.json
$PY make_tables.py           | tee "$R/make_tables.log"           # sim_partA.tex, sim_partB.tex

# --- Consistency tests: networks and simulated level/power ----------------------------------
$PY run_qinc.py              | tee "$R/run_qinc.log"              # ~3 min; qinc_networks.json (constrained-MLE null)
# LONG: simulation cells, ~2-5 min each (not logged; estimated from one replicate); rows are
# appended to qinc_sim.csv in this order, which is the row order of qinc_sim.tex
$PY run_qinc.py low 0.0      | tee -a "$R/run_qinc_cells.log"
$PY run_qinc.py low 0.06     | tee -a "$R/run_qinc_cells.log"
$PY run_qinc.py verylow 0.0  | tee -a "$R/run_qinc_cells.log"
$PY run_qinc.py verylow 0.06 | tee -a "$R/run_qinc_cells.log"

# LONG: Berger-Boos supremum convergence (supplementary table supconv); independent, can run in parallel.
# Measured with all four running in parallel on 2 cores: gurusamy ~5 min, cf ~8 min, covid ~9 min,
# dong ~30 min
$PY sup_convergence.py gurusamy > "$R/supconv_gurusamy.log"
$PY sup_convergence.py covid    > "$R/supconv_covid.log"
$PY sup_convergence.py dong     > "$R/supconv_dong.log"
$PY sup_convergence.py cf       > "$R/supconv_cf.log"
$PY sup_convergence.py table    | tee "$R/supconv_table.log"      # supconv.tex, supconv_summary.json
$PY make_qinc_tables.py      | tee "$R/make_qinc_tables.log"      # qinc_networks.tex, qinc_macros.tex (needs supconv_summary.json)

# --- Consistency-test simulation with the constrained-MLE null generator (sim_mle.tex) ------
# LONG-ish: ~0.5-1 min per cell (time_s column of sim_mle.csv: 44, 34, 26, 23 s)
# LONG-ish: ~5 min per cell with the supremum test on 400 / 200 data sets
$PY run_sim_mle.py low 0.0 1000 400      | tee -a "$R/run_sim_mle.log"
$PY run_sim_mle.py verylow 0.0 1000 400  | tee -a "$R/run_sim_mle.log"
$PY run_sim_mle.py low 0.06 1000 200     | tee -a "$R/run_sim_mle.log"
$PY run_sim_mle.py verylow 0.06 1000 200 | tee -a "$R/run_sim_mle.log"
$PY make_sim_mle_table.py            | tee "$R/make_sim_mle_table.log"   # sim_mle.tex

# --- Design-based simulation on the cystic fibrosis network (sim_main.tex) -----------------
# ~4-8 min per cell
$PY sim_cf_design.py 0.0 1000 400   | tee -a "$R/sim_cf_design.log"
$PY sim_cf_design.py 0.05 1000 200  | tee -a "$R/sim_cf_design.log"
$PY make_sim_table.py               | tee "$R/make_sim_table.log"      # sim_main.tex, sim_partA_supp.tex

# --- Robustness: interval coverage, misspecification, NMOSD allocation ----------------------
$PY nmosd_allocation.py             | tee "$R/nmosd_allocation.log"    # ~1 min; nmosd_allocation.json/.tex
# ~7 min (4,000 data sets per design; the e-value interval is computed by bisection)
$PY interval_coverage.py 4000       | tee "$R/interval_coverage.log"   # interval_coverage.csv/.tex
# LONG: ~6-7 min each; the two truths are independent and can run in parallel
$PY sim_misspec.py additive 2000    | tee "$R/sim_misspec_additive.log"  # sim_misspec_additive.csv
$PY sim_misspec.py logit 2000       | tee "$R/sim_misspec_logit.log"     # sim_misspec_logit.csv
$PY make_robustness_tables.py       | tee "$R/make_robustness_tables.log" # misspec.tex

# --- Checks quoted in the supplement ----------------------------------------------------------
$PY check_constrained_mle.py        | tee "$R/check_constrained_mle.log"   # SLSQP vs interior point
$PY supplementary_checks.py         | tee "$R/supplementary_checks.log"    # ~30 s; supplementary_checks.json

# --- Tables and macros read by the manuscript and supplement --------------------------------
$PY make_text_macros.py             | tee "$R/make_text_macros.log"    # text_macros.tex
$PY make_supp_tables.py             | tee "$R/make_supp_tables.log"    # supp_mle/eff/cfsens/primary.tex
if [ -f ../manuscript/main_acs.tex ]; then $PY make_blind.py; fi       # manuscript/main_acs_blind.tex (review package only)

echo "All results regenerated in $(cd "$R" && pwd)"
