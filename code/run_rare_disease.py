"""Rare-disease applications built from ClinicalTrials.gov posted results and published tables.

  1. NMOSD relapse prevention (data/nmosd2025_relapse.csv; John et al. 2025):
     contributions for six targets, CHAMPION single-arm / merged-into-PREVENT analysis,
     Fisher p-values, planned-ravulizumab-trial direct weights
     -> results/nmosd_summary.json, results/nmosd_table.tex (Table: NMOSD contributions)
  2. SMA safety, three nusinersen RCTs, pneumonia (data/sma_nusinersen_pneumonia.csv):
     contributions, exact SE, Q_het with exact null expectation and constrained-MLE
     calibration, moment tau, random-effects SE                -> results/sma_summary.json
  3. SMA efficacy and external controls (data/sma_efficacy_registry.csv):
     later-onset randomized network (CHERISH, SUNFISH) and infant-onset shared-baseline
     group (ENDEAR + single-arm STR1VE, FIREFISH)              -> results/sma_efficacy_summary.json
  4. Cystic fibrosis serious adverse events, 8 CFTR-modulator trials (data/cf_sae_registry.csv):
     contributions, routes and identification, Q decomposition with constrained-MLE
     calibration, Berger-Boos supremum p-value for T_W        -> results/cf_summary.json,
                                                                  results/cf_identification.csv
Numbers quoted in the text but not stored in these files are printed to stdout
(run_all.sh keeps them in results/run_rare_disease.log).

Usage:  python3 run_rare_disease.py [OUTDIR]     (OUTDIR defaults to ../results)
Run time: about 2-3 minutes (the cystic fibrosis calibration and supremum search dominate).
All Monte Carlo quantities use fixed seeds and reproduce the stored results exactly.
"""
import json
import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd

from rdcsp import (Network, binomial_interval, q_decomposition, binomial_calibration_mle,
                   null_expectation_Q, study_significance, route_table, berger_boos_sup_wald)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "results")
os.makedirs(OUT, exist_ok=True)
pd.set_option("display.width", 200)


def data(name):
    return os.path.join(DATA, name)


def out(name):
    return os.path.join(OUT, name)


# --------------------------------------------------------------------------- #
# 1. NMOSD relapse prevention
# --------------------------------------------------------------------------- #
def nmosd():
    print("\n=== NMOSD relapse prevention ===")
    df = pd.read_csv(data("nmosd2025_relapse.csv"))
    net = Network(df)
    print('connected', net.connected, 'treatments', net.trts, 'studies', net.studies)
    print('CHAMPION (single-arm) Laplacian zero?', np.abs(net.K['CHAMPION']).sum() == 0)
    res_out = {}
    for a, b in [('Control', 'Eculizumab'), ('Control', 'Inebilizumab'), ('Control', 'Satralizumab'),
                 ('Control', 'Rituximab'), ('Eculizumab', 'Inebilizumab'), ('Satralizumab', 'Eculizumab')]:
        res = net.contributions(a, b)
        ci = binomial_interval(net, a, b, B=20000, seed=3)
        sub = res[res.W > 1e-9][['study', 'd', 'W', 'C']].round(4)
        res_out[f'{a}:{b}'] = dict(est=float(res.attrs['estimate']), se=float(ci['exact_se']),   # full precision; rounded once when printed
                                   pct=[round(x, 4) for x in ci['percentile']],
                                   direct_mass=round(1 - res.attrs['q'], 3), rows=sub.to_dict('records'))
        print(a, b, 'RD', round(res.attrs['estimate'], 4), 'SE', round(ci['exact_se'], 4),
              'direct', round(1 - res.attrs['q'], 3))
        print(sub.to_string())
    q = q_decomposition(net)
    print('Q df net/het/inc', q['df_net'], q['df_het'], q['df_inc'], ' Q_het', round(q['Q_het'], 5),
          ' Q_inc', round(q['Q_inc'], 5))
    # CHAMPION merged with the PREVENT placebo arm, as the source review did
    df2 = df.copy()
    df2.loc[df2.study == 'CHAMPION', 'study'] = 'PREVENT'
    net2 = Network(df2)
    res2 = net2.contributions('Control', 'Ravulizumab')
    print('\nCHAMPION merged into PREVENT (3-arm): Control:Ravulizumab RD', round(res2.attrs['estimate'], 4),
          'SE', round(np.sqrt(net2.exact_variance('Control', 'Ravulizumab')), 4))
    print(res2[res2.W > 1e-9][['study', 'd', 'W', 'C']].round(4).to_string())
    res3 = net2.contributions('Control', 'Eculizumab')
    print('Control:Eculizumab with 3-arm PREVENT:', round(res3.attrs['estimate'], 4),
          '(was', round(net.contributions('Control', 'Eculizumab').attrs['estimate'], 4), ')')
    sig = study_significance(net)
    print(sig[['study', 'a', 'b', 'r_a', 'n_a', 'r_b', 'n_b', 'rd', 'fisher_p']].round(4).to_string())
    # planned randomized ravulizumab trial (Theorem: design-stage contribution)
    e = net2.e('Control', 'Ravulizumab')
    G = 1 / float(e @ net2.Lp @ e)
    for n in [50, 100, 200]:
        print('planned Control vs Ravulizumab n/arm', n, 'direct weight', round((n / 2) / ((n / 2) + G), 3))
    json.dump(res_out, open(out('nmosd_summary.json'), 'w'), indent=1)

    # LaTeX table (Table "NMOSD relapse prevention: study contributions")
    s = json.load(open(out('nmosd_summary.json')))
    lines = ["\\begin{tabular}{llrrrr}", "\\toprule",
             "Target (second minus first) & Study & $d_k$ & $\\tau_k$ & $C_k$ & Estimate (SE) \\\\", "\\midrule"]
    for t, v in s.items():
        first = True
        for r in v['rows']:
            lines.append(f"{t if first else ''} & {r['study']} & {r['d']:.3f} & {r['W']:.3f} & {r['C']:+.4f} & "
                         f"{('%.3f (%.3f)' % (v['est'], v['se'])) if first else ''} \\\\")
            first = False
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(out('nmosd_table.tex'), 'w').write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# 2. SMA safety: three nusinersen RCTs, pneumonia
# --------------------------------------------------------------------------- #
def sma_safety():
    print("\n=== SMA safety: nusinersen vs sham, pneumonia ===")
    net = Network(pd.read_csv(data('sma_nusinersen_pneumonia.csv')))
    res = net.contributions('Sham', 'Nusinersen')
    ci = binomial_interval(net, 'Sham', 'Nusinersen', B=20000, seed=4)
    print(res[['study', 'd', 'direct_est', 'C']].round(4).to_string())
    print('estimate', round(res.attrs['estimate'], 4), 'SE', round(ci['exact_se'], 4), 'pct', np.round(ci['percentile'], 4))
    q = q_decomposition(net)
    cal = binomial_calibration_mle(net, B=20000, seed=6)
    ex = null_expectation_Q(net, cal['p_null'])
    print('Q_het', round(q['Q_het'], 5), 'df', q['df_het'], 'E0', round(ex['E_Q_het'], 5), 'p', cal['p_het'], ' Q_inc df', q['df_inc'])
    sig = study_significance(net)
    print(sig[['study', 'r_a', 'n_a', 'r_b', 'n_b', 'rd', 'fisher_p']].round(4).to_string())
    # moment estimator of the between-trial variance
    sw = np.sqrt(net.w)
    Xd = net.design_specific()
    Xw = sw[:, None] * Xd
    Hd = Xw @ np.linalg.pinv(Xw.T @ Xw) @ Xw.T
    Ad = np.eye(len(sw)) - Hd
    # E Q_het = E0(p_bar) + sigma^2 sum_i A_ii w_i (1 - 1/n_i)   (Proposition 3; E{p(1-p)} = p_bar(1-p_bar) - sigma^2)
    tr = float(np.sum(np.diag(Ad) * net.w * (1 - 1 / net.n)))
    s2 = max(q['Q_het'] - ex['E_Q_het'], 0) / tr
    print('sigma_delta^2', s2, 'tau', np.sqrt(2 * s2))
    c = net.coefficient_vector('Sham', 'Nusinersen')
    v = (c ** 2 * net.phat * (1 - net.phat) / (net.n - 1)).sum()
    print('SE common', round(np.sqrt(v), 4), 'SE with tau^2', round(np.sqrt(v + s2 * (c ** 2).sum()), 4))
    json.dump(dict(est=res.attrs['estimate'], se=ci['exact_se'], pct=list(ci['percentile']),
                   d=res.set_index('study')['d'].round(4).to_dict(), Q_het=q['Q_het'], E0=ex['E_Q_het'],
                   p_het=cal['p_het'], tau=float(np.sqrt(2 * s2)), se_re=float(np.sqrt(v + s2 * (c ** 2).sum()))),
              open(out('sma_summary.json'), 'w'), indent=1)


# --------------------------------------------------------------------------- #
# 3. SMA efficacy: randomized later-onset network and infant-onset external controls
# --------------------------------------------------------------------------- #
def sma_efficacy():
    print("\n=== SMA efficacy and external controls ===")
    df = pd.read_csv(data('sma_efficacy_registry.csv'))
    res_out = {}
    # (A) later-onset randomized network: CHERISH, SUNFISH (control = sham/placebo)
    A = Network(df[df.study.isin(['CHERISH', 'SUNFISH'])][['study', 'treatment', 'events', 'n']])
    for a, b in [('Control', 'Nusinersen'), ('Control', 'Risdiplam'), ('Nusinersen', 'Risdiplam')]:
        r = A.contributions(a, b)
        ci = binomial_interval(A, a, b, B=20000, seed=2)
        res_out[f'A {a}:{b}'] = dict(est=round(r.attrs['estimate'], 4), se=round(ci['exact_se'], 4),
                                     pct=[round(x, 3) for x in ci['percentile']], direct=round(1 - r.attrs['q'], 3),
                                     rows=r[r.W > 1e-9][['study', 'd', 'W', 'C']].round(4).to_dict('records'))
        print('A', a, b, res_out[f'A {a}:{b}'])
    # (B) infant-onset: ENDEAR randomized; STR1VE and FIREFISH single-arm
    B0 = Network(df[df.study.isin(['ENDEAR', 'STR1VE', 'FIREFISH'])][['study', 'treatment', 'events', 'n']])
    print('B study-specific baselines: connected?', B0.connected, '| K zero for single-arm:',
          [k for k in B0.studies if np.abs(B0.K[k]).sum() == 0])
    # shared-baseline group: STR1VE and FIREFISH borrow the ENDEAR control arm (merged into one group study)
    dB = df[df.study.isin(['ENDEAR', 'STR1VE', 'FIREFISH'])][['study', 'treatment', 'events', 'n']].copy()
    dB['study'] = 'ENDEAR-group'
    B = Network(dB)
    for a, b in [('Control', 'Nusinersen'), ('Control', 'Onasemnogene'), ('Control', 'Risdiplam-infant'),
                 ('Nusinersen', 'Onasemnogene')]:
        r = B.contributions(a, b)
        se = np.sqrt(B.exact_variance(a, b))
        idx = B.rows['ENDEAR-group']
        arms = list(B.arm_trt[idx])
        cur = B.currents(a, b)['ENDEAR-group']
        # borrowed share: every comparison involving a single-arm treatment is carried by cross-study pairs
        borrowed = 1.0 if (b in ['Onasemnogene', 'Risdiplam-infant'] or a in ['Onasemnogene', 'Risdiplam-infant']) else 0.0
        res_out[f'B {a}:{b}'] = dict(est=round(r.attrs['estimate'], 4), se=round(se, 4), borrowed_share=borrowed,
                                     currents={u: round(float(c), 3) for u, c in zip(arms, cur)})
        print('B', a, b, res_out[f'B {a}:{b}'])
    # Fisher p-values of the randomized trials
    for net, name in [(A, 'A'), (Network(df[df.study == 'ENDEAR'][['study', 'treatment', 'events', 'n']]), 'ENDEAR')]:
        print(name, study_significance(net)[['study', 'r_a', 'n_a', 'r_b', 'n_b', 'rd', 'fisher_p']].round(4).to_string())
    # planned head-to-head nusinersen vs risdiplam (later-onset), 100 per arm
    e = A.e('Nusinersen', 'Risdiplam')
    r0 = float(e @ A.Lp @ e)
    g = 100 / 2
    d = A.e('Nusinersen', 'Risdiplam')
    ruv = float(d @ A.Lp @ d)
    print('later-onset Nusinersen:Risdiplam r_ab', round(r0, 5), 'reduction from 100/arm head-to-head',
          round(g * float(d @ (A.Lp @ e)) ** 2 / (1 + g * ruv), 5), 'direct weight of that trial', round(g / (g + 1 / r0), 3))
    json.dump(res_out, open(out('sma_efficacy_summary.json'), 'w'), indent=1)


# --------------------------------------------------------------------------- #
# 4. Cystic fibrosis: serious adverse events, network with loops
# --------------------------------------------------------------------------- #
def cystic_fibrosis():
    print("\n=== Cystic fibrosis: serious adverse events ===")
    df = pd.read_csv(data('cf_sae_registry.csv'))
    net = Network(df[['study', 'treatment', 'events', 'n']])
    print('connected', net.connected, 'T', net.T, 'studies', len(net.studies), 'arms', len(net.n),
          'events', int(net.r.sum()), 'n', int(net.n.sum()))
    q = q_decomposition(net)
    print('df net/het/inc', q['df_net'], q['df_het'], q['df_inc'])
    res_out = {}
    for a, b in [('Placebo', 'Triple'), ('TEZ/IVA', 'Triple'), ('Placebo', 'TEZ/IVA'), ('Placebo', 'Ivacaftor')]:
        r = net.contributions(a, b)
        ci = binomial_interval(net, a, b, B=20000, seed=3)
        d, rows, resid, ident = route_table(net, a, b)
        print(f"\n{a}:{b} RD {r.attrs['estimate']:.4f} SE {ci['exact_se']:.4f} direct mass {1 - r.attrs['q']:.3f} "
              f"routes {len(rows)} unique {ident['unique']} "
              f"maxwidth {max([x['upper'] - x['lower'] for x in rows], default=0):.3f}")
        print(r[r.W > 1e-9][['study', 'd', 'W', 'C']].round(4).to_string())
        for x in sorted(rows, key=lambda z: -z['weight'])[:4]:
            print('   route', round(x['weight'], 4), x['studies'], x['path'])
        res_out[f'{a}:{b}'] = dict(est=r.attrs['estimate'], se=ci['exact_se'], direct=1 - r.attrs['q'], n_routes=len(rows),
                                   unique=ident['unique'], rows=r[r.W > 1e-9][['study', 'd', 'W', 'C']].round(4).to_dict('records'),
                                   routes=[(round(x['weight'], 4), x['studies'], x['path'])
                                           for x in sorted(rows, key=lambda z: -z['weight'])[:4]])
    cal = binomial_calibration_mle(net, B=20000, seed=5)
    ex = null_expectation_Q(net, cal['p_null'])
    from rdcsp import wald_inconsistency_stat
    from scipy.stats import chi2 as _chi2
    t_w = float(wald_inconsistency_stat(net, net.phat[None, :])[0])   # plug-in and Berger-Boos p-values: run_qinc.py, sup_convergence.py
    bbw = dict(t_obs=t_w, df=int(q['df_inc']), chi2_p=float(_chi2.sf(t_w, int(q['df_inc']))))
    print('\nQ_het', round(q['Q_het'], 4), 'E0', round(ex['E_Q_het'], 4), 'p', cal['p_het'],
          '| Q_inc', round(q['Q_inc'], 4), 'E0', round(ex['E_Q_inc'], 4), 'p', cal['p_inc'],
          '| T_W', round(bbw['t_obs'], 3), 'df', bbw['df'], 'chi2 p', round(bbw['chi2_p'], 3))
    sig = study_significance(net)
    print(sig[['study', 'a', 'b', 'r_a', 'n_a', 'r_b', 'n_b', 'rd', 'fisher_p']].round(3).to_string())
    # evidence concentration and identification across all targets
    idf = []
    for x, y in combinations(net.trts, 2):
        d_, rs, _, idn = route_table(net, x, y)
        r_ = net.contributions(x, y)
        idf.append(dict(target=f'{x}:{y}', direct=round(1 - r_.attrs['q'], 3), routes=len(rs), unique=idn['unique'],
                        eff=round(r_['d'].sum() ** 2 / np.sum(r_['d'] ** 2), 2) if r_['d'].sum() > 0 else 0))
    idf = pd.DataFrame(idf)
    print(idf.to_string())
    idf.to_csv(out('cf_identification.csv'), index=False)
    res_out['Q'] = dict(Q_het=q['Q_het'], E0_het=ex['E_Q_het'], p_het=cal['p_het'], Q_inc=q['Q_inc'], E0_inc=ex['E_Q_inc'],
                        p_inc=cal['p_inc'], df_inc=int(q['df_inc']), df_het=int(q['df_het']), T_W=bbw['t_obs'],
                        chi2_p=bbw['chi2_p'])
    json.dump(res_out, open(out('cf_summary.json'), 'w'), indent=1, default=float)


if __name__ == "__main__":
    nmosd()
    sma_safety()
    sma_efficacy()
    cystic_fibrosis()
    print("\nwrote to", os.path.abspath(OUT))
