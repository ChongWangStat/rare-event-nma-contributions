"""Design weights from the trial registry alone (no outcomes needed).

  1. NMOSD: direct weights and throughputs recomputed from the ClinicalTrials.gov enrolment
     count and registered allocation ratio of the five registered randomized trials, against
     those from the published arm sizes             -> results/nmosd_registry_check.csv
  2. All three rare-disease networks (NMOSD, cystic fibrosis, SMA later-onset): direct weights
     and throughputs recomputed from the enrolment anticipated at first registration and from
     the final (actual) registered enrolment, against the published arm sizes
     (input: data/registry_history.csv)              -> results/registry_design_check.csv,
                                                         results/registry_design_check.tex (Table: registry)

The enrolment and allocation figures in (1) are the ClinicalTrials.gov API v2 values read on
25 Oct 2026 (they equal the last_enrollment / ratio columns of data/registry_history.csv).

Usage:  python3 registry_checks.py [OUTDIR]     (OUTDIR defaults to ../results)
Run time: a few seconds.
"""
import os
import sys

import numpy as np
import pandas as pd

from rdcsp import Network

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
# 1. NMOSD: current registry enrolment + allocation ratio vs published arm sizes
# --------------------------------------------------------------------------- #
def nmosd_registry_check():
    print("\n=== NMOSD: registry enrolment and allocation ratio vs published arm sizes ===")
    pub = pd.read_csv(data('nmosd2025_relapse.csv'))
    pub = pub[pub.study != 'CHAMPION']
    # registry (ClinicalTrials.gov v2 API, 25 Oct 2026): actual enrolment and allocation ratio
    reg = {'PREVENT': (143, 'Eculizumab', 2, 1), 'N-MOmentum': (231, 'Inebilizumab', 3, 1),
           'SAkuraSky': (85, 'Satralizumab', 1, 1), 'SAkuraStar': (95, 'Satralizumab', 2, 1),
           'TANGO': (118, 'Tocilizumab', 1, 1)}
    rows = []
    for k, g in pub.groupby('study', sort=False):
        if k in reg:
            N, trt, ra, rb = reg[k]
            n_t = round(N * ra / (ra + rb))
            n_c = N - n_t
            for t, n in zip(g.treatment, g.n):
                rows.append(dict(study=k, treatment=t, events=0, n=n_t if t == trt else n_c))
        else:
            for t, n in zip(g.treatment, g.n):
                rows.append(dict(study=k, treatment=t, events=0, n=n))
    regnet = Network(pd.DataFrame(rows))
    pubnet = Network(pub)
    res = []
    for t in ['Eculizumab', 'Inebilizumab', 'Satralizumab', 'Tocilizumab', 'Rituximab']:
        r1 = pubnet.contributions('Control', t)
        r2 = regnet.contributions('Control', t)
        m = r1.merge(r2, on='study', suffixes=('_pub', '_reg'))
        m = m[m.d_pub > 0]
        for _, r in m.iterrows():
            res.append(dict(target=f'Control:{t}', study=r['study'], d_published=round(r['d_pub'], 4),
                            d_registry=round(r['d_reg'], 4)))
    # indirect target: throughput of each trial for eculizumab vs inebilizumab
    tgt = ('Eculizumab', 'Inebilizumab')
    r1 = pubnet.contributions(*tgt)
    r2 = regnet.contributions(*tgt)
    m = r1.merge(r2, on='study', suffixes=('_pub', '_reg'))
    for _, r in m[m.W_pub > 0].iterrows():
        res.append(dict(target='Eculizumab:Inebilizumab (throughput)', study=r['study'],
                        d_published=round(r['W_pub'], 4), d_registry=round(r['W_reg'], 4)))
    df = pd.DataFrame(res)
    print(df.to_string())
    df.to_csv(out('nmosd_registry_check.csv'), index=False)
    print('max abs diff', (df.d_published - df.d_registry).abs().max())


# --------------------------------------------------------------------------- #
# 2. Anticipated (first registered version) vs actual vs published, three networks
# --------------------------------------------------------------------------- #
def registry_design_check():
    print("\n=== Registration history: anticipated vs realized direct weights ===")
    reg = pd.read_csv(data('registry_history.csv'))
    nets = {'nmosd': ('nmosd2025_relapse.csv', 'Control'), 'cf': ('cf_sae_registry.csv', 'Placebo'),
            'sma': ('sma_efficacy_registry.csv', 'Control')}
    rows = []
    for netname, (f, ctrl) in nets.items():
        pub = pd.read_csv(data(f))[['study', 'treatment', 'events', 'n']]
        if netname == 'sma':
            pub = pub[pub.study.isin(['CHERISH', 'SUNFISH'])]
        if netname == 'nmosd':
            pub = pub[pub.study != 'CHAMPION']

        def rebuilt(col):
            d = pub.copy()
            for k, g in pub.groupby('study'):
                r = reg[reg.study == k]
                if len(r) == 0 or r.ratio.iloc[0] == 'single':
                    continue
                N = int(r[col].iloc[0])
                parts = [int(x) for x in r.ratio.iloc[0].split(':')]
                arms = list(g.treatment)
                # allocation: equal parts -> equal arms; otherwise treatment:control with control last
                if len(set(parts)) == 1:
                    ns = [N / len(arms)] * len(arms)
                else:
                    ns = [N * (parts[1] if t == ctrl else parts[0]) / sum(parts) for t in arms]
                d.loc[d.study == k, 'n'] = [round(x) for x in ns]
            return Network(d)

        pubnet = Network(pub)
        regA = rebuilt('first_enrollment')
        regL = rebuilt('last_enrollment')
        from itertools import combinations as _comb
        # every comparison (not only those against the control), so active-comparator trials are included
        targets = [(ctrl, t) for t in pubnet.trts if t != ctrl] + [(a, b) for a, b in _comb(pubnet.trts, 2) if ctrl not in (a, b)]
        for a, b in targets:
            r0 = pubnet.contributions(a, b)
            r1 = regA.contributions(a, b)
            r2 = regL.contributions(a, b)
            m = r0.merge(r1, on='study', suffixes=('', '_ant')).merge(r2, on='study', suffixes=('', '_act'))
            for _, x in m[m.W > 1e-9].iterrows():
                rows.append(dict(network=netname, target=f'{a}:{b}', study=x['study'], d_pub=round(x['d'], 3),
                                 d_anticipated=round(x['d_ant'], 3), d_actual=round(x['d_act'], 3),
                                 W_pub=round(x['W'], 3), W_anticipated=round(x['W_ant'], 3),
                                 W_actual=round(x['W_act'], 3)))
    df = pd.DataFrame(rows)
    df.to_csv(out('registry_design_check.csv'), index=False)
    print(df.to_string())
    dd = df[df.d_pub > 0]
    print('\nmax |d_anticipated - d_pub|', (dd.d_anticipated - dd.d_pub).abs().max(),
          ' mean', (dd.d_anticipated - dd.d_pub).abs().mean(), ' corr', np.corrcoef(dd.d_anticipated, dd.d_pub)[0, 1])
    print('max |d_actual - d_pub|', (dd.d_actual - dd.d_pub).abs().max())
    print('max |W_anticipated - W_pub|', (df.W_anticipated - df.W_pub).abs().max(),
          ' corr', np.corrcoef(df.W_anticipated, df.W_pub)[0, 1])
    reg['ratio_actual_over_anticipated'] = (reg.last_enrollment / reg.first_enrollment).round(2)
    print(reg[['study', 'first_enrollment', 'last_enrollment', 'ratio_actual_over_anticipated']].to_string())

    # LaTeX table (Table "registry": anticipated vs realized direct weights, direct trials only)
    df = pd.read_csv(out('registry_design_check.csv'))
    lines = ["\\begin{tabular}{llrrrr}", "\\toprule",
             "Network & Target & Study & $d_k$ anticipated & $d_k$ realized & Anticipated/realized enrolment \\\\",
             "\\midrule"]
    for _, x in df[df.d_pub > 0].iterrows():
        r = reg[reg.study == x['study']]
        enr = f"{int(r.first_enrollment.iloc[0])}/{int(r.last_enrollment.iloc[0])}" if len(r) else "not in ClinicalTrials.gov"
        lines.append(f"{ {'nmosd': 'NMOSD', 'cf': 'CF', 'sma': 'SMA'}[x['network']] } & {x['target']} & {x['study']} & "
                     f"{x['d_anticipated']:.3f} & {x['d_pub']:.3f} & {enr} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(out('registry_design_check.tex'), 'w').write("\n".join(lines) + "\n")


if __name__ == "__main__":
    nmosd_registry_check()
    registry_design_check()
    print("\nwrote to", os.path.abspath(OUT))
