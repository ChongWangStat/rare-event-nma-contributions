"""Competing methods, continuity-correction instability, evidence concentration,
and optimal allocation computations for the ACS revision."""
import numpy as np, pandas as pd, json, heapq
from itertools import combinations
from rdcsp import Network
from scipy.optimize import minimize
OUT = "../results"; R = {}
dong = Network(pd.read_csv("../data/dong2013_copd_mortality.csv"))
a, b = "Placebo", "TIO-HH"

# ---------- 1. IV NMA on a chosen scale with continuity correction; CSP-type weights via arm-based covariance ----
def iv_network(df, scale="rd", cc=0.5, cc_rule="constant"):
    """Return a Network whose arm statistics are the chosen scale with continuity correction and
    inverse-variance arm weights (plug-in).  cc_rule: 'constant' adds cc to r and 2cc to n in studies with
    a zero cell; 'armsize' uses the treatment-arm correction of Sweeting et al. (cc proportional to the
    reciprocal of the opposite arm size, total 1 per study)."""
    d = df.copy(); z = np.zeros(len(d)); w = np.zeros(len(d))
    for k, g in d.groupby("study", sort=False):
        idx = g.index; r = g.events.to_numpy(float); n = g.n.to_numpy(float)
        if np.any(r == 0) or np.any(r == n):
            if cc_rule == "constant":
                r = r + cc; n = n + 2 * cc
            else:
                tot = n.sum(); add = (tot - n) / tot * len(n) * cc   # reciprocal-of-opposite-arm, scaled to sum len*cc
                r = r + add; n = n + 2 * add
        p = r / n
        if scale == "rd":
            z[idx] = p; w[idx] = n / (p * (1 - p))
        else:
            z[idx] = np.log(p / (1 - p)); w[idx] = n * p * (1 - p)
    net = Network(d, weights=w); net.phat = z
    return net
rows = []
for scale, cc, rule in [("rd", 0.5, "constant"), ("logit", 0.5, "constant"), ("logit", 0.25, "constant"), ("logit", 0.5, "armsize"), ("logit", 1.0, "constant")]:
    net = iv_network(dong.df, scale, cc, rule); res = net.contributions(a, b)
    top = res.sort_values("d", ascending=False).head(4)
    rows.append(dict(scale=scale, cc=cc, rule=rule, estimate=res.attrs["estimate"], q=res.attrs["q"],
                     d_study3=float(res.loc[res.study == "3", "d"].iloc[0]), d_study10=float(res.loc[res.study == "10", "d"].iloc[0]),
                     d_study36=float(res.loc[res.study == "36", "d"].iloc[0]), W_study12=float(res.loc[res.study == "12", "W"].iloc[0]),
                     n_zero_cc_studies=int(sum((g.events == 0).any() for _, g in dong.df.groupby("study")))))
rd0 = dong.contributions(a, b)
rows.insert(0, dict(scale="rd (design-fixed, this paper)", cc=0, rule="none", estimate=rd0.attrs["estimate"], q=rd0.attrs["q"],
                    d_study3=float(rd0.loc[rd0.study == "3", "d"].iloc[0]), d_study10=float(rd0.loc[rd0.study == "10", "d"].iloc[0]),
                    d_study36=float(rd0.loc[rd0.study == "36", "d"].iloc[0]), W_study12=float(rd0.loc[rd0.study == "12", "W"].iloc[0]), n_zero_cc_studies=0))
cc_tab = pd.DataFrame(rows); cc_tab.to_csv(f"{OUT}/cc_instability.csv", index=False); R["cc"] = cc_tab.to_dict("records")
with open(f"{OUT}/cc_instability.tex", "w") as f:
    f.write("\\begin{tabular}{llrrrrrr}\n\\toprule\nScale & Correction & Estimate & $q$ & $d_3$ & $d_{10}$ & $d_{36}$ & $\\tau_{12}$ \\\\\n\\midrule\n")
    for r in rows:
        lab = {"none": "none", "constant": f"{r['cc']} to every cell of a zero-cell study", "armsize": "treatment-arm (Sweeting)"}[r["rule"]]
        f.write(f"{r['scale']} & {lab} & {r['estimate']:.4f} & {r['q']:.3f} & {r['d_study3']:.3f} & {r['d_study10']:.3f} & {r['d_study36']:.3f} & {r['W_study12']:.3f} \\\\\n")
    f.write("\\bottomrule\n" + "\\end{tabular}\n")

# ---------- 2. Shortest-path (Papakonstantinou 2018) study contributions and reconstruction check, risk scale ----
def comparison_level(net):
    """Aggregate direct estimates per comparison (two-arm studies only pooled by study weights g_k; multi-arm
    studies contribute each pair with its mesh conductance), then the comparison-level hat matrix."""
    edges = {}
    for k in net.studies:
        idx = net.rows[k]; arms = list(net.arm_trt[idx]); w = net.w[idx]; p = net.phat[idx]
        for i, j in combinations(range(len(idx)), 2):
            u, v = sorted([arms[i], arms[j]]); g = w[i] * w[j] / w.sum()
            y = p[arms.index(v)] - p[arms.index(u)]
            edges.setdefault((u, v), []).append((k, g, y))
    comp = {e: (sum(g for _, g, _ in L), sum(g * y for _, g, y in L) / sum(g for _, g, _ in L)) for e, L in edges.items()}
    return edges, comp
def shortest_path_contrib(net, a, b):
    edges, comp = comparison_level(net)
    # comparison-level GLS hat row for target: flows on comparison edges = g_e * (phi_u - phi_v)
    phi = net.Lp @ net.e(a, b)
    flow = {}
    for (u, v), (g, y) in comp.items():
        f = g * (phi[net.tix[u]] - phi[net.tix[v]])
        if abs(f) > 1e-12:
            flow[(u, v) if f > 0 else (v, u)] = abs(f)
    # shortest-path decomposition: repeatedly take the shortest a->b path (fewest edges), assign its bottleneck flow
    paths = []; flow = dict(flow)
    while True:
        # BFS shortest path on positive-flow edges
        prev = {a: None}; Q = [a]; found = False
        while Q and not found:
            u = Q.pop(0)
            for (x, y), f in flow.items():
                if x == u and f > 1e-12 and y not in prev:
                    prev[y] = x; Q.append(y)
                    if y == b: found = True; break
        if not found: break
        path = []; node = b
        while prev[node] is not None: path.append((prev[node], node)); node = prev[node]
        path = path[::-1]; wmin = min(flow[e] for e in path)
        for e in path: flow[e] -= wmin
        paths.append((wmin, path))
    # comparison contribution = sum over paths of flow/length (Papakonstantinou et al.)
    ccontrib = {}
    for wmin, path in paths:
        for e in path: ccontrib[e] = ccontrib.get(e, 0) + wmin / len(path)
    total = sum(ccontrib.values())
    ccontrib = {e: c / total for e, c in ccontrib.items()}
    # study contributions: split each comparison's percentage by the study's weight share within the comparison
    study = {}
    for (u, v), c in ccontrib.items():
        key = tuple(sorted([u, v])); L = edges[key]; gs = sum(g for _, g, _ in L)
        for k, g, y in L: study[(k, key)] = study.get((k, key), 0) + c * g / gs
    # reconstruction attempt: sum of study percentage times the study's direct estimate of the target
    recon = 0.0; n_undefined = 0
    for (k, key), c in study.items():
        if key == tuple(sorted([a, b])):
            y = [y for kk, g, y in edges[key] if kk == k][0]; recon += c * y
        else:
            n_undefined += 1
    return paths, ccontrib, study, recon, n_undefined
paths, ccont, scont, recon, nund = shortest_path_contrib(dong, a, b)
R["shortest_path"] = dict(n_paths=len(paths), recon_with_direct_only=recon, n_study_shares_without_target_estimate=nund,
                          estimate=rd0.attrs["estimate"], top_comparisons=sorted(ccont.items(), key=lambda x: -x[1])[:5],
                          study3_share=scont.get(("3", ("Placebo", "TIO-HH")), 0), csp_d3=float(rd0.loc[rd0.study == "3", "d"].iloc[0]))
# ---------- 3. Leave-one-out importance (Rucker et al. 2020: relative increase in variance on removal) ----
# Evaluated on the design scale, Var proportional to sum_i c_i^2 / n_i = r_ab, so that the importance is a
# property of the design (as in the original definition with fixed variances) and lies in [0, 1].
# (An earlier version plugged in outcome-based arm variances, which can give negative importances.)
def _rab(net_):
    e_ = net_.e(a, b); return float(e_ @ net_.Lp @ e_)
full_var = _rab(dong)
loo = []
for k in dong.studies:
    sub = Network(dong.df[dong.df.study != k])
    if sub.connected and a in sub.trts and b in sub.trts:
        loo.append((k, 1 - full_var / _rab(sub)))
    else:
        loo.append((k, np.nan))
loo = pd.DataFrame(loo, columns=["study", "loo_importance"]).merge(rd0[["study", "d", "W", "C"]], on="study")
loo.to_csv(f"{OUT}/loo_importance.csv", index=False)
R["loo"] = dict(top=loo.sort_values("loo_importance", ascending=False).head(5).round(4).to_dict("records"),
                corr_with_d=float(loo[["loo_importance", "d"]].corr().iloc[0, 1]), corr_with_W=float(loo[["loo_importance", "W"]].corr().iloc[0, 1]))
# ---------- 4. MH-NMA exclusions (netmetabin defaults) on COPD and NMOSD ----
def mh_exclusions(df):
    zero_studies = [k for k, g in df.groupby("study") if g.events.sum() == 0]
    d2 = df[~df.study.isin(zero_studies)]
    design = d2.groupby("study").treatment.apply(lambda s: "|".join(sorted(s)))
    excl_arms = []
    for des, ks in design.groupby(design).groups.items():
        sub = d2[d2.study.isin(ks)]
        for t in des.split("|"):
            if sub[sub.treatment == t].events.sum() == 0: excl_arms.append((des, t, len(ks)))
    return zero_studies, excl_arms
R["mh"] = dict(copd=mh_exclusions(dong.df), nmosd=mh_exclusions(pd.read_csv("../data/nmosd2025_relapse.csv")))
# ---------- 5. Evidence concentration ----
conc = []
for x, y in combinations(dong.trts, 2):
    r_ = dong.contributions(x, y); d = r_["d"].to_numpy(); W = r_["W"].to_numpy()
    conc.append(dict(target=f"{x}:{y}", eff_direct=(d.sum() ** 2 / np.sum(d ** 2)) if d.sum() > 0 else 0.0, direct_mass=d.sum(), n_zero_throughput=int((W < 1e-9).sum()), n_studies=len(d)))
conc = pd.DataFrame(conc); conc.to_csv(f"{OUT}/evidence_concentration.csv", index=False); R["conc"] = conc.round(3).to_dict("records")
# ---------- 6. Optimal allocation: ranking candidate two-arm trials for a target; convexity check; balanced arms ----
def delta(net, u, v, g, target):
    d = net.e(u, v); phi = net.Lp @ net.e(*target); r_uv = float(d @ net.Lp @ d)
    return g * float(d @ phi) ** 2 / (1 + g * r_uv)
N = 1000
rank = []
for u, v in combinations(dong.trts, 2):
    rank.append(dict(candidate=f"{u}:{v}", potential_gap=float((dong.Lp @ dong.e(a, b))[dong.tix[u]] - (dong.Lp @ dong.e(a, b))[dong.tix[v]]),
                     reduction_r_ab=delta(dong, u, v, N / 4, (a, b))))
rank = pd.DataFrame(rank).sort_values("reduction_r_ab", ascending=False); rank.to_csv(f"{OUT}/allocation_ranking.csv", index=False)
R["allocation"] = dict(r_ab=float(dong.e(a, b) @ dong.Lp @ dong.e(a, b)), ranking=rank.head(6).round(6).to_dict("records"),
                       balanced_vs_2to1=dict(balanced=delta(dong, a, b, N / 4, (a, b)), two_to_one=delta(dong, a, b, (N / 3) * (2 * N / 3) / N, (a, b))))
# convexity of r_ab in conductances: random chords
e_ab = dong.e(a, b); rng = np.random.default_rng(1); viol = 0
cands = list(combinations(dong.trts, 2))
def r_with(gvec):
    L = dong.L.copy()
    for (u, v), g in zip(cands, gvec):
        d = dong.e(u, v); L += g * np.outer(d, d)
    return float(e_ab @ np.linalg.pinv(L) @ e_ab)
for _ in range(300):
    g1 = rng.uniform(0, 2000, len(cands)); g2 = rng.uniform(0, 2000, len(cands)); t = rng.uniform()
    if r_with(t * g1 + (1 - t) * g2) > t * r_with(g1) + (1 - t) * r_with(g2) + 1e-12: viol += 1
R["allocation"]["convexity_violations_of_300"] = viol
# budget allocation across candidate comparisons (N=2000 patients, balanced arms, g_j = N_j/4): minimise r_ab
def obj(x):
    return r_with(np.maximum(x, 0) / 4.0)
x0 = np.full(len(cands), 2000 / len(cands))
cons = [{"type": "eq", "fun": lambda x: x.sum() - 2000}]
sol = minimize(obj, x0, bounds=[(0, 2000)] * len(cands), constraints=cons, method="SLSQP")
alloc = sorted([(c, float(n_)) for c, n_ in zip([f"{u}:{v}" for u, v in cands], sol.x) if n_ > 1], key=lambda z: -z[1])
R["allocation"]["budget2000"] = dict(allocation=alloc, r_ab_after=float(sol.fun), r_ab_before=R["allocation"]["r_ab"],
                                     all_direct=r_with(np.array([2000 / 4 if (u, v) == (a, b) else 0 for u, v in cands])))
# NMOSD: which trial next for eculizumab vs inebilizumab
nm = Network(pd.read_csv("../data/nmosd2025_relapse.csv")[lambda d: d.study != "CHAMPION"])
tgt = ("Eculizumab", "Inebilizumab")
rk = sorted([(f"{u}:{v}", delta(nm, u, v, 100 / 4, tgt)) for u, v in combinations(nm.trts, 2)], key=lambda z: -z[1])
R["allocation"]["nmosd_next_trial_100"] = dict(r_before=float(nm.e(*tgt) @ nm.Lp @ nm.e(*tgt)), ranking=rk[:5])
json.dump(R, open(f"{OUT}/extras.json", "w"), indent=1, default=str)
print(json.dumps(R, indent=1, default=str)[:6000])
