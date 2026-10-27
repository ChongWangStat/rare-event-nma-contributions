"""Route table, identification summary, flow figure, p-value vs weight figure, and
study table with SE(C_k) for the COPD placebo vs TIO-HH comparison."""
ARROW = r'$\to$'  # LaTeX arrow for route labels (kept outside f-strings for portability)
import numpy as np, pandas as pd, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import combinations
from rdcsp import Network, route_table, study_pairings

OUT = "../results"
dong = Network(pd.read_csv("../data/dong2013_copd_mortality.csv"))
a, b = "Placebo", "TIO-HH"
direct, rows, resid, ident = route_table(dong, a, b)
rows = sorted(rows, key=lambda r: -r["weight"])
# ---- route table (top 10 + rest) ----
with open(f"{OUT}/dong_routes.tex", "w") as f:
    f.write("\\begin{tabular}{llrrr}\n\\toprule\nStudies & Route & Weight & Sharp lower & Sharp upper \\\\\n\\midrule\n")
    for k, d in sorted(direct.items(), key=lambda x: -x[1])[:5]:
        if d > 0:
            f.write(f"{k} & direct & {d:.4f} & {d:.4f} & {d:.4f} \\\\\n")
    f.write(f"other direct studies & direct & {sum(d for k, d in sorted(direct.items(), key=lambda x: -x[1])[5:]):.4f} & -- & -- \\\\\n")
    for r in rows[:8]:
        route = r['path'].replace('\u2192', ARROW)
        f.write(f"{r['studies']} & {route} & {r['weight']:.4f} & {r['lower']:.4f} & {r['upper']:.4f} \\\\\n")
    f.write(f"other {len(rows)-8} routes & & {sum(r['weight'] for r in rows[8:]):.4f} & & \\\\\n")
    f.write(f"\\midrule\nSum & & {sum(direct.values()) + sum(r['weight'] for r in rows):.4f} & & \\\\\n\\bottomrule\n" + "\\end{tabular}\n")
# ---- identification summary across targets ----
res = []
for x, y in combinations(dong.trts, 2):
    d_, rs, _, idn = route_table(dong, x, y)
    res.append(dict(target=f"{x}:{y}", q=1 - sum(d_.values()), n_routes=len(rs), unique=idn["unique"],
                    max_width=max([r["upper"] - r["lower"] for r in rs], default=0.0)))
idf = pd.DataFrame(res); idf.to_csv(f"{OUT}/dong_identification.csv", index=False)
with open(f"{OUT}/dong_identification.tex", "w") as f:
    f.write("\\begin{tabular}{lrrlr}\n\\toprule\nTarget & Indirect mass & Canonical routes & Unique? & Widest sharp interval \\\\\n\\midrule\n")
    for _, r in idf.iterrows():
        f.write(f"{r['target']} & {r['q']:.3f} & {r['n_routes']} & {'yes' if r['unique'] else 'no'} & {r['max_width']:.3f} \\\\\n")
    f.write("\\bottomrule\n" + "\\end{tabular}\n")
summary = dict(n_unique=int(idf.unique.sum()), n_targets=len(idf), widest=float(idf.max_width.max()),
               widest_target=idf.loc[idf.max_width.idxmax(), "target"], pt_routes=len(rows), pt_width=float(idf.loc[idf.target == "Placebo:TIO-HH", "max_width"].iloc[0]),
               top_route=rows[0], top3_share=float(sum(r["weight"] for r in rows[:3]) / sum(r["weight"] for r in rows)))
# ---- flow figure on the potential axis ----
import figstyle as fs
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.legend_handler import HandlerBase
phi = dong.Lp @ dong.e(a, b)
cur = dong.currents(a, b)
tau = {k: 0.5 * np.abs(cur[k]).sum() for k in dong.studies}
keep = [k for k in dong.studies if tau[k] >= 0.01]
tphi = {t: float(phi[dong.tix[t]]) for t in dong.trts}
pmax, pmin = max(tphi.values()), min(tphi.values())
def X(p): return (pmax - p) / (pmax - pmin)          # rescaled potential: 0 at placebo (source), 1 at TIO-HH (sink)
# Vertical placement carries no information: it is chosen by hand only to keep nodes, labels and
# arrows apart (indirect routes above the placebo--TIO-HH line, direct placebo--TIO-HH trials below).
TY = {"Placebo": 0.0, "TIO-HH": 0.0, "TIO-SMI": -0.62, "ICS": 1.25, "LABA-ICS": 1.25, "LABA": 0.30}
TLAB = {"Placebo": "l", "TIO-SMI": "l", "TIO-HH": "r", "ICS": "t", "LABA-ICS": "t", "LABA": "tr"}
SY = {"17": (0.08, "l"), "15": (0.20, "l"), "38": (0.36, "l"), "36": (0.53, "l"), "39": (0.69, "l"), "40": (0.81, "l"),
      "37": (0.93, "l"), "41": (1.05, "l"), "16": (0.14, "r"), "32": (0.45, "r"), "35": (0.67, "r"), "22": (0.62, "l"),
      "12": (0.30, "b"), "14": (0.88, "t"), "11": (-0.10, "b"), "6": (-0.24, "r"),
      "13": (-0.34, "l"), "5": (-0.465, "l"), "4": (-0.58, "l"), "8": (-0.73, "l"), "10": (-0.92, "r"), "3": (-1.21, "l")}
S_TRT, S_TAU = 80.0, 1200.0                            # marker areas (pt^2); study area = S_TAU * throughput
def LW(c): return 0.35 + 9.0 * abs(c)                  # arrow width (pt), increasing linearly in |current|
def ASTYLE(w): return f"-|>,head_length={3.0 + 1.4 * w:.2f},head_width={1.4 + 0.75 * w:.2f}"
class _ArrowHandler(HandlerBase):                      # draws FancyArrowPatch handles in the legend
    def create_artists(self, legend, h, xd, yd, wd, ht, fontsize, trans):
        y = 0.5 * ht - yd
        return [FancyArrowPatch((-xd, y), (wd - xd, y), arrowstyle=h.get_arrowstyle(), mutation_scale=1, lw=h.get_linewidth(),
                                color=h.get_edgecolor(), shrinkA=0, shrinkB=0, transform=trans)]
with plt.rc_context(fs.RC):
    FIG_H = 4.6                                       # inches; printed at 0.95 x textwidth, so this is the printed size
    fig = plt.figure(figsize=(fs.width(0.95), FIG_H))
    ax = fig.add_axes([0.01, 1.1 / FIG_H, 0.98, 3.47 / FIG_H])
    ax.set_xlim(-0.115, 1.115); ax.set_ylim(-1.4, 1.46)
    for s in ("left", "right", "top"): ax.spines[s].set_visible(False)
    ax.set_yticks([]); ax.spines["bottom"].set_bounds(0, 1); ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.xaxis.set_major_formatter(fs.plain_log)
    ax.set_xlabel("potential, rescaled from the source (placebo = 0) to the sink (TIO-HH = 1)", labelpad=3)
    ax.text(1.115, 1.46, "Vertical placement is arbitrary\n(chosen only for legibility).", ha="right", va="top", fontsize=8,
            style="italic", color=fs.GREY)
    pos = {t: (X(tphi[t]), TY[t]) for t in dong.trts}
    size = {t: S_TRT for t in dong.trts}
    for k in keep:
        idx = dong.rows[k]; arms = list(dong.arm_trt[idx])
        pos[k] = (X(np.average([phi[dong.tix[u]] for u in arms], weights=dong.w[idx])), SY[str(k)][0])
        size[k] = S_TAU * tau[k]
    pts = 72.0 / fig.dpi
    def reach(n, d):                                   # centre-to-outline distance (pt) of node n along unit direction d
        if n in dong.trts: return 0.5 * np.sqrt(size[n]) + 1.2
        return 0.5 * np.sqrt(2 * size[n]) / (abs(d[0]) + abs(d[1])) + 1.0
    arrows = []
    for k in keep:
        idx = dong.rows[k]; arms = list(dong.arm_trt[idx])
        for j, u in enumerate(arms):
            c = cur[k][j]
            if abs(c) < 1e-6: continue
            arrows.append((abs(c), u, k, fs.VERMILION) if c > 0 else (abs(c), k, u, fs.BLUE))
    for c, s, e, col in sorted(arrows, key=lambda r: -r[0]):   # thick arrows first so thin ones stay visible
        A, B = (np.asarray(ax.transData.transform(pos[n])) * pts for n in (s, e))
        d = (B - A) / np.linalg.norm(B - A)
        ax.add_patch(FancyArrowPatch(pos[s], pos[e], arrowstyle=ASTYLE(LW(c)), mutation_scale=1, lw=LW(c), color=col,
                                     alpha=0.9, shrinkA=reach(s, d), shrinkB=reach(e, d), joinstyle="miter", zorder=2))
    halo = [pe.withStroke(linewidth=2.4, foreground="white")]
    off = {"l": ((-1, 0), "right", "center"), "r": ((1, 0), "left", "center"), "t": ((0, 1), "center", "bottom"), "b": ((0, -1), "center", "top"),
           "tr": ((0.55, 0.75), "left", "bottom")}
    for t in dong.trts:
        ax.scatter(*pos[t], s=S_TRT, marker="o", facecolor="white", edgecolor=fs.INK, linewidth=1.3, zorder=4)
        (ux, uy), ha, va = off[TLAB[t]]; r = 0.5 * np.sqrt(S_TRT) + 2.5
        ax.annotate(t, pos[t], xytext=(ux * r, uy * r), textcoords="offset points", ha=ha, va=va, fontsize=9,
                    fontweight="bold", path_effects=halo, zorder=6)
    for k in keep:
        ax.scatter(*pos[k], s=size[k], marker="D", facecolor="#D9D9D9", edgecolor=fs.INK, linewidth=0.7, zorder=3)
        (ux, uy), ha, va = off[SY[str(k)][1]]; r = 0.5 * np.sqrt(2 * size[k]) + 2.0
        ax.annotate(str(k), pos[k], xytext=(ux * r, uy * r), textcoords="offset points", ha=ha, va=va, fontsize=8,
                    path_effects=halo, zorder=6)
    # legends below the axis: node and arrow types (left); arrow-width and diamond-area keys (right, one row each)
    ah = lambda c, col: FancyArrowPatch((0, 0), (1, 0), arrowstyle=ASTYLE(LW(c)), lw=LW(c), color=col)
    dh = lambda v: Line2D([], [], ls="", marker="D", ms=np.sqrt(S_TAU * v), mfc="#D9D9D9", mec=fs.INK, mew=0.7)
    kw = dict(frameon=False, handler_map={FancyArrowPatch: _ArrowHandler()}, columnspacing=1.0, borderpad=0, borderaxespad=0)
    H = FIG_H * 72                                    # figure height in points
    fig.legend([Line2D([], [], ls="", marker="o", ms=np.sqrt(S_TRT), mfc="white", mec=fs.INK, mew=1.3), dh(0.03),
                ah(0.1, fs.VERMILION), ah(0.1, fs.BLUE)],
               ["treatment", "study (number shown)", "inflow: treatment to study", "outflow: study to treatment"],
               loc="lower left", bbox_to_anchor=(0.012, 2 / H), ncol=1, handlelength=2.4, labelspacing=0.3, **kw)
    fig.text(0.375, 40 / H, "arrow width = current:", ha="left", va="center", fontsize=8)
    fig.legend([ah(c, fs.GREY) for c in (0.01, 0.1, 0.3)], ["0.01", "0.1", "0.3"], loc="center left",
               bbox_to_anchor=(0.6, 40 / H), ncol=3, handlelength=2.6, **kw)
    fig.text(0.375, 17 / H, "diamond area = throughput:", ha="left", va="center", fontsize=8)
    fig.legend([dh(v) for v in (0.01, 0.1, 0.3)], ["0.01", "0.1", "0.3"], loc="center left", bbox_to_anchor=(0.6, 17 / H),
               ncol=3, handlelength=4.6, handleheight=1.0, **{**kw, "columnspacing": 0.5})
    fig.savefig(f"{OUT}/figures/fig_dong_flow.pdf"); plt.close(fig)
# ---- p-value vs direct weight ----
sig = pd.read_csv(f"{OUT}/dong_significance_vs_contribution.csv")
from matplotlib.ticker import FixedLocator, NullFormatter
S_N = 1 / 25.0                                         # marker area (pt^2) per patient: area proportional to trial size
with plt.rc_context(fs.RC):
    fig, ax = plt.subplots(figsize=(fs.width(0.7), 2.95))
    fs.margins(fig, left=0.52, right=0.12, bottom=0.45, top=0.5)
    s0, s1 = sig[sig.fisher_p >= 0.05], sig[sig.fisher_p < 0.05]
    ax.scatter(s0.fisher_p, s0.direct_weight, s=S_N * s0.n_total, facecolor="white", edgecolor=fs.BLUE, linewidth=0.9, zorder=2)
    ax.scatter(s1.fisher_p, s1.direct_weight, s=S_N * s1.n_total, facecolor=fs.BLUE, edgecolor=fs.BLUE, linewidth=0.9, zorder=3)
    for _, r in pd.concat([sig.nlargest(3, "direct_weight"), s1]).drop_duplicates().iterrows():      # trials named in the text
        off_pt = 0.5 * np.sqrt(S_N * r.n_total) + 2
        ax.annotate(f"study {r['study']}", (r.fisher_p, r.direct_weight), xytext=(off_pt, 0.6 * off_pt), textcoords="offset points",
                    fontsize=8, va="bottom")
    ax.axvline(0.05, ls=(0, (4, 3)), color=fs.GREY, lw=0.7, zorder=1)
    ax.text(0.05, 1.0, " p = 0.05", transform=ax.get_xaxis_transform(), ha="left", va="top", fontsize=8, color=fs.GREY)
    ax.set_xscale("log"); ax.set_xlim(0.04, 1.3); ax.set_ylim(-0.03, 0.76)
    ax.xaxis.set_major_locator(FixedLocator([0.05, 0.1, 0.2, 0.5, 1])); ax.xaxis.set_major_formatter(fs.plain_log)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Fisher exact p-value of the trial's own placebo comparison"); ax.set_ylabel("direct weight $d_k$")
    grey = dict(mec=fs.GREY, mew=0.9, marker="o")
    fs.legend_rows(ax, [([fs.mark(mfc="white", mec=fs.BLUE, mew=0.9, marker="o", ms=7), fs.mark(mfc=fs.BLUE, mec=fs.BLUE, marker="o", ms=7)],
                         ["not significant", "significant at 5%"], 13),
                        ([fs.mark(mfc="white", ms=np.sqrt(S_N * n), **grey) for n in (200, 1000, 5000)],
                         ["200", "1,000", "5,000 patients"], 16)])
    fig.savefig(f"{OUT}/figures/fig_dong_pvalue_weight.pdf"); plt.close(fig)
# ---- study table with SE(C_k) and direct/indirect covariance ----
res = dong.contributions(a, b)
rows2 = []
for _, r in res.iterrows():
    k = r["study"]; idx = dong.rows[k]; c = -cur[k]; p = dong.phat[idx]; n = dong.n[idx]
    v = p * (1 - p) / np.maximum(n - 1, 1)
    arms = list(dong.arm_trt[idx])
    cD = np.zeros(len(idx))
    if a in arms and b in arms:
        cD[arms.index(b)] += r["d"]; cD[arms.index(a)] -= r["d"]
    cI = c - cD
    rows2.append(dict(study=k, arms=r["arms"], d=r["d"], C_D=r["C_D"], C_I=r["C_I"], C=r["C"], se_C=np.sqrt((c ** 2 * v).sum()), cov_DI=float((cD * cI * v).sum())))
t2 = pd.DataFrame(rows2).sort_values("se_C", ascending=False)
t2.to_csv(f"{OUT}/dong_placebo_tiohh_studies_se.csv", index=False)
with open(f"{OUT}/dong_placebo_tiohh_studies_se.tex", "w") as f:
    f.write("\\begin{tabular}{llrrrrrr}\n\\toprule\nStudy & Arms & $d_k$ & $C^D_k$ & $C^I_k$ & $C_k$ & SE($C_k$) & Cov($C^D_k,C^I_k$) \\\\\n\\midrule\n")
    for _, r in t2.head(12).iterrows():
        f.write(f"{r['study']} & {r['arms']} & {r['d']:.3f} & {r['C_D']:.5f} & {r['C_I']:.5f} & {r['C']:.5f} & {r['se_C']:.5f} & {r['cov_DI']:.1e} \\\\\n")
    rest = t2.iloc[12:]
    f.write(f"Other {len(rest)} studies & & {rest['d'].sum():.3f} & {rest['C_D'].sum():.5f} & {rest['C_I'].sum():.5f} & {rest['C'].sum():.5f} & -- & -- \\\\\n")
    f.write(f"\\midrule\nSum & & {t2['d'].sum():.3f} & {t2['C_D'].sum():.5f} & {t2['C_I'].sum():.5f} & {t2['C'].sum():.5f} & {np.sqrt((t2['se_C']**2).sum()):.5f} & \\\\\n\\bottomrule\n" + "\\end{tabular}\n")
summary["se_total"] = float(np.sqrt((t2["se_C"] ** 2).sum()))
summary["largest_cov_study"] = str(t2.loc[t2.cov_DI.abs().idxmax(), "study"]); summary["largest_cov"] = float(t2.cov_DI.abs().max())
json.dump(summary, open(f"{OUT}/dong_routes_summary.json", "w"), indent=1, default=str)
print(json.dumps(summary, indent=1, default=str))
print(open(f"{OUT}/dong_routes.tex").read())
