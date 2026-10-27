"""Do non-significant trials contribute meaningfully?  For every target with direct evidence: the share of
direct weight supplied by trials whose own comparison was not significant; the design-based precision gain
of the network over the best single trial, r_single / r_ab with r_single = 1/n_a + 1/n_b (the ratio of
minimax worst-case variances, which needs no outcomes); and, with the same Wald construction and exact
plug-in variance for both, whether the network and the best single trial have 95% half-widths below a
minimally important absolute difference (MID)."""
import numpy as np, pandas as pd, json
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from itertools import combinations
from rdcsp import Network, study_significance
OUT="../results"; Z=1.959963984540054
nets={"COPD":("dong2013_copd_mortality.csv",None,0.01),"Cystic fibrosis":("cf_sae_registry.csv",None,0.05),"NMOSD":("nmosd2025_relapse.csv",["CHAMPION"],0.10)}
rows=[]
for name,(f,drop,mid) in nets.items():
    df=pd.read_csv(f"../data/{f}")[["study","treatment","events","n"]]
    if drop: df=df[~df.study.isin(drop)]
    net=Network(df); sig=study_significance(net)
    for a,b in combinations(net.trts,2):
        res=net.contributions(a,b); direct=res[res.d>0]
        if len(direct)==0: continue
        c=net.coefficient_vector(a,b); se=np.sqrt((c**2*net.phat*(1-net.phat)/np.maximum(net.n-1,1)).sum())
        est=res.attrs["estimate"]; hw=Z*se
        w_nonsig=0.0; n_sig=0; best=None
        e=net.e(a,b); r_ab=float(e@net.Lp@e)
        for _,r in direct.iterrows():
            k=r["study"]; s=sig[(sig.study==k)&(((sig.a==a)&(sig.b==b))|((sig.a==b)&(sig.b==a)))].iloc[0]
            p=s.fisher_p; n_sig+= p<0.05
            if p>=0.05: w_nonsig+=r["d"]
            r1=1/s.n_a+1/s.n_b                     # the trial's own design resistance for a:b
            if best is None or r1<best[0]: best=(r1,s)
        # design-based precision gain: ratio of minimax worst-case variances (Theorem 2), outcome-free
        gain=best[0]/r_ab
        # same interval construction for the network and the design-best trial: Wald with the exact
        # (unbiased plug-in) variance; reported only where the best trial has events in both arms
        s=best[1]
        v1=(s.r_a/s.n_a)*(1-s.r_a/s.n_a)/max(s.n_a-1,1)+(s.r_b/s.n_b)*(1-s.r_b/s.n_b)/max(s.n_b-1,1)
        hw1=Z*np.sqrt(v1) if (0<s.r_a<s.n_a and 0<s.r_b<s.n_b) else np.nan
        network_within_mid = hw < mid                       # resolving: half-width smaller than an important difference
        rows.append(dict(network=name,target=f"{a}:{b}",estimate=est,halfwidth=hw,n_direct=len(direct),n_direct_sig=int(n_sig),
                         weight_share_nonsig=w_nonsig/direct.d.sum(),direct_mass=direct.d.sum(),best_trial=s.study,
                         best_single_halfwidth=hw1,precision_gain=gain,mid=mid,network_excludes_mid=bool(network_within_mid),
                         best_trial_excludes_mid=bool(hw1<mid) if np.isfinite(hw1) else False))
D=pd.DataFrame(rows); D.to_csv(f"{OUT}/informativeness.csv",index=False)
pd.set_option("display.width",220); print(D.round(4).to_string())
summ={}
for name,g in D.groupby("network"):
    allnon=g[g.weight_share_nonsig>=0.999]
    summ[name]=dict(targets=len(g),targets_all_nonsig=len(allnon),
        median_precision_gain_all_nonsig=float(allnon.precision_gain.median()) if len(allnon) else None,
        n_network_excludes_mid=int(g.network_excludes_mid.sum()),n_best_trial_excludes_mid=int(g.best_trial_excludes_mid.sum()),
        n_network_only=int((g.network_excludes_mid & ~g.best_trial_excludes_mid).sum()),mid=float(g.mid.iloc[0]),
        median_precision_gain=float(g.precision_gain.median()))
json.dump(summ,open(f"{OUT}/informativeness.json","w"),indent=1); print(json.dumps(summ,indent=1))
# figure: share of weight from non-significant trials vs precision gain of the network over the best single trial
import figstyle as fs
from matplotlib.ticker import FixedLocator, NullFormatter
with plt.rc_context(fs.RC):
    fig,ax=plt.subplots(figsize=(fs.width(0.7),3.1))
    fs.margins(fig,left=0.62,right=0.12,bottom=0.45,top=0.6)
    for name,g in D.groupby("network"):
        c=fs.NET_COLOR[name]
        ax.scatter(g.weight_share_nonsig,g.precision_gain,marker=fs.NET_MARKER[name],s=32,linewidth=0.9,zorder=3,
                   facecolor=[c if e else "white" for e in g.network_excludes_mid],edgecolor=c)
    ax.axhline(1,ls=(0,(4,3)),color=fs.GREY,lw=0.7,zorder=1)
    ax.text(0.5,1,"equal precision",ha="center",va="bottom",fontsize=8,color=fs.GREY)
    ax.set_yscale("log"); ax.set_ylim(0.8,10.5); ax.set_xlim(-0.05,1.05); ax.set_xticks([0,0.25,0.5,0.75,1])
    ax.yaxis.set_major_locator(FixedLocator([1,2,5,10])); ax.yaxis.set_major_formatter(fs.plain_log); ax.yaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_major_formatter(fs.plain_log)
    ax.set_xlabel("share of direct weight from non-significant trials")
    ax.set_ylabel("design precision gain over\nthe best single trial")
    nets=[n for n in fs.NET_COLOR if n in set(D.network)]
    fs.legend_rows(ax,[([fs.mark(marker=fs.NET_MARKER[n],mfc=fs.NET_COLOR[n],mec=fs.NET_COLOR[n],ms=6) for n in nets],nets,13),
                       ([fs.mark(marker="o",mfc=fs.GREY,mec=fs.GREY,ms=6)],["filled: network half-width below the minimally important difference"],12),
                       ([fs.mark(marker="o",mfc="white",mec=fs.GREY,mew=0.9,ms=6)],["open: not below it"],12)])
    fig.savefig(f"{OUT}/figures/fig_informativeness.pdf"); plt.close(fig)
