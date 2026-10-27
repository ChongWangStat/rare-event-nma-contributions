"""Significance vs weight vs size across the networks: direct weight d_k of each trial for each
target it compares directly, against (A) the Fisher p-value of its own comparison and (B) its size."""
import numpy as np, pandas as pd, json
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from itertools import combinations
from rdcsp import Network, study_significance
from scipy.stats import spearmanr
OUT="../results"
nets={"COPD":("dong2013_copd_mortality.csv",None),"Cystic fibrosis":("cf_sae_registry.csv",None),"NMOSD":("nmosd2025_relapse.csv",["CHAMPION"]),"SMA":("sma_efficacy_registry.csv",["STR1VE","FIREFISH","ENDEAR"])}
rows=[]
for name,(f,drop) in nets.items():
    df=pd.read_csv(f"../data/{f}")[["study","treatment","events","n"]]
    if drop: df=df[~df.study.isin(drop)]
    net=Network(df); sig=study_significance(net)
    for a,b in combinations(net.trts,2):
        res=net.contributions(a,b)
        for _,r in res[res.d>0].iterrows():
            k=r["study"]; s=sig[(sig.study==k)&(((sig.a==a)&(sig.b==b))|((sig.a==b)&(sig.b==a)))]
            if len(s)==0: continue
            n_pair=int(s.n_a.iloc[0]+s.n_b.iloc[0]); n_tot=int(net.n[net.rows[k]].sum())
            rows.append(dict(network=name,target=f"{a}:{b}",study=k,d=float(r["d"]),W=float(r["W"]),p=float(s.fisher_p.iloc[0]),n_pair=n_pair,n_total=n_tot,
                             n_direct_trials=int((res.d>0).sum())))
D=pd.DataFrame(rows); D.to_csv(f"{OUT}/significance_size_weight.csv",index=False)
summ={}
for name,g in D.groupby("network"):
    summ[name]=dict(pairs=len(g),spearman_p_d=float(spearmanr(g.p,g.d).correlation) if len(g)>2 else None,spearman_n_d=float(spearmanr(g.n_pair,g.d).correlation) if len(g)>2 else None,
                    share_weight_from_nonsig=float(g[g.p>=0.05].d.sum()/g.d.sum()),frac_pairs_nonsig=float((g.p>=0.05).mean()),
                    smallest_with_weight_1=g[g.d>0.99].sort_values("n_pair").head(1)[["study","n_pair","target"]].to_dict("records"),
                    largest_trial=g.sort_values("n_pair",ascending=False).head(1)[["study","n_pair","d","target"]].to_dict("records"))
all_=D; summ["all"]=dict(pairs=len(D),spearman_p_d=float(spearmanr(D.p,D.d).correlation),spearman_n_d=float(spearmanr(D.n_pair,D.d).correlation),share_weight_from_nonsig=float(D[D.p>=0.05].d.sum()/D.d.sum()))
# partial: within-target rank of size vs weight (size explains weight only relative to the other trials of the same comparison)
within=[]
for (name,t),g in D.groupby(["network","target"]):
    if len(g)>=3: within.append(spearmanr(g.n_pair,g.d).correlation)
summ["within_target_spearman_n_d_mean"]=float(np.nanmean(within)); summ["within_target_count"]=len(within)
json.dump(summ,open(f"{OUT}/significance_size_weight.json","w"),indent=1,default=str); print(json.dumps(summ,indent=1,default=str))
import figstyle as fs
from matplotlib.ticker import FixedLocator, NullFormatter, LogLocator
with plt.rc_context(fs.RC):
    fig,axes=plt.subplots(1,2,figsize=(fs.width(1.0),2.85),sharey=True)
    fs.margins(fig,left=0.5,right=0.1,bottom=0.45,top=0.52,wspace=0.12)
    for name in [n for n in fs.NET_COLOR if n in set(D.network)]:
        g=D[D.network==name]; c=fs.NET_COLOR[name]; fc=[c if p<0.05 else "white" for p in g.p]
        kw=dict(marker=fs.NET_MARKER[name],s=28,facecolor=fc,edgecolor=c,linewidth=0.9,zorder=3)
        axes[0].scatter(g.p,g.d,**kw); axes[1].scatter(g.n_pair,g.d,**kw)
    ax=axes[0]
    ax.axvline(0.05,ls=(0,(4,3)),color=fs.GREY,lw=0.7,zorder=1)
    ax.text(0.05,0.015,"p = 0.05 ",transform=ax.get_xaxis_transform(),ha="right",va="bottom",fontsize=8,color=fs.GREY)
    ax.set_xscale("log"); ax.set_xlim(2e-9,2); ax.xaxis.set_major_locator(FixedLocator([1e-8,1e-6,1e-4,1e-2,1]))
    ax.xaxis.set_minor_locator(LogLocator(numticks=20)); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(-0.04,1.07); ax.set_yticks([0,0.2,0.4,0.6,0.8,1])
    ax.set_xlabel("Fisher exact p-value of the trial's own comparison"); ax.set_ylabel("direct weight $d_k$ for that comparison")
    ax.set_title("(A) Significance and weight",pad=4)
    ax=axes[1]
    ax.set_xscale("log"); ax.set_xlim(25,12000); ax.xaxis.set_major_formatter(fs.thousands); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("patients in the trial's two compared arms"); ax.set_title("(B) Size and weight",pad=4)
    ax.tick_params(axis="y",labelleft=False)
    lead=dict(arrowstyle="-",lw=0.6,color=fs.GREY,shrinkA=0,shrinkB=3)
    for _,r in D[(D.d>0.99)&(D.n_pair<60)].iterrows():
        ax.annotate(r["study"],(r.n_pair,r.d),xytext=(4,-22),textcoords="offset points",fontsize=8,ha="left",va="top",arrowprops=lead)
    big=D.sort_values("n_pair",ascending=False).head(1).iloc[0]
    ax.annotate(f"study {big['study']} ({big['network']})",(big.n_pair,big.d),xytext=(-14,-10),textcoords="offset points",fontsize=8,ha="right",va="top",arrowprops=lead)
    nets=[n for n in fs.NET_COLOR if n in set(D.network)]
    fs.legend_rows(axes[0],[([fs.mark(marker=fs.NET_MARKER[n],mfc=fs.NET_COLOR[n],mec=fs.NET_COLOR[n],ms=6) for n in nets]
                             +[fs.mark(marker="o",mfc="white",mec=fs.GREY,mew=0.9,ms=6),fs.mark(marker="o",mfc=fs.GREY,mec=fs.GREY,ms=6)],
                             nets+["not significant","significant at 5%"],14)],gap=18)
    fig.savefig(f"{OUT}/figures/fig_significance_size_weight.pdf"); plt.close(fig)
