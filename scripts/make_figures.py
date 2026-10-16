"""Reproduce three original figures from sanitized SOFTWARE benchmark metadata.

No private scientific arrays or simulation is needed. No error bars: these are
reported timings or diagnostic maxima, not independent experimental replicates.
Figure contract: 183 mm single axes; original readable sans-serif; 9 pt glyphs;
zero-baseline bars only for comparable timing scopes, log points for ratios.
"""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"docs/figures"
DATA=json.loads((ROOT/"benchmarks/data/public_results.json").read_text())
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,
    "axes.titlesize":11,"axes.labelsize":9,"xtick.labelsize":8,"ytick.labelsize":9,
    "pdf.fonttype":42,"svg.fonttype":"none","savefig.dpi":300})


def save(fig,name):
    OUT.mkdir(parents=True,exist_ok=True)
    fig.savefig(OUT/f"{name}.png",dpi=300)
    fig.savefig(OUT/f"{name}.svg")
    fig.savefig(OUT/f"{name}.pdf",metadata={"Creator":"phono3py-cupy-accelerator"})
    plt.close(fig)


def timing_stages():
    fig,ax=plt.subplots(figsize=(7.2047244,4.1338583))
    fig.subplots_adjust(left=.25,right=.89,bottom=.24,top=.82)
    values=list(DATA["contraction"]["seconds"].values())
    labels=["Native C (8 threads)","NumPy (8 threads)","CuPy (block 32)"]
    # Points, not log-length bars: equal distances mean ratios on this log axis.
    for y,(value,label,color,marker) in enumerate(zip(values,labels,["#777777","#0072B2","#D55E00"],["s","o","D"])):
        ax.scatter(value,y,color=color,marker=marker,s=55,zorder=3)
        ax.annotate(f"{value:.3f} s",(value,y),xytext=(7,8),textcoords="offset points",fontsize=9)
    ax.set(yticks=range(3),yticklabels=labels,xscale="log",xlim=(.07,500),ylim=(2.6,-.65),
           xlabel="Warm single-triplet contraction time (seconds; log scale)")
    ax.set_title("219³ channels: algorithm + backend microbenchmark",pad=18)
    ax.spines[["right","top","left"]].set_visible(False)
    fig.text(.02,.09,"Historical private fixture; nonexclusive CPU host; no matched end-to-end speedup claim.",fontsize=8)
    f=DATA["fourier"]
    ratio=sum(f["native_sample_s"])/sum(f["warm_cupy_sample_s"])
    fig.text(.02,.035,f"Separate Fourier stage: {ratio:.1f}× steady-state; excludes preparation and first-call overhead.",fontsize=8)
    save(fig,"01_stage_timings")


def pipeline_scopes():
    fig,ax=plt.subplots(figsize=(7.2047244,4.2519685))
    fig.subplots_adjust(left=.36,right=.87,bottom=.26,top=.83)
    p=DATA["pipeline"]
    values=[p["old_gpu_worker_s"]/60,p["new_gpu_segmented_q_sum_s"]/60,p["new_gpu_fresh_process_s"]/60]
    labels=["Old GPU\nworker wall","New GPU\nsegmented q sum","New GPU\nfresh process wall"]
    bars=ax.barh(range(3),values,color=["#8A8A8A","#E69F00","#0072B2"],height=.52)
    ax.set(yticks=range(3),yticklabels=labels,xlim=(0,49),ylim=(2.65,-.65),xlabel="Recorded time (minutes)")
    for y,v in enumerate(values):ax.text(v+.6,y,f"{v:.2f}",va="center")
    ax.set_title("Complete M14 observations — unequal timing scopes",pad=16)
    ax.spines[["right","top","left"]].set_visible(False)
    fig.text(.02,.10,"Segmented = 16 reused + 8 resumed q points, not a fresh whole-process benchmark.",fontsize=8)
    fig.text(.02,.045,"Historical runs were not a matched isolated comparison. No end-to-end speedup ratio is asserted.",fontsize=8)
    save(fig,"02_pipeline_scopes")


def accuracy():
    fig,ax=plt.subplots(figsize=(7.2047244,4.4094488))
    fig.subplots_adjust(left=.33,right=.87,bottom=.24,top=.82)
    # Show accepted representation deviations in percent in a separate figure
    # section, with no invented numerical threshold for owner acceptance.
    ratios = list(DATA["representation72"]["relative_errors"].values())
    ratios = [v*100 for v in ratios]
    labels=["72 representation: κP", "72 representation: κC", "72 representation: total κ"]
    for y,value in enumerate(ratios):
        ax.scatter(value,y,s=55,marker="o",color="#0072B2",zorder=3)
        ax.annotate(f"{value:.6f}%",(value,y),xytext=(8,9),textcoords="offset points")
    ax.set(yticks=range(3),yticklabels=labels,xlim=(0,.45),ylim=(2.6,-.6),
           xlabel="Absolute relative difference (%)")
    ax.set_title("72-atom end-to-end representation comparison: accepted",pad=18)
    ax.spines[["right","top","left"]].set_visible(False)
    fig.text(.02,.10,"73-atom modal κP / κC: max tolerance ratios 0.006343 / 8.811e-6 (PASS).",fontsize=8)
    g=DATA["gamma24"]
    fig.text(.02,.045,f"All 24 q points × 219 branches: max |Δγ| = {g['max_abs_THz']:.3e} THz vs streamed NumPy.",fontsize=8)
    save(fig,"03_accuracy_boundaries")


if __name__=="__main__":
    timing_stages(); pipeline_scopes(); accuracy()
    print("Three figures exported as PNG, SVG and PDF")
