"""No-fit zero/positive power error decomposition on all common later cases."""
import json
from pathlib import Path
import time
import run_local
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from field_control.activity_diagnostics import error_allocation, summarize_allocation, COMPONENTS
from field_control.response import read_response_submission

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/operational_activity_v1"
MODELS=("Ridge","LightGBM","N-HiTS")
GROUP=["source","site","period","model","span_minutes"]


def prediction(path, ids, width):
    with np.load(path, allow_pickle=False) as archive:
        stored=pd.Index(archive["case_id"].astype(str))
        if not stored.is_unique: raise ValueError("Duplicate prediction IDs")
        order=stored.get_indexer(ids)
        if (order<0).any(): raise ValueError("Missing case prediction")
        value=archive["prediction_kw"][order]
    if value.shape != (len(ids),width): raise ValueError("Wrong future prediction width")
    return value


def contexts():
    bpa=ROOT/"results/benchmark_bpa_sequences_v1"
    for period in ["spring","summer"]:
        meta=pd.read_csv(bpa/period/"cases.csv.gz", dtype={"case_id":str,"alias":str})
        y=np.load(bpa/period/"target_kw.npy", mmap_mode="r")[:,2:]
        paths={"Ridge":[bpa/"telemetry_value_v1/ridge_reference_float64/full"/f"{period}.npz"],
               "LightGBM":[bpa/"standard_refresh_v1"/f"{period}.npz"],
               "N-HiTS":[bpa/"telemetry_value_v1"/f"nhits_seed{s}_full"/f"{period}.npz" for s in [1,2,3]]}
        yield "BPA","Field study",period,meta,y,15,paths
    acn=ROOT/"results/acn_fleet_months_v1"
    all_meta=pd.read_csv(acn/"cases.csv.gz")
    y_all=np.load(acn/"neural_packets_v1/target.npy", mmap_mode="r")
    for site,label in [("caltech","Caltech"),("office001","Office")]:
        paths={"Ridge":[acn/"ridge_references_float64_v1"/site/"full/predictions.npz"],
               "LightGBM":[acn/"references/full.npz"],
               "N-HiTS":[acn/"neural_references_v1"/site/f"nhits_seed{s}_full/predictions.npz" for s in [1,2,3]]}
        for period in ["autumn","winter"]:
            ix=np.flatnonzero((all_meta.site==site)&(all_meta.split==period))
            yield "ACN",label,period,all_meta.iloc[ix].copy(),y_all[ix],15,paths
    epfl=ROOT/"results/epfl_dataset_v1"
    all_meta=pd.read_csv(epfl/"cases.csv")
    ix=np.flatnonzero(all_meta.split=="later")
    with np.load(epfl/"packet.npz",allow_pickle=False) as packet:
        if not np.array_equal(packet["case_id"].astype(str), all_meta.case_id): raise ValueError("EPFL packet IDs differ")
        y=packet["y"][ix,:,0]
    paths={"Ridge":[ROOT/"results/epfl_ridge_references_v1/full/predictions.npz"],
           "LightGBM":[ROOT/"results/epfl_operational_pilot_v1/full.npz"],
           "N-HiTS":[ROOT/f"results/epfl_neural_references_v1/nhits_seed{s}_full/predictions.npz" for s in [1,2,3]]}
    meta=all_meta.iloc[ix].rename(columns={"date":"local_date"})
    yield "EPFL","Fast charging","later",meta,y,5,paths


def main():
    started=time.monotonic(); OUT.mkdir(exist_ok=True)
    rows=[]; source_support=[]; used=[]
    for source,site,period,meta,y,step,paths in contexts():
        if not meta.case_id.is_unique: raise ValueError("Duplicate dataset cases")
        audit=dict(source=source,site=site,period=period,cases=len(meta),dates=meta.local_date.nunique(),
                   negative_bins=int(np.sum(y<0)),nonfinite_bins=int(np.sum(~np.isfinite(y))),
                   minimum_target_kw=float(np.nanmin(y)))
        source_support.append(audit)
        (OUT/"source_audit.json").write_text(json.dumps(source_support,indent=2)+"\n",encoding="utf-8")
        if audit["negative_bins"] or audit["nonfinite_bins"]: raise ValueError("Source audit requires review")
        for model in MODELS:
            for seed,path in enumerate(paths[model],start=1):
                used.append(str(path.relative_to(ROOT)))
                if source=="BPA":
                    pred=read_response_submission(path,meta.case_id)
                elif source=="EPFL" and model=="LightGBM":
                    with np.load(path,allow_pickle=False) as archive:
                        ids=pd.Index(archive["case_id"].astype(str)); order=ids.get_indexer(meta.case_id)
                        if not ids.is_unique or (order<0).any(): raise ValueError("EPFL tree IDs differ")
                        pred=archive["prediction_kw"][order,:,0]
                else: pred=prediction(path,meta.case_id,y.shape[1])
                for bins in ([1,3,6] if source=="EPFL" else [2,6,10]):
                    daily=error_allocation(meta.local_date,y,pred,bins)
                    rows.append(daily.assign(source=source,site=site,period=period,model=model,
                                             span_minutes=bins*step,seed=seed))
        print(json.dumps(dict(completed_context=[source,site,period],cases=len(meta))),flush=True)
    individual=pd.concat(rows,ignore_index=True)
    individual.to_csv(OUT/"per_seed_daily_internal.csv.gz",index=False)
    # Average each model's errors, not predictions. Every date/seed is retained.
    daily=individual.groupby(GROUP+["date"])[list(COMPONENTS)+["cases"]].mean().reset_index()
    daily.to_csv(OUT/"daily_scores.csv.gz",index=False)
    summary=pd.DataFrame([dict(zip(GROUP,key),**summarize_allocation(group))
                          for key,group in daily.groupby(GROUP)])
    original=pd.read_csv(ROOT/"results/common_device_comparison_v1/common_device_scores.csv")
    original=original[original.condition=="full"]
    check=summary.merge(original,on=GROUP,validate="one_to_one",suffixes=("_new","_original"))
    if len(check)!=63 or len(summary)!=63: raise ValueError("Common device comparison is incomplete")
    for metric in ["mse","mae"]:
        if not np.allclose(check[metric+"_new"],check[metric+"_original"],rtol=1e-6,atol=1e-7):
            check.to_csv(OUT/"score_mismatch.csv",index=False)
            raise ValueError("Does not recover the existing date-balanced score")
    if not (check.units_new==check.units_original).all() or not (check.dates_new==check.dates_original).all():
        raise ValueError("Case/date support differs")
    summary.to_csv(OUT/"summary.csv",index=False)
    average=summary.groupby(GROUP[:-1])[list(COMPONENTS)].mean().reset_index()
    average["zero_mse_fraction"]=average.zero_mse/average.mse
    average.to_csv(OUT/"span_average.csv",index=False)
    lines=["# Recorded-zero and positive-power error allocation","",
        "Full-input models on all existing later periods. Contributions use the original equal-date weights and sum to total MSE. MSE is in kW squared. The zero bin has exactly zero recorded target; it is not an inferred physical disconnection. Entries average native spans (30/90/150 min for BPA/ACN; 5/15/30 min for EPFL). N-HiTS averages three seed errors.","",
        "| Source / site / period | Model | Zero-target bins (%) | MSE on zero bins (contribution) | MSE on positive bins (contribution) | Total MSE | Zero-bin share of MSE (%) |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for (source,site,period),part in average.groupby(["source","site","period"],sort=False):
        for model in MODELS:
            r=part[part.model==model].iloc[0]
            lines.append(f"| {source} / {site} / {period} | {model} | {100*r.zero_share:.1f} | {r.zero_mse:.3f} | {r.positive_mse:.3f} | {r.mse:.3f} | {100*r.zero_mse_fraction:.1f} |")
    (ROOT/"paper/generated/operational_activity_results.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    # Plot shares by horizon; table gives separate-period span averages.
    fig,axes=plt.subplots(1,4,figsize=(11.8,3.4),sharey=True)
    colors={"Ridge":"#7567a2","LightGBM":"#167c94","N-HiTS":"#c37c24"}
    for ax,(source,site,label) in zip(axes,[("BPA","Field study","BPA"),("ACN","Caltech","ACN Caltech"),
                                             ("ACN","Office","ACN Office"),("EPFL","Fast charging","EPFL")]):
        part=summary[(summary.source==source)&(summary.site==site)]
        for model in MODELS:
            p=part[part.model==model].groupby("span_minutes")[["zero_mse","mse"]].mean()
            ax.plot(p.index,100*p.zero_mse/p.mse,marker="o",ms=4,color=colors[model],label=model)
        support=part[part.model=="Ridge"].groupby("span_minutes").zero_share.mean()
        ax.plot(support.index,100*support,color="#444444",ls="--",lw=1.2,label="Zero-target bin share")
        ax.set_title(label,fontsize=11);ax.set_xlabel("Forecast span (minutes)",fontsize=9)
        ax.set_xticks([5,15,30] if source=="EPFL" else [30,90,150])
        ax.set_ylim(0,100);ax.spines[["top","right"]].set_visible(False);ax.grid(axis="y",alpha=.2)
    axes[0].set_ylabel("Share (%)")
    fig.legend(*axes[0].get_legend_handles_labels(),loc="lower center",ncol=4,frameon=False,fontsize=9)
    fig.tight_layout(rect=[0,.13,1,1]);fig.savefig(OUT/"zero_power_error_shares.png",dpi=160);plt.close(fig)
    receipt=dict(completed=True,new_fits=0,seconds=time.monotonic()-started,contexts=7,
        score_cells=len(summary),model_forecasts=35,source_support=source_support,
        reused_prediction_files=sorted(set(used)),recovered_existing_mse_mae=True,
        negative_targets=0,weighting="All eligible dates, case-bin mean within each date; seed-error means",
        zero_definition="Recorded target equals zero, with no threshold fitting")
    (OUT/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8")
    print("\n".join(lines));print(json.dumps({k:v for k,v in receipt.items() if k not in ["reused_prediction_files","source_support"]}))


if __name__=="__main__":main()
