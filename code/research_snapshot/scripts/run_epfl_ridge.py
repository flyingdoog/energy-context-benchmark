"""Three default Ridge references on EPFL actual power; no source reread or HPO."""
import json
from pathlib import Path
import time
import run_local
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
import pilot_epfl_operational_forecasting as pilot

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"results/epfl_operational_pilot_v1"
OUT=ROOT/"results/epfl_ridge_references_v1"


def main():
    OUT.mkdir(exist_ok=True);start=time.monotonic()
    meta=pd.read_csv(BASE/"cases.csv")
    case_ids=meta.case_id.to_numpy(dtype=str)
    with np.load(BASE/"packet.npz",allow_pickle=False) as packet:
        if not np.array_equal(packet["case_id"].astype(str),case_ids):
            raise ValueError("Case ordering differs")
        x=packet["x"].astype(np.float64);y=packet["y"]
    train=meta.split.to_numpy()=="train"
    mean=x[train].mean(axis=0);scale=x[train].std(axis=0)
    scale[scale<1e-6]=1
    # Keep source binary control flag unscaled. Other numeric columns are
    # centered/scaled on the training cases only.
    mean[66:94:3]=0;scale[66:94:3]=1
    z=(x-mean)/scale
    rows=[];receipts=[];new_fits=0
    for condition,width in [("power",14),("power_state",64),("full",94)]:
        folder=OUT/condition;folder.mkdir(exist_ok=True)
        receipt=folder/"receipt.json";dest=folder/"predictions.npz"
        if receipt.exists() and json.loads(receipt.read_text()).get("completed"):
            saved=np.load(dest,allow_pickle=False)
            if not np.array_equal(saved["case_id"].astype(str),case_ids):
                raise ValueError("Cached prediction ordering differs")
            pred=saved["prediction_kw"];info=json.loads(receipt.read_text())
        else:
            tick=time.monotonic();columns=np.arange(width)
            keep=np.ptp(z[train,:width],axis=0)>0
            columns=columns[keep]
            model=Ridge()
            with threadpool_limits(limits=2):
                model.fit(z[train][:,columns],y[train,:,0])
                pred=np.maximum(model.predict(z[:,columns]),0)
            if not np.isfinite(pred).all():raise ValueError("Nonfinite Ridge forecast")
            np.savez_compressed(dest,case_id=case_ids,prediction_kw=pred,target="actual_power")
            np.savez_compressed(folder/"linear_model.npz",coef=model.coef_,intercept=model.intercept_,
                                mean=mean,scale=scale,columns=columns)
            info=dict(completed=True,condition=condition,seconds=time.monotonic()-tick,
                      training_cases=int(train.sum()),columns=columns.tolist(),
                      parameters=model.get_params(),dtype="float64",
                      recipe="Default Ridge alpha1/intercept; training-only numeric scaling; six actual-power outputs. No HPO.")
            receipt.write_text(json.dumps(info,indent=2)+"\n",encoding="utf-8");new_fits+=1
        full=np.zeros_like(y);full[:,:,0]=pred
        score=pilot.score(meta,y,full,condition)
        rows.append(score[score.target=="actual_power"]);receipts.append(info)
    daily=pd.concat(rows,ignore_index=True)
    daily.to_csv(OUT/"daily_scores.csv.gz",index=False)
    summary=daily.groupby(["split","target","span_minutes","method"])[["mse","mae","energy_ae_kwh"]].mean().reset_index()
    summary.to_csv(OUT/"summary.csv",index=False)
    final=dict(completed=True,new_fits_this_run=new_fits,total_multioutput_models=3,
               seconds=time.monotonic()-start,models=receipts)
    (OUT/"receipt.json").write_text(json.dumps(final,indent=2)+"\n",encoding="utf-8")
    print(summary[summary.split=="later"].groupby("method")[["mse","mae"]].mean().round(3).to_string(),flush=True)
    print(json.dumps({k:v for k,v in final.items() if k!="models"}),flush=True)


if __name__=="__main__":main()

