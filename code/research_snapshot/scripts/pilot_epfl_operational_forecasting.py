"""Fixed-recipe dual logged-target pilot; no baseline HPO or future inputs."""
import json
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor
import run_local
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from field_control.epfl_preparation import build_packet

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'results/benchmark_source_feasibility_v1/epfl_level3'
OUT=ROOT/'results/epfl_operational_pilot_v1'
STATE=['SOC (%)','TotalCapacity','RemainingCapacity','BulkCapacity','Energy (Wh)']
CONTROL=['Pset (W)','Preq (W)','Ctrl_t (0 = False, 1 = True)']


def packet():
    return build_packet(SOURCE)

def score(meta,y,prediction,method):
    rows=[]
    for target,name in enumerate(['actual_power','requested_power']):
        for bins in [1,3,6]:
            error=prediction[:,:bins,target]-y[:,:bins,target]
            frame=meta[['split','date']].copy()
            frame['mse']=np.mean(error**2,axis=1);frame['mae']=np.mean(abs(error),axis=1)
            frame['energy_ae_kwh']=abs(error.sum(axis=1)*5/60)
            days=frame.groupby(['split','date']).mean().reset_index()
            rows.append(days.assign(method=method,target=name,span_minutes=bins*5))
    return pd.concat(rows,ignore_index=True)


def main():
    start=time.monotonic();OUT.mkdir(exist_ok=True)
    meta,x,y=packet();meta.to_csv(OUT/'cases.csv',index=False)
    np.savez_compressed(OUT/'packet.npz',x=x,y=y,case_id=meta.case_id.to_numpy(dtype=str))
    train=meta.split.to_numpy()=='train'
    predictions={};receipts=[]
    def fit(condition,columns):
        prediction=np.zeros_like(y);folder=OUT/condition;folder.mkdir(exist_ok=True)
        for target in range(2):
            for lead in range(6):
                model=LGBMRegressor(n_jobs=2,random_state=2027,verbosity=-1)
                model.fit(x[train][:,columns],y[train,lead,target])
                prediction[:,lead,target]=np.maximum(model.predict(x[:,columns]),0)
                model.booster_.save_model(str(folder/f'target{target}_lead{lead}.txt'))
        np.savez_compressed(OUT/(condition+'.npz'),case_id=meta.case_id.to_numpy(dtype=str),prediction_kw=prediction)
        return condition,prediction,dict(condition=condition,lead_fits=12,features=len(columns),parameters=model.get_params())
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs=[pool.submit(fit,'power',np.arange(14)),pool.submit(fit,'power_state',np.arange(64)),
            pool.submit(fit,'full',np.arange(94))]
        for job in jobs:
            condition,prediction,info=job.result();predictions[condition]=prediction;receipts.append(info)
            print(json.dumps(dict(completed=condition,fits=12)),flush=True)
    # Strong transparent persistence for each target uses that target's own past.
    predictions['own_target_persistence']=np.repeat(meta[['last_actual_kw','last_requested_kw']].to_numpy()[:,None,:],6,axis=1)
    daily=pd.concat([score(meta,y,prediction,name) for name,prediction in predictions.items()],ignore_index=True)
    daily.to_csv(OUT/'daily_scores.csv.gz',index=False)
    summary=daily.groupby(['split','target','span_minutes','method'])[['mse','mae','energy_ae_kwh']].mean().reset_index()
    summary.to_csv(OUT/'summary.csv',index=False)
    table=summary[summary.split=='later'].groupby(['target','method'])[['mse','mae']].mean().reset_index()
    table.to_csv(OUT/'span_average.csv',index=False)
    receipt=dict(completed=True,seconds=time.monotonic()-start,new_lead_fits=36,models=receipts,
        support=meta.groupby('split').agg(sessions=('case_id','size'),dates=('date','nunique')).reset_index().to_dict(orient='records'),
        query_minutes=10,history_minutes=10,target_bins_minutes=5,spans_minutes=[5,15,30],
        targets='Reported actual AC power and reported EV-requested power under recorded operation.',
        end_rule='Zero only after the source-confirmed departure of this same session; no within-session missing fill.',
        validation='No HPO or model selection. Fixed official/default LightGBM recipe; results are a pilot, not evidence all neural models fail.',
        clock='Source-native naive timestamps. No external weather or timezone-aligned input.',
        excluded_inputs=['future setpoint','whole-session control flag','realized final energy','actual departure','session maxima','departure SoC','session-derived capacity'])
    (OUT/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(table.round(3).to_string(index=False),flush=True)
    print(json.dumps({k:v for k,v in receipt.items() if k!='models'}),flush=True)


if __name__=='__main__':main()
