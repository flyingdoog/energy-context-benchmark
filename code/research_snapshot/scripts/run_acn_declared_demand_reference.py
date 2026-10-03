"""Same default tree recipe, with time-stamped declared-demand inputs."""
import json
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor
import run_local
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from field_control.acn_requests import declaration_features
from field_control.acn_scoring import score_acn_residuals
from field_control.history_neighbors import date_block_interval

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'results/acn_fleet_months_v1'
OUT=DATA/'declared_demand_reference_v1'


def main():
    OUT.mkdir(exist_ok=True);start=time.monotonic()
    meta=pd.read_csv(DATA/'cases.csv.gz');population=pd.read_csv(DATA/'query_population.csv.gz')
    requests=pd.read_csv(DATA/'request_metadata_v1/query_requests.csv.gz')
    requests=requests.set_index('case_id').reindex(meta.case_id)
    if requests.site.isna().any():raise ValueError('Missing request-input row')
    with np.load(DATA/'packet.npz') as p:x,y=p['x'],p['y']
    delivered=np.load(DATA/'neural_packets_v1/history_exog.npy',mmap_mode='r')[:,-1,5]
    raw,gap=declaration_features(requests.requested_kwh,requests.requested_departure_hours,delivered)
    with np.load(DATA/'references/full.npz') as p:
        ix=pd.Index(p['case_id'].astype(str)).get_indexer(meta.case_id)
        if (ix<0).any():raise ValueError('Missing original telemetry forecast')
        baseline=p['prediction_kw'][ix]
    outputs={'telemetry':baseline};receipts=[]
    def fit(condition,extra):
        prediction=baseline.copy();info=[]
        for site in sorted(meta.site.unique()):
            ids=np.flatnonzero(meta.site==site);train=meta.iloc[ids].split.to_numpy()=='train'
            if not np.any(extra[ids]):
                info.append(dict(site=site,new_fits=0,reused_original=True,reason='No declarations in any input row at this site'))
                continue
            features=np.concatenate([x[ids],extra[ids]],axis=1)
            folder=OUT/condition/site;folder.mkdir(parents=True,exist_ok=True)
            began=time.monotonic()
            for lead in range(10):
                model=LGBMRegressor(n_jobs=2,random_state=2027,verbosity=-1)
                model.fit(features[train],y[ids[train],lead])
                prediction[ids,lead]=np.maximum(model.predict(features),0)
                model.booster_.save_model(str(folder/f'lead_{lead:02d}.txt'))
                print(json.dumps(dict(condition=condition,site=site,lead=lead+1,seconds=time.monotonic()-began)),flush=True)
            info.append(dict(site=site,new_fits=10,reused_original=False,training_cases=int(train.sum()),seconds=time.monotonic()-began,hyperparameters=model.get_params()))
        np.savez_compressed(OUT/(condition+'.npz'),case_id=meta.case_id.to_numpy(dtype=str),prediction_kw=prediction)
        return condition,prediction,info
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(fit,'declared',raw),pool.submit(fit,'declared_gap',np.concatenate([raw,gap],axis=1))]
        for job in jobs:
            condition,prediction,info=job.result();outputs[condition]=prediction;receipts.extend([dict(condition=condition,**v) for v in info])
    tables=[];days=[]
    for name,prediction in outputs.items():
        table,daily=score_acn_residuals(meta,prediction-y,name,population);tables.append(table);days.append(daily)
    summary=pd.concat(tables,ignore_index=True);daily=pd.concat(days,ignore_index=True)
    summary.to_csv(OUT/'summary.csv',index=False);daily.to_csv(OUT/'daily_scores.csv.gz',index=False)
    effects=[]
    for (site,split,horizon,scale),part in daily[daily.split.isin(['autumn','winter'])].groupby(['site','split','span_minutes','scale']):
        for metric in ['mse','mae','energy_ae_kwh']:
            p=part.pivot(index='local_date',columns='method',values=metric)
            for variant in ['declared','declared_gap']:
                lo,hi=date_block_interval(p.index,p.telemetry,p[variant])
                effects.append(dict(site=site,split=split,span_minutes=horizon,scale=scale,metric=metric,variant=variant,
                    base=p.telemetry.mean(),alternative=p[variant].mean(),reduction_pct=100*(1-p[variant].mean()/p.telemetry.mean()),ci_low=lo,ci_high=hi))
    pd.DataFrame(effects).to_csv(OUT/'effects.csv',index=False)
    view=summary[(summary.scale=='cohort_mean')&summary.split.isin(['autumn','winter'])]
    lines=['# ACN: declared demand on the same observed cohorts','',
        'Cohort-mean MSE / MAE, date-balanced. Same default LightGBM recipe and chronological split. Telemetry is the original complete power/pilot/state condition. Declared adds query-time requested kWh and departure, with masks; Declared + gap also adds max(requested minus already delivered kWh, 0) and its mask. All original queries/cohorts remain. Office has no observed declarations in these data, so its original forecasts are reused.','',
        '| Site / period | Span (min) | Telemetry | + declared demand | + declared demand and gap |',
        '|---|---:|---:|---:|---:|']
    for (site,split,horizon),part in view.groupby(['site','split','span_minutes']):
        part=part.set_index('method')
        cells=[f"{part.loc[n,'mse']:.3f} / {part.loc[n,'mae']:.3f}" for n in ['telemetry','declared','declared_gap']]
        lines.append(f'| {site} / {split} | {horizon} | '+' | '.join(cells)+' |')
    (ROOT/'paper/generated/acn_declared_demand.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    receipt=dict(completed=True,new_lead_fits=sum(r['new_fits'] for r in receipts),seconds=time.monotonic()-start,models=receipts,
        protocol='Official/default fixed tree recipe. No HPO, changed target, altered cohort or realized final-session outcome input.')
    (OUT/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('\n'.join(lines),flush=True);print(json.dumps(dict(completed=True,new_lead_fits=receipt['new_lead_fits'],seconds=receipt['seconds'])),flush=True)


if __name__=='__main__':main()
