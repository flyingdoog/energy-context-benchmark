"""Complete three-seed, equal-width comparison of declared-demand inputs."""
import json
from pathlib import Path
import run_local
import numpy as np
import pandas as pd
from field_control.acn_scoring import score_acn_residuals
from field_control.history_neighbors import date_block_interval

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'results/acn_fleet_months_v1'
REF=DATA/'declared_demand_neural_v1'


def main():
    out=REF/'analysis_mean3';out.mkdir(exist_ok=True)
    meta=pd.read_csv(DATA/'cases.csv.gz');pop=pd.read_csv(DATA/'query_population.csv.gz')
    ix=np.flatnonzero(meta.site=='caltech');meta=meta.iloc[ix].reset_index(drop=True)
    y=np.load(DATA/'neural_packets_v1/target.npy',mmap_mode='r')[ix]
    scores=[];days=[];receipts=[]
    for view in ['padded','raw','gap']:
        for seed in [1,2,3]:
            folder=REF/'caltech'/f'nhits_seed{seed}_full_{view}'
            receipt=json.loads((folder/'receipt.json').read_text())
            if not receipt.get('completed'):raise ValueError('A required model is not complete')
            receipts.append(receipt)
            with np.load(folder/'predictions.npz') as p:
                order=pd.Index(p['case_id'].astype(str)).get_indexer(meta.case_id)
                if (order<0).any():raise ValueError('Missing prediction case')
                prediction=p['prediction_kw'][order]
            summary,daily=score_acn_residuals(meta,prediction-y,view,pop)
            scores.append(summary.assign(seed=seed));days.append(daily.assign(seed=seed))
    if len({r['parameters'] for r in receipts})!=1:raise ValueError('Unequal network widths')
    scores=pd.concat(scores,ignore_index=True);days=pd.concat(days,ignore_index=True)
    scores.to_csv(out/'per_seed_scores.csv',index=False)
    keys=['site','split','method','scale','span_minutes'];metrics=['mse','mae','energy_ae_kwh']
    summary=scores.groupby(keys)[metrics].mean().join(scores.groupby(keys)[metrics].std().add_suffix('_seed_sd')).reset_index()
    daily=days.groupby(keys+['local_date'])[metrics].mean().reset_index()
    summary.to_csv(out/'summary.csv',index=False);daily.to_csv(out/'daily_scores.csv.gz',index=False)
    effects=[]
    for (split,horizon,scale),part in daily[daily.split.isin(['autumn','winter'])].groupby(['split','span_minutes','scale']):
        for metric in metrics:
            p=part.pivot(index='local_date',columns='method',values=metric)
            for method in ['raw','gap']:
                lo,hi=date_block_interval(p.index,p.padded,p[method])
                effects.append(dict(split=split,span_minutes=horizon,scale=scale,metric=metric,method=method,
                    baseline=p.padded.mean(),alternative=p[method].mean(),reduction_pct=100*(1-p[method].mean()/p.padded.mean()),ci_low=lo,ci_high=hi))
    pd.DataFrame(effects).to_csv(out/'effects.csv',index=False)
    selected=summary[(summary.scale=='cohort_mean')&summary.split.isin(['autumn','winter'])]
    lines=['# ACN Caltech: declared demand in N-HiTS','',
        'Cohort-mean MSE / MAE. Mean errors across three independently trained models; same cohorts, 1,000-step MSE recipe and static-input width. The telemetry control pads the six added input positions with zeros. Declarations are selected as of each query. No hyperparameter search.','',
        '| Period | Span (min) | Telemetry | + declared demand | + declared demand and gap |',
        '|---|---:|---:|---:|---:|']
    for (split,h),part in selected.groupby(['split','span_minutes']):
        p=part.set_index('method')
        cells=[f"{p.loc[n,'mse']:.3f} / {p.loc[n,'mae']:.3f}" for n in ['padded','raw','gap']]
        lines.append(f'| {split} | {h} | '+' | '.join(cells)+' |')
    (ROOT/'paper/generated/acn_declared_demand_nhits.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    receipt=dict(completed=True,seeds=[1,2,3],fits=len(receipts),parameters=receipts[0]['parameters'],
        total_fit_seconds=sum(r['fit_seconds'] for r in receipts),
        aggregation='Mean of model errors, not predictions. Date bootstrap conditional on these training seeds; seed dispersion saved separately.')
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('\n'.join(lines),flush=True);print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
