"""Standard latest-window forecast at +30, scored on unchanged event targets.

This is a benchmark capability control, not a proposed method or HPO run.
"""
import argparse
import copy
import json
import time
from concurrent.futures import ThreadPoolExecutor,as_completed
import run_local
import numpy as np
import pandas as pd
from lightgbm import Booster,LGBMRegressor
from field_control import BpaDataset,read_response_submission,score_response_predictions
from bpa_update_validation_common import OUT,load_split,write_prediction
from prepare_bpa_model_packets import arrays_for,HIST_NAMES,STATIC_NAMES
from bpa_typed_inputs import encode
from analyze_bpa_update_validation import contrasts

DEST=OUT/'standard_refresh_v1'


def query_view(data,observed_minutes=30):
    if observed_minutes!=30:raise ValueError('This control is specified at +30 minutes')
    # Share the read-only grid and prefix; only predictor origins move.
    data._power_prefix()
    view=copy.copy(data);view.meta=data.meta.copy()
    view.meta['origin_minute']=view.meta.origin_minute+observed_minutes
    return view


def features(view,indices):
    ix=np.atleast_1d(indices);p=arrays_for(view,ix)
    # Unlike the original event origin, the refreshed minute may be offline.
    rows=view.meta.iloc[ix]
    online=view.grid['online'][rows.device_index.to_numpy(),rows.origin_minute.to_numpy()]
    p['static_exog'][~online,:6]=np.nan
    p['static_exog'][~online,6:12]=0
    p['static_exog'][~online,12:21]=0
    p['static_exog'][~online,21]=np.nan
    hist=encode(p['history_exog'],HIST_NAMES)
    static=encode(p['static_exog'],STATIC_NAMES)
    # At +30, local leads1..150 coincide with original minutes31..180.
    return np.concatenate([p['history'],hist.reshape(len(ix),-1),static,
        p['calendar'][:,96:106].reshape(len(ix),-1)],axis=1).astype(np.float32)


def all_features(data,indices):
    view=query_view(data);x=None
    for begin in range(0,len(indices),1024):
        stop=min(begin+1024,len(indices));value=features(view,indices[begin:stop])
        if x is None:x=np.empty((len(indices),value.shape[1]),dtype=np.float32)
        x[begin:stop]=value
    return x


def fit_lead(lead,x,y):
    path=DEST/f'lead_{lead:02d}.txt'
    if not path.exists():
        model=LGBMRegressor(n_jobs=2,random_state=2027,verbosity=-1)
        model.fit(x,y[:,lead]);model.booster_.save_model(str(path))
    return lead


def write_legacy_contrasts(daily):
    """Historical controls are optional; the main reference needs no old fits."""
    paired=contrasts(daily)
    # Add a paired direct comparison to the existing full-information model.
    swap=daily[daily.method.isin(['refresh','full'])].copy()
    swap['method']=swap.method.map({'refresh':'full','full':'initial'})
    # contrasts expects all four named conditions, so calculate its same weekly
    # cluster contrast without inventing additional predictions.
    extra=[]
    keys=['season','scale','command_type','target_end_minutes']
    for dims,frame in swap.groupby(keys):
        for metric in ['mse','mae','reported_energy_ae_kwh']:
            t=frame.pivot(index='local_date',columns='method',values=metric).sort_index()
            if t.isna().any().any():raise ValueError('Different comparison dates')
            dates=pd.to_datetime(t.index);week=(dates-pd.to_timedelta(dates.dayofweek,unit='D')).strftime('%Y-%m-%d')
            unique=sorted(set(week));index=np.array([unique.index(w) for w in week])
            count=np.random.default_rng(2027).multinomial(len(unique),np.ones(len(unique))/len(unique),size=1500)
            weights=count[:,index].astype(float);weights/=weights.sum(axis=1,keepdims=True)
            boot=weights@t.to_numpy();a=t.columns.get_loc('full');b=t.columns.get_loc('initial')
            ratio=100*(1-boot[:,a]/np.maximum(boot[:,b],np.finfo(float).tiny));lo,hi=np.quantile(ratio,[.025,.975])
            extra.append(dict(zip(keys,dims),metric=metric,method='refresh',reference='full',
                error=t['full'].mean(),reference_error=t.initial.mean(),reduction_pct=100*(1-t['full'].mean()/t.initial.mean()),
                ci_low=lo,ci_high=hi,dates=len(t),weeks=len(unique)))
    pd.concat([paired,pd.DataFrame(extra)],ignore_index=True).to_csv(DEST/'paired_contrasts.csv',index=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-only',action='store_true',
        help='Train/score the main reference without historical comparator predictions')
    args=parser.parse_args(argv)
    receipt_path=DEST/'receipt.json'
    if args.reference_only and receipt_path.exists():
        prior=json.loads(receipt_path.read_text())
        if prior.get('completed') and all((DEST/f'{p}.npz').is_file()
                for p in ['winter_validation','spring','summer']):
            print(json.dumps(dict(reused_completed_reference=True,**prior)),flush=True)
            return prior
    DEST.mkdir(parents=True,exist_ok=True);started=time.perf_counter();train,validation=load_split()
    data=BpaDataset(OUT,'winter');x=all_features(data,train);y=np.asarray(data.target[train,2:])
    width=x.shape[1]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(fit_lead,j,x,y) for j in range(10)]
        for f in as_completed(futures):print(json.dumps({'trained_lead':f.result(),'seconds':round(time.perf_counter()-started,1)}),flush=True)
    del x,y,data
    tables=[];days=[]
    for season in ['winter','spring','summer']:
        data=BpaDataset(OUT,season);ix=validation if season=='winter' else np.arange(len(data))
        period='winter_validation' if season=='winter' else season
        x=all_features(data,ix);prediction=np.empty((len(ix),10),dtype=np.float32)
        for j in range(10):
            model=Booster(model_file=str(DEST/f'lead_{j:02d}.txt'))
            prediction[:,j]=np.maximum(model.predict(x,num_threads=2),0)
        write_prediction(DEST/f'{period}.npz',data.meta.iloc[ix],prediction)
        table,daily=score_response_predictions(data.meta.iloc[ix],data.target[ix,2:],prediction,'refresh',period)
        tables.append(table);days.append(daily)
        if not args.reference_only:
            # Same scorer and case IDs for the optional historical comparators.
            for condition in ['initial','physical','command','full']:
                pred=read_response_submission(OUT/'update_validation_typed'/f'lightgbm_{condition}'/f'{period}.npz',data.meta.iloc[ix].case_id)
                a,b=score_response_predictions(data.meta.iloc[ix],data.target[ix,2:],pred,condition,period)
                tables.append(a);days.append(b)
        print(json.dumps({'scored':period,'cases':len(ix)}),flush=True)
        del x,prediction,data
    summary=pd.concat(tables,ignore_index=True);daily=pd.concat(days,ignore_index=True)
    summary.to_csv(DEST/'summary.csv',index=False);daily.to_csv(DEST/'daily_scores.csv.gz',index=False)
    if not args.reference_only:
        write_legacy_contrasts(daily)
    receipt=dict(completed=True,features=width,fit_count=10,train_cases=len(train),seconds=time.perf_counter()-started,
        comparison_mode='reference_only' if args.reference_only else 'legacy_controls',
        legacy_comparisons_included=not args.reference_only,
        recipe='Default LightGBM100trees/31leaves/MSE/seed2027; unchanged chronological split and targets; no HPO.',
        inputs='Latest24h at query+30; latest state/command; lead-aligned historical day/week and known calendar. Typed categorical fields. No minute-level extra prefix or separate original-origin snapshot.',
        comparison='A conventional rolling-window reference with the same target/cases. Input representation is intentionally different, not an isolated causal ablation.',
        labels='Original ten bins31..180 remain unchanged; chronological split uses original target end.')
    receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
    print(summary[(summary.scale=='cohort_mean')&(summary.command_type=='all')][['season','target_end_minutes','method','mse','mae']].round(5).to_string(index=False),flush=True)
    return receipt


if __name__=='__main__':main()
