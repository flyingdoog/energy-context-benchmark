"""Fixed-recipe record-linkage control; original-pairing models are reused."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import gc
import json
from pathlib import Path
import time
import run_local
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor,Booster
from field_control.record_linkage import reassign_blocks
from field_control.acn_scoring import score_acn_residuals
from field_control.response import score_response_predictions,read_response_submission,ABSOLUTE_BINS

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/record_linkage_v1'


def progress(output,value):
    value=dict(at=pd.Timestamp.now().isoformat(),**value)
    with (output/'progress.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(value)+'\n')
    print(json.dumps(value),flush=True)


def load_inputs(name,seed):
    folder=BASE/'support'/name
    receipt=json.loads((folder/'receipt.json').read_text())
    if not receipt['completed']:raise ValueError('Input support is incomplete')
    meta=pd.read_csv(folder/'cases.csv.gz',dtype={'member':str})
    x=np.load(folder/'features.npy',mmap_mode='r',allow_pickle=False)
    with np.load(folder/f'mapping_{seed}.npz',allow_pickle=False) as p:
        if not np.array_equal(p['case_id'].astype(str),meta.case_id.to_numpy(dtype=str)):
            raise ValueError('Saved mapping identities differ')
        donor,moved=p['donor_index'],p['moved_columns']
    return meta,x,donor,moved


def fit_heads(output,folder,x,y):
    folder.mkdir(parents=True,exist_ok=True)
    if len(x)!=len(y) or y.shape[1]!=10:raise ValueError('Invalid training matrix')
    started=time.monotonic()
    def fit(lead):
        path=folder/f'lead_{lead:02d}.txt'
        if path.exists():return lead,False
        model=LGBMRegressor(n_jobs=2,random_state=2027,verbosity=-1)
        model.fit(x,y[:,lead])
        model.booster_.save_model(str(path))
        return lead,True
    count=0
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(fit,i) for i in range(10)]
        for future in as_completed(futures):
            lead,new=future.result();count+=int(new)
            progress(output,dict(stage='lead_complete',source=folder.name,lead=lead,
                                 newly_fitted=new,seconds=time.monotonic()-started))
    record=dict(completed=True,training_cases=len(x),leads=10,new_fits_this_run=count,
        seconds=time.monotonic()-started,recipe='Default LightGBM100trees/31leaves/MSE/model-seed2027/two threads; no HPO',
        gpu_used=False)
    (folder/'fit_receipt.json').write_text(json.dumps(record,indent=2)+'\n')
    return [Booster(model_file=str(folder/f'lead_{lead:02d}.txt')) for lead in range(10)]


def predict(models,x,meta,path,source):
    if path.exists():
        with np.load(path,allow_pickle=False) as p:
            if not np.array_equal(p['case_id'].astype(str),meta.case_id.to_numpy(dtype=str)):
                raise ValueError('Saved prediction identities differ')
            return p['prediction_kw']
    result=np.column_stack([np.maximum(0,model.predict(x,num_threads=2)) for model in models]).astype(np.float32)
    extra={'absolute_bins':ABSOLUTE_BINS} if source=='BPA' else {'target_minutes':np.arange(0,151,15)}
    np.savez_compressed(path,case_id=meta.case_id.to_numpy(dtype=str),prediction_kw=result,**extra)
    return result


def normalize(table,source,site,period):
    if source=='BPA':
        table=table[table.command_type.astype(str)=='all'].copy()
        table=table.rename(columns={'reported_energy_ae_kwh':'energy_ae_kwh'})
        table['span_minutes']=table.target_end_minutes-30
    else:table=table.copy()
    return table.assign(source=source,site=site,period=period)


def acn(output,seed):
    base=ROOT/'results/acn_fleet_months_v1'
    original_meta=pd.read_csv(base/'cases.csv.gz')
    population=pd.read_csv(base/'query_population.csv.gz')
    with np.load(base/'packet.npz',allow_pickle=False) as p:y=p['y']
    meta,x,donor,moved=load_inputs('acn',seed)
    with np.load(base/'references/full.npz',allow_pickle=False) as p:
        baseline_index=pd.Index(p['case_id'].astype(str))
        baseline=p['prediction_kw']
    tables=[];days=[]
    for site in ['caltech','office001']:
        folder=output/f'acn_{site}'
        train=np.flatnonzero((meta.site==site)&(meta.period=='train')&(meta.original_index>=0))
        original=meta.original_index.iloc[train].to_numpy(dtype=int)
        if not ((original_meta.iloc[original].site==site)&(original_meta.iloc[original].split=='train')).all():
            raise ValueError('Training indices are outside the original training split')
        transformed=reassign_blocks(x,donor,moved,train)
        models=fit_heads(output,folder,transformed,y[original])
        del transformed;gc.collect()
        for period in ['validation','autumn','winter']:
            ix=np.flatnonzero((meta.site==site)&(meta.period==period)&(meta.original_index>=0))
            original=meta.original_index.iloc[ix].to_numpy(dtype=int)
            native=original_meta.iloc[original].reset_index(drop=True)
            transformed=reassign_blocks(x,donor,moved,ix)
            prediction=predict(models,transformed,native,folder/f'{period}.npz','ACN')
            order=baseline_index.get_indexer(native.case_id)
            if (order<0).any():raise ValueError('Missing original baseline predictions')
            for method,value in [('Paired',baseline[order]),('Reassigned',prediction)]:
                table,daily=score_acn_residuals(native,value-y[original],method,population)
                tables.append(normalize(table,'ACN',site,period))
                days.append(normalize(daily,'ACN',site,period))
            del transformed
            progress(output,dict(stage='scored',source='ACN',site=site,period=period,cases=len(ix)))
        del models;gc.collect()
    del x,y;gc.collect()
    return tables,days


def bpa(output,seed):
    base=ROOT/'results/benchmark_bpa_sequences_v1'
    meta,x,donor,moved=load_inputs('bpa_train',seed)
    if set(meta.period)!= {'train'}:raise ValueError('Unexpected BPA training split')
    target=np.load(base/'winter/target_kw.npy',mmap_mode='r',allow_pickle=False)
    indices=meta.original_index.to_numpy(dtype=int)
    with np.load(base/'update_validation/split.npz',allow_pickle=False) as p:
        if not np.array_equal(indices,p['train']):raise ValueError('BPA training indices differ')
    transformed=reassign_blocks(x,donor,moved)
    folder=output/'bpa'
    models=fit_heads(output,folder,transformed,np.asarray(target[indices,2:]))
    del transformed,x,target;gc.collect()
    tables=[];days=[]
    for period,season in [('validation','winter'),('spring','spring'),('summer','summer')]:
        meta,x,donor,moved=load_inputs('bpa_'+period,seed)
        native_all=pd.read_csv(base/season/'cases.csv.gz',dtype={'alias':str})
        indices=meta.original_index.to_numpy(dtype=int)
        native=native_all.iloc[indices].reset_index(drop=True)
        if not np.array_equal(native.case_id,meta.case_id):raise ValueError('BPA query identities differ')
        target=np.load(base/season/'target_kw.npy',mmap_mode='r',allow_pickle=False)
        y=np.asarray(target[indices,2:])
        transformed=reassign_blocks(x,donor,moved)
        name='winter_validation' if period=='validation' else period
        prediction=predict(models,transformed,native,folder/f'{name}.npz','BPA')
        baseline=read_response_submission(base/'standard_refresh_v1'/f'{name}.npz',native.case_id)
        for method,value in [('Paired',baseline),('Reassigned',prediction)]:
            table,daily=score_response_predictions(native,y,value,method,name)
            tables.append(normalize(table,'BPA','Field study',period))
            days.append(normalize(daily,'BPA','Field study',period))
        progress(output,dict(stage='scored',source='BPA',period=period,cases=len(meta)))
        del transformed,x,target,y;gc.collect()
    return tables,days


def write_comparison(output,table):
    use=table[(table.period!='validation')&(table.scale=='cohort_total')]
    means=use.groupby(['source','site','period','method'])[['mse','mae','energy_ae_kwh']].mean().reset_index()
    lines=['# Record-linkage control: first fixed mapping','',
           'Group-total errors average the three original spans. Paired predictions reuse',
           'the existing full-input reference. Reassigned models use the same fixed recipe;',
           'this is one mapping realization, not the three-mapping aggregate.','',
           '| Source / site | Period | Paired MSE / MAE | Reassigned MSE / MAE | Paired / reassigned energy AE (kWh) |',
           '|---|---|---:|---:|---:|']
    for key,frame in means.groupby(['source','site','period'],sort=False):
        values=frame.set_index('method')
        a,b=values.loc['Paired'],values.loc['Reassigned']
        lines.append(f"| {key[0]} / {key[1]} | {key[2]} | {a.mse:.3f} / {a.mae:.3f} | "
                     f"{b.mse:.3f} / {b.mae:.3f} | {a.energy_ae_kwh:.3f} / {b.energy_ae_kwh:.3f} |")
    (output/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mapping-seed',type=int,choices=[2027,2028,2029],default=2027)
    args=parser.parse_args()
    output=BASE/'experiments'/f'seed_{args.mapping_seed}'
    output.mkdir(parents=True,exist_ok=True)
    if (output/'receipt.json').exists():
        record=json.loads((output/'receipt.json').read_text())
        if not record.get('completed'):raise ValueError('Incomplete recorded experiment')
        print(json.dumps(dict(reused_completed_run=True,**record)),flush=True)
        return
    started=time.monotonic();tables=[];days=[]
    for source in ['acn','bpa']:
        a,b={'acn':acn,'bpa':bpa}[source](output,args.mapping_seed)
        tables+=a;days+=b
    summary=pd.concat(tables,ignore_index=True);daily=pd.concat(days,ignore_index=True)
    summary.to_csv(output/'summary.csv',index=False);daily.to_csv(output/'daily.csv.gz',index=False)
    write_comparison(output,summary)
    record=dict(completed=True,mapping_seed=args.mapping_seed,fit_count=30,
        baseline_refits=0,baseline_hpo=False,gpu_used=False,seconds=time.monotonic()-started,
        summary_rows=len(summary),training='Same original labeled training cases; only query-local state/control pairing changes')
    (output/'receipt.json').write_text(json.dumps(record,indent=2)+'\n')
    progress(output,dict(stage='batch_complete',**record))


if __name__=='__main__':
    main()
