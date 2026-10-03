"""Value of control/state fields at the same standard refreshed query."""
import json
import time
from concurrent.futures import ThreadPoolExecutor,as_completed
import run_local
import numpy as np
import pandas as pd
from lightgbm import Booster,LGBMRegressor
from field_control import BpaDataset,read_response_submission,score_response_predictions
from bpa_standard_refresh import all_features,DEST as REFERENCE
from bpa_update_validation_common import OUT,load_split,write_prediction
from bpa_typed_inputs import expanded
from prepare_bpa_model_packets import HIST_NAMES,STATIC_NAMES,CALENDAR_NAMES

DEST=OUT/'telemetry_value_v1'
CONDITIONS=('power','power_commands','power_state','full')
SCORE_KEYS=['season','scale','command_type','target_end_minutes']
METRICS=['mse','mae','reported_energy_ae_kwh']


def feature_names():
    return ([f'history_power:{j}' for j in range(96)]+
        [f'history{j}:{name}' for j in range(96) for name in expanded(HIST_NAMES)]+
        [f'snapshot:{name}' for name in expanded(STATIC_NAMES)]+
        [f'calendar{j}:{name}' for j in range(10) for name in CALENDAR_NAMES])


def field_group(name):
    prefix,leaf=name.split(':',1)
    if prefix=='history_power' or prefix.startswith('calendar'):return 'power'
    if 'command' in leaf:return 'commands'
    if any(leaf==n or leaf.startswith(n+'_') for n in ['present','capacity','oper_state','running','opt_out']):return 'state'
    if leaf in ['power_kw','power_kw_observed','power_observed_fraction'] or leaf.startswith('previous_'):return 'power'
    raise ValueError(f'Unclassified field {name}')


def active_mask(condition):
    allowed={'power':{'power'},'power_commands':{'power','commands'},
             'power_state':{'power','state'},'full':{'power','commands','state'}}
    if condition not in allowed:raise ValueError('Unknown information condition')
    return np.array([field_group(name) in allowed[condition] for name in feature_names()])


def restrict(x,condition):
    if x.ndim!=2 or x.shape[1]!=len(feature_names()):raise ValueError('Wrong refreshed feature layout')
    x=np.array(x,copy=True);x[:,~active_mask(condition)]=0
    return x


def fit_condition(condition,base,y):
    folder=DEST/condition;folder.mkdir(exist_ok=True);start=time.perf_counter()
    x=restrict(base,condition)
    for lead in range(10):
        path=folder/f'lead_{lead:02d}.txt'
        if not path.exists():
            model=LGBMRegressor(n_jobs=2,random_state=2027,verbosity=-1)
            model.fit(x,y[:,lead]);model.booster_.save_model(str(path))
        progress=dict(condition=condition,trained_leads=lead+1,seconds=round(time.perf_counter()-start,1))
        (folder/'progress.json').write_text(json.dumps(progress)+'\n');print(json.dumps(progress),flush=True)
    result=dict(completed=True,condition=condition,features=x.shape[1],active_columns=int(active_mask(condition).sum()),
        train_cases=len(x),seconds=time.perf_counter()-start,
        recipe='Default LightGBM100trees/31leaves/MSE/seed2027,2threads. No HPO. Removed fields constant zero.')
    (folder/'fit_receipt.json').write_text(json.dumps(result,indent=2)+'\n');return result


def paired_contrasts(daily):
    pairs=[('power_commands','power'),('power_state','power'),('full','power'),
           ('full','power_commands'),('full','power_state')]
    rows=[];cache={}
    for dims,frame in daily.groupby(SCORE_KEYS):
        for metric in METRICS:
            t=frame.pivot(index='local_date',columns='method',values=metric).sort_index()
            if t.isna().any().any() or set(t.columns)!=set(CONDITIONS):raise ValueError('Unequal paired support')
            key=tuple(t.index)
            if key not in cache:
                dates=pd.to_datetime(t.index);weeks=(dates-pd.to_timedelta(dates.dayofweek,unit='D')).strftime('%Y-%m-%d')
                unique=sorted(set(weeks));index=np.array([unique.index(w) for w in weeks])
                count=np.random.default_rng(2027).multinomial(len(unique),np.ones(len(unique))/len(unique),size=1500)
                weights=count[:,index].astype(float);weights/=weights.sum(axis=1,keepdims=True)
                cache[key]=(weights,len(unique))
            weights,nweeks=cache[key];boot=weights@t.to_numpy();means=t.mean()
            for method,reference in pairs:
                a=t.columns.get_loc(method);b=t.columns.get_loc(reference)
                relative=100*(1-boot[:,a]/np.maximum(boot[:,b],np.finfo(float).tiny))
                lo,hi=np.quantile(relative,[.025,.975])
                rows.append(dict(zip(SCORE_KEYS,dims),metric=metric,method=method,reference=reference,
                    error=means[method],reference_error=means[reference],
                    reduction_pct=100*(1-means[method]/means[reference]),ci_low=lo,ci_high=hi,dates=len(t),weeks=nweeks))
    return pd.DataFrame(rows)


def main():
    DEST.mkdir(exist_ok=True);start=time.perf_counter()
    reference=json.loads((REFERENCE/'receipt.json').read_text())
    if not reference.get('completed'):raise ValueError('Finish the rich-input rolling reference first')
    train,validation=load_split();data=BpaDataset(OUT,'winter')
    base=all_features(data,train);y=np.asarray(data.target[train,2:]);names=feature_names()
    if base.shape[1]!=len(names):raise ValueError('Feature names do not match actual design')
    (DEST/'input_groups.json').write_text(json.dumps(dict(names=names,groups=[field_group(n) for n in names],
        active_columns={c:int(active_mask(c).sum()) for c in CONDITIONS}),indent=2)+'\n')
    results=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(fit_condition,c,base,y) for c in CONDITIONS if c!='full']
        for future in as_completed(futures):results.append(future.result())
    del base,y,data
    tables=[];days=[]
    for season in ['winter','spring','summer']:
        data=BpaDataset(OUT,season);ix=validation if season=='winter' else np.arange(len(data))
        period='winter_validation' if season=='winter' else season;meta=data.meta.iloc[ix]
        base=all_features(data,ix)
        for condition in CONDITIONS:
            if condition=='full':
                prediction=read_response_submission(REFERENCE/f'{period}.npz',meta.case_id)
            else:
                x=restrict(base,condition);prediction=np.empty((len(ix),10),dtype=np.float32)
                for lead in range(10):
                    model=Booster(model_file=str(DEST/condition/f'lead_{lead:02d}.txt'))
                    prediction[:,lead]=np.maximum(model.predict(x,num_threads=2),0)
                write_prediction(DEST/condition/f'{period}.npz',meta,prediction);del x
            table,daily=score_response_predictions(meta,data.target[ix,2:],prediction,condition,period)
            tables.append(table);days.append(daily)
        print(json.dumps({'scored':period,'cases':len(ix)}),flush=True);del base,prediction,data
    summary=pd.concat(tables,ignore_index=True);daily=pd.concat(days,ignore_index=True)
    summary.to_csv(DEST/'summary.csv',index=False);daily.to_csv(DEST/'daily_scores.csv.gz',index=False)
    paired_contrasts(daily).to_csv(DEST/'paired_contrasts.csv',index=False)
    val=summary[(summary.season=='winter_validation')&(summary.scale=='cohort_mean')&(summary.command_type=='all')].groupby('method').mse.mean()
    result=dict(completed=True,new_fits=30,reused_rich_input_fits=10,seconds=time.perf_counter()-start,
        query_minutes=30,fixed_targets='Original minutes31..180; all cases and groups unchanged.',
        information_conditions=list(CONDITIONS),fits=results,validation_macro_mse=val.to_dict(),
        best_validation_condition=val.idxmin(),
        interpretation='Information-value comparison at one query, not a proposed-method gain or causal effect.',
        uncertainty='Paired calendar-week intervals conditional on a fixed default tree recipe and seed.')
    (DEST/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    print(summary[(summary.scale=='cohort_mean')&(summary.command_type=='all')][['season','target_end_minutes','method','mse','mae']].round(5).to_string(index=False),flush=True)


if __name__=='__main__':main()
