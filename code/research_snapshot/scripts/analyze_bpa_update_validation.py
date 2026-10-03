"""Score completed chronological conditions and validation-only simple blends."""
import argparse
import json
import run_local
import numpy as np
import pandas as pd
from bpa_fixed_target_update import score,read_updated,DIMENSIONS
from bpa_update_validation_common import (DEST,CONDITIONS,load,load_split,
    evaluation_parts,recent_power,blend_coefficient,write_prediction)


def fit_group_blend(meta,base,recent,target):
    keys=['local_date','cohort_id','command_type']
    bcols=[f'b{j}' for j in range(10)];ycols=[f'y{j}' for j in range(10)]
    table=meta[keys].reset_index(drop=True).copy()
    table[bcols]=base;table[ycols]=target;table['recent']=recent
    grouped=table.groupby(keys,sort=False)
    means=grouped[bcols+ycols+['recent']].mean().reset_index()
    means['devices']=grouped.size().to_numpy();means=means[means.devices>=10]
    weights=1/means.groupby('local_date').local_date.transform('size').to_numpy()
    alpha=blend_coefficient(means[bcols].to_numpy(),means.recent.to_numpy(),
                            means[ycols].to_numpy(),weights)
    return alpha,dict(groups=len(means),dates=int(means.local_date.nunique()))


def contrasts(daily):
    results=[];cache={}
    for dims,frame in daily.groupby(DIMENSIONS):
        for metric in ['mse','mae','reported_energy_ae_kwh']:
            table=frame.pivot(index='local_date',columns='method',values=metric).sort_index()
            if table.isna().any().any():raise ValueError('Different date support between methods')
            key=tuple(table.index)
            if key not in cache:
                dates=pd.to_datetime(table.index)
                week=(dates-pd.to_timedelta(dates.dayofweek,unit='D')).strftime('%Y-%m-%d')
                unique=sorted(set(week));index=np.array([unique.index(w) for w in week])
                counts=np.random.default_rng(2027).multinomial(len(unique),np.ones(len(unique))/len(unique),size=1500)
                weights=counts[:,index].astype(float);weights/=weights.sum(axis=1,keepdims=True)
                cache[key]=weights,len(unique)
            weights,nweeks=cache[key];boot=weights@table.to_numpy();means=table.mean()
            pairs=[(m,'initial') for m in table.columns if m!='initial']
            pairs+=[('full','physical'),('full','command')]
            pairs+=[('full',m) for m in ['blend_device','blend_group'] if m in table.columns]
            for method,reference in pairs:
                a=table.columns.get_loc(method);b=table.columns.get_loc(reference)
                ratio=100*(1-boot[:,a]/np.maximum(boot[:,b],np.finfo(float).tiny))
                lo,hi=np.quantile(ratio,[.025,.975])
                results.append(dict(zip(DIMENSIONS,dims),metric=metric,method=method,reference=reference,
                    error=means[method],reference_error=means[reference],
                    reduction_pct=100*(1-means[method]/means[reference]) if means[reference]>0 else np.nan,
                    ci_low=lo,ci_high=hi,dates=len(table),weeks=nweeks))
    return pd.DataFrame(results)


def main():
    global DEST
    parser=argparse.ArgumentParser();parser.add_argument('--family',choices=['lightgbm','nhits_seed1','nhits_seed2','nhits_seed3'],required=True)
    parser.add_argument('--encoding',choices=['raw','typed'],default='raw')
    args=parser.parse_args();family=args.family
    if args.encoding=='typed':DEST=DEST.parent/'update_validation_typed'
    for c in CONDITIONS:
        path=DEST/f'{family}_{c}'/'receipt.json'
        if not path.exists() or not json.loads(path.read_text()).get('completed'):
            raise ValueError(f'Wait for actual completed {family}/{c} fit')
    folder=DEST/f'analysis_{family}';folder.mkdir(exist_ok=True)
    train,validation=load_split();packets={s:load(s) for s in ['winter','spring','summer']}
    parts=evaluation_parts(packets,validation)
    p=packets['winter'];meta=p['meta'].iloc[validation];target=np.asarray(p['target'][validation,2:])
    base=read_updated(DEST/f'{family}_initial/winter_validation.npz',meta.case_id)
    recent=recent_power(p,validation)
    alphas={'blend_device':blend_coefficient(base,recent,target)}
    alphas['blend_group'],group_support=fit_group_blend(meta,base,recent,target)
    (folder/'calibration.json').write_text(json.dumps(dict(coefficients={k:v.tolist() for k,v in alphas.items()},
        calibration_period='Winter validation only; no calibration fit scores presented as held-out estimates.',
        individual_objective='Equal-window point MSE, ten independent closed-form convex coefficients.',
        group_objective='Date-balanced synchronous-group-mean MSE, groups>=10, ten closed-form convex coefficients.',
        signal='Mean observed power in minutes16to30; forecast targets remain minutes31to180.',
        group_support=group_support),indent=2)+'\n')
    tables=[];days=[]
    for period,p,indices in parts:
        meta=p['meta'].iloc[indices];target=np.asarray(p['target'][indices,2:])
        predictions={c:read_updated(DEST/f'{family}_{c}'/f'{period}.npz',meta.case_id) for c in CONDITIONS}
        if period!='winter_validation':
            base=predictions['initial'];recent=recent_power(p,indices)
            for name,alpha in alphas.items():
                pred=base+alpha[None,:]*(recent[:,None]-base)
                predictions[name]=pred;write_prediction(folder/f'{period}_{name}.npz',meta,pred)
        for method,pred in predictions.items():
            table,daily=score(meta,target,pred,method,period)
            tables.append(table);days.append(daily)
    summary=pd.concat(tables,ignore_index=True);daily=pd.concat(days,ignore_index=True)
    summary['family']=family;daily['family']=family
    summary['encoding']=args.encoding;daily['encoding']=args.encoding
    summary.to_csv(folder/'summary.csv',index=False);daily.to_csv(folder/'daily_scores.csv.gz',index=False)
    paired=contrasts(daily);paired['family']=family;paired['encoding']=args.encoding;paired.to_csv(folder/'paired_contrasts.csv',index=False)
    mainrows=summary[(summary.scale=='cohort_mean')&(summary.command_type=='all')]
    val=mainrows[mainrows.season=='winter_validation'].groupby('method').mse.mean()
    selected=val.idxmin()
    (folder/'validation_selection.json').write_text(json.dumps(dict(selected_condition=selected,
        criterion='Mean date-balanced group MSE across fixed endpoints60/120/180 on winter validation; same condition for all endpoints.',
        validation_mse=val.to_dict(),calibrated_rules_excluded='Their coefficients use this block; compare only on spring/summer.'),indent=2)+'\n')
    mainrows[['family','season','method','target_end_minutes','mse','mae','dates','units']].to_csv(folder/'group_comparison.csv',index=False)
    (folder/'receipt.json').write_text(json.dumps(dict(completed=True,family=family,encoding=args.encoding,
        selection=selected,model_fits=4,calibrated_rules=2,
        comparison='Separate reduced-training chronological replication, not directly comparable to full-winter pilot numbers.',
        uncertainty='Paired calendar-week cluster intervals conditional on one model seed and this chronological split.'),indent=2)+'\n')
    print(mainrows[['season','target_end_minutes','method','mse','mae']].sort_values(['season','target_end_minutes','method']).to_string(index=False),flush=True)
    print(json.dumps(dict(validation_selected=selected,validation_mse=val.to_dict())),flush=True)


if __name__=='__main__':main()
