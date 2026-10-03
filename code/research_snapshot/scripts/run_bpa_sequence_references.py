"""Full-information default Ridge/LightGBM references and transparent templates."""
import argparse
import gc
import importlib.metadata
import json
from pathlib import Path
import time
import run_local
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from field_control import BpaDataset,score_predictions,read_submission
from build_bpa_sequence_release import OUT
from prepare_bpa_model_packets import PACKETS,STATIC_NAMES

DEST=OUT/'references'


def packet(season):
    return {name:np.load(PACKETS/season/f'{name}.npy',mmap_mode='r')
            for name in ['history','history_exog','static_exog','calendar','target']}


def tabular(p):
    n=len(p['history'])
    return np.concatenate([p['history'],p['history_exog'].reshape(n,-1),
        p['static_exog'],p['calendar'][:,96:].reshape(n,-1)],axis=1)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--family',choices=['templates','ridge','lightgbm'],required=True)
    args=parser.parse_args();DEST.mkdir(exist_ok=True)
    name=args.family;folder=DEST/name;folder.mkdir(exist_ok=True)
    if (folder/'receipt.json').exists():print('Complete reference reused',flush=True);return
    started=time.perf_counter();data={s:BpaDataset(OUT,s) for s in ['winter','spring','summer']}
    packs={s:packet(s) for s in data};predictions={s:{} for s in ['spring','summer']};fit_seconds=0
    if name=='templates':
        for s in predictions:
            p=packs[s];indices=[STATIC_NAMES.index(f'previous_day_lead_{j}') for j in range(12)]
            week_indices=[STATIC_NAMES.index(f'previous_week_lead_{j}') for j in range(12)]
            persistence=np.repeat(p['static_exog'][:,:1],12,axis=1)
            # Fixed fallback, including on offline previous-day/week windows.
            day=np.where(np.isfinite(p['static_exog'][:,indices]),p['static_exog'][:,indices],persistence)
            week=np.where(np.isfinite(p['static_exog'][:,week_indices]),p['static_exog'][:,week_indices],persistence)
            shed=day.copy();shed[data[s].meta.command_type==1]=0
            predictions[s]=dict(current_power=persistence,previous_day=day,previous_week=week,
                                day_week_mean=.5*(day+week),shed_zero_other_day=shed)
    else:
        x={s:tabular(p) for s,p in packs.items()};tick=time.perf_counter()
        if name=='ridge':
            model=make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler(),Ridge())
            model.fit(x['winter'],packs['winter']['target'])
            fit_seconds=time.perf_counter()-tick
            for s in predictions:predictions[s][name]=np.maximum(model.predict(x[s]),0)
            import joblib
            joblib.dump(model,folder/'model.joblib')
            params=model[-1].get_params()
        else:
            from lightgbm import LGBMRegressor
            for s in predictions:predictions[s][name]=np.empty_like(packs[s]['target'])
            for lead in range(12):
                model=LGBMRegressor(n_jobs=2,random_state=2027,verbosity=-1)
                model.fit(x['winter'],packs['winter']['target'][:,lead])
                model.booster_.save_model(str(folder/f'lead_{lead:02d}.txt'))
                for s in predictions:predictions[s][name][:,lead]=np.maximum(model.predict(x[s]),0)
                print(json.dumps({'reference':name,'leads':lead+1,'seconds':time.perf_counter()-tick}),flush=True)
            fit_seconds=time.perf_counter()-tick;params=model.get_params()
        feature_count=x['winter'].shape[1]
        del x;gc.collect()
    tables=[];days=[]
    for season,models in predictions.items():
        for method,prediction in models.items():
            path=folder/f'{season}_{method}.npz';data[season].submission(path,prediction)
            # Score the actual on-disk reference-submission contract.
            aligned=read_submission(path,data[season].meta.case_id)
            table,daily=score_predictions(data[season].meta,data[season].target,aligned,method,season)
            tables.append(table);days.append(daily)
    pd.concat(tables).to_csv(folder/'scores.csv',index=False)
    pd.concat(days).to_csv(folder/'daily_scores.csv.gz',index=False)
    receipt=dict(family=name,completed=True,seconds=time.perf_counter()-started,fit_seconds=fit_seconds,
        cases={s:len(d) for s,d in data.items()},training_season='winter',hpo=0,
        features=feature_count if name!='templates' else None,
        parameters=params if name!='templates' else None,
        recipe='Default estimator recipes. Ridge: training-only median/indicator and standardization. LightGBM: 100 trees,31leaves,default squared error,one estimator per lead. Predictions clipped to nonnegative reported kW for all learned references.',
        information='All learned references receive 24h power/state/command histories, current power/state/command, historical day/week profiles, and future calendar.',
        versions={p:importlib.metadata.version(p) for p in ['numpy','scikit-learn']+(['lightgbm'] if name=='lightgbm' else [])})
    (folder/'receipt.json').write_text(json.dumps(receipt,indent=2,default=str)+'\n')
    rows=pd.concat(tables);show=rows[(rows.scale=='cohort_mean')&(rows.command_type=='all')]
    print(show[['season','method','horizon_minutes','mse','mae']].to_string(index=False),flush=True)


if __name__=='__main__':main()
