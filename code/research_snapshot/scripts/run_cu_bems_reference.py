"""Fixed one-floor HVAC pilot: prepare, train-only profile, default fits, score."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import time

import run_local
import numpy as np
import pandas as pd
from lightgbm import Booster, LGBMRegressor
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from census_cu_bems import SOURCE, parse_columns, complete_windows
from field_control.cu_bems_scoring import make_groups, forecast_rows
from field_control.cu_bems_preparation import SPLITS, split_queries as split_origins

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/cu_bems_reference_pilot_v1'
TREE = dict(n_jobs=2, random_state=2027, verbosity=-1)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf8')


def frame_json(path, frame):
    frame.to_json(path, orient='records', indent=2)


def prepare():
    receipt = OUT / 'preparation.json'
    if receipt.exists() and json.loads(receipt.read_text())['completed']:
        return json.loads(receipt.read_text())
    start = time.monotonic()
    raw = pd.read_csv(SOURCE)
    clock = pd.DatetimeIndex(pd.to_datetime(raw['Date'], format='%Y-%m-%d %H:%M:%S'))
    if clock.has_duplicates or not np.all(np.diff(clock.as_unit('ns').asi8) == 60_000_000_000):
        raise ValueError('Unresolved source clock grid')
    devices = [x for x in parse_columns(raw.columns) if x['kind'] == 'AC']
    rows, arrays = [], {name: [] for name in ('power', 'thermal', 'calendar', 'target')}
    train_channels = []
    for device_id, item in enumerate(devices):
        values = raw[item['column']].to_numpy(dtype=np.float32)
        thermal = raw[[f"z{item['zone']}_S1(degC)", f"z{item['zone']}_S1(RH%)"]].to_numpy(dtype=np.float32)
        observed = np.isfinite(values)
        for split, begin, end in SPLITS:
            q = split_origins(clock, begin, end)
            q = q[complete_windows(observed, q, 60, 150)]
            history_ids = q[:, None] + np.arange(-60, 0)
            target_ids = q[:, None] + np.arange(150)
            target = values[target_ids].reshape(-1, 10, 15).mean(axis=2, dtype=np.float64).astype(np.float32)
            h = values[history_ids]
            s = thermal[history_ids]
            dates = clock[q]
            hour = dates.hour.to_numpy() + dates.minute.to_numpy()/60
            weekday = dates.dayofweek.to_numpy()
            calendar = np.column_stack([np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24),
                np.sin(2*np.pi*weekday/7), np.cos(2*np.pi*weekday/7)]).astype(np.float32)
            rows.append(pd.DataFrame(dict(case_id=[f'cu_f2_{device_id}_{i}' for i in q],
                device_index=device_id, device=item['column'], zone=item['zone'], split=split,
                query=dates.strftime('%Y-%m-%d %H:%M:%S'), date=dates.strftime('%Y-%m-%d'))))
            for name, value in [('power',h), ('thermal',s), ('calendar',calendar), ('target',target)]:
                arrays[name].append(value)
            if split == 'train':
                train_channels.append(dict(device=item['column'], cases=len(q),
                    mean_power=float(h.mean()), history_zero_fraction=float((h==0).mean()),
                    history_standard_deviation=float(h.std()), target_mean=float(target.mean()),
                    thermal_observed_fraction=float(np.isfinite(s).mean())))
    meta = pd.concat(rows, ignore_index=True)
    if meta.case_id.duplicated().any() or meta.duplicated(['query','device_index']).any():
        raise ValueError('Repeated device cases')
    OUT.mkdir(parents=True, exist_ok=True)
    meta.to_json(OUT/'cases.jsonl.gz', orient='records', lines=True, compression='gzip')
    for name, values in arrays.items():
        np.save(OUT/f'{name}.npy', np.concatenate(values))
    feature_recipe = dict(power='All 60 past power samples',
        thermal='60 past temperature/RH pairs, with 120 observed masks',
        calendar=['hour_sin','hour_cos','weekday_sin','weekday_cos'],
        identity='16 one-hot device indicators',
        missing='Sensors never filter cases. Tree: zero plus mask. Ridge: training-mean fill plus unscaled mask.',
        source='ACN history_features raw-sample/calendar pattern; HVAC has no pilot/state/elapsed-session fields.')
    support = meta.groupby('split').agg(cases=('case_id','size'),dates=('date','nunique'),devices=('device_index','nunique')).reset_index()
    groups = make_groups(meta)
    group_support = []
    for split, group, gmeta, ids in groups:
        group_support.append(dict(split=split, group=group, windows=len(ids), members=ids.shape[1], dates=gmeta.date.nunique()))
    result = dict(completed=True, seconds=time.monotonic()-start, cases=len(meta), devices=devices,
        support=support.to_dict('records'), group_support=group_support,
        source=str(SOURCE.relative_to(ROOT)), splits=SPLITS, history_minutes=60, bin_minutes=15,
        target_bins=10, query_stride_minutes=30, feature_recipe=feature_recipe,
        clock='Naive source calendar, no assumed UTC/receipt timestamps', new_downloads=0)
    dump(OUT/'training_channel_summary.json', train_channels)
    dump(receipt, result)
    print(json.dumps(dict(stage='prepared', seconds=result['seconds'],support=result['support'])), flush=True)
    return result


def load():
    meta = pd.read_json(OUT/'cases.jsonl.gz', orient='records', lines=True, convert_dates=False)
    packet = {name: np.load(OUT/f'{name}.npy', mmap_mode='r') for name in ('power','thermal','calendar','target')}
    packet['meta'] = meta
    return packet


def features(packet, condition, scaled=False, stats=None):
    p = np.asarray(packet['power'], dtype=np.float32)
    s = np.asarray(packet['thermal'], dtype=np.float32) if packet.get('thermal') is not None else None
    if condition == 'thermal' and s is None:
        raise ValueError('Thermal condition requires real source sensor fields')
    train = packet['meta'].split.to_numpy() == 'train'
    if scaled:
        if stats is None:
            stats = dict(power_mean=float(p[train].mean(dtype=np.float64)),
                power_scale=max(float(p[train].std(dtype=np.float64)),1e-6))
            if s is not None and condition == 'thermal':
                sample = s[train].astype(np.float64)
                stats.update(thermal_mean=np.nanmean(sample,axis=(0,1)).tolist(),
                    thermal_scale=np.maximum(np.nanstd(sample,axis=(0,1)),1e-6).tolist())
        p = (p-stats['power_mean'])/stats['power_scale']
        if condition == 'thermal':
            s = (s-np.asarray(stats['thermal_mean'],dtype=np.float32))/np.asarray(stats['thermal_scale'],dtype=np.float32)
    count = packet.get('device_count',16)
    columns = [p, np.asarray(packet['calendar']), np.eye(count,dtype=np.float32)[packet['meta'].device_index]]
    if condition == 'thermal':
        columns += [np.nan_to_num(s,nan=0,posinf=0,neginf=0).reshape(len(s),-1),np.isfinite(s).reshape(len(s),-1).astype(np.float32)]
    elif condition != 'power':
        raise ValueError('Unknown condition')
    x = np.concatenate(columns,axis=1)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite model inputs')
    return x, stats


def profile(packet):
    receipt=OUT/'profile.json'
    if receipt.exists() and json.loads(receipt.read_text()).get('completed'):
        return json.loads(receipt.read_text())
    train=np.flatnonzero(packet['meta'].split.to_numpy()=='train')
    rows=[]
    for condition in ('power','thermal'):
        folder=OUT/'models'/f'tree_{condition}';folder.mkdir(parents=True,exist_ok=True)
        x,_=features(packet,condition)
        info_path=folder/'lead_00.json'
        if info_path.exists():
            info=json.loads(info_path.read_text())
        else:
            started=time.monotonic();model=LGBMRegressor(**TREE)
            model.fit(x[train],packet['target'][train,0])
            seconds=time.monotonic()-started
            model.booster_.save_model(str(folder/'lead_00.txt'))
            info=dict(completed=True,lead=0,fit_seconds=seconds,training_cases=len(train),
                      condition=condition,parameters=model.get_params(),profile_train_only=True)
            dump(info_path,info)
        rows.append(info)
        print(json.dumps(dict(stage='train_only_profile',condition=condition,seconds=info['fit_seconds'])),flush=True)
    result=dict(completed=True,heads=2,profiles=rows,
        estimated_total_tree_fit_seconds=10*sum(x['fit_seconds'] for x in rows),
        note='First full default head per condition is retained and reused; estimate excludes full prediction/Ridge/scoring.')
    dump(receipt,result)
    return result


def fit_method(packet, family, condition, out=OUT):
    folder=out/'models'/f'{family}_{condition}';folder.mkdir(parents=True,exist_ok=True)
    receipt=folder/'receipt.json'
    if receipt.exists() and json.loads(receipt.read_text())['completed']:
        return json.loads(receipt.read_text())
    started=time.monotonic();train=np.flatnonzero(packet['meta'].split.to_numpy()=='train')
    x,stats=features(packet,condition,scaled=family=='ridge')
    y=packet['target'];new_fits=0;old_heads=0
    if family=='ridge':
        keep=np.ptp(x[train],axis=0)!=0
        model=Ridge()
        with threadpool_limits(limits=2):
            model.fit(x[train][:,keep].astype(np.float64),y[train].astype(np.float64))
            prediction=np.maximum(model.predict(x[:,keep].astype(np.float64)),0).astype(np.float32)
        np.savez_compressed(folder/'model.npz',coef=model.coef_,intercept=model.intercept_,keep=keep)
        dump(folder/'preprocessing.json',stats)
        parameters=model.get_params();new_fits=1
    else:
        prediction=np.empty(y.shape,dtype=np.float32)
        for lead in range(10):
            modelpath=folder/f'lead_{lead:02d}.txt';infopath=folder/f'lead_{lead:02d}.json'
            if modelpath.exists() and infopath.exists():
                model=Booster(model_file=str(modelpath));old_heads+=1
            else:
                fit_start=time.monotonic();estimator=LGBMRegressor(**TREE)
                estimator.fit(x[train],y[train,lead]);model=estimator.booster_;new_fits+=1
                model.save_model(str(modelpath))
                dump(infopath,dict(completed=True,lead=lead,fit_seconds=time.monotonic()-fit_start,
                     training_cases=len(train),condition=condition,parameters=estimator.get_params()))
            prediction[:,lead]=np.maximum(model.predict(x,num_threads=2),0)
            dump(folder/'progress.json',dict(leads_completed=lead+1,seconds=time.monotonic()-started))
            print(json.dumps(dict(stage='tree_head',floor=packet.get('floor',2),condition=condition,lead=lead+1,seconds=time.monotonic()-started)),flush=True)
        parameters=LGBMRegressor(**TREE).get_params()
    if not np.isfinite(prediction).all():raise ValueError('Invalid predictions')
    np.save(folder/'prediction.npy',prediction)
    result=dict(completed=True,family=family,condition=condition,seconds=time.monotonic()-started,
        new_fits=new_fits,reused_heads=old_heads,training_cases=len(train),cases=len(y),
        feature_columns=x.shape[1],parameters=parameters,validation_used_for_selection=False)
    dump(receipt,result)
    return result


def run(packet):
    start=time.monotonic();results=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(fit_method,packet,family,condition) for family in ('tree','ridge') for condition in ('power','thermal')]
        for future in as_completed(jobs):
            result=future.result();results.append(result)
            print(json.dumps(dict(stage='method_complete',**result)),flush=True)
    result=dict(completed=True,seconds=time.monotonic()-start,workers=2,methods=results,
        total_models='20 default tree heads (including 2 retained profile heads), 2 multi-output Ridge fits',gpu=False)
    dump(OUT/'fit_receipt.json',result)
    return result


def score(packet, out=OUT, definitions=None, conditions=('power','thermal')):
    start=time.monotonic();meta=packet['meta'];target=packet['target'];groups=make_groups(meta,definitions)
    predictions={'Persistence':np.repeat(packet['power'][:,-1,None],10,axis=1)}
    for family,label in [('ridge','Ridge'),('tree','Tree')]:
        for condition in conditions:
            predictions[f'{label}: {condition}']=np.load(out/'models'/f'{family}_{condition}'/'prediction.npy',mmap_mode='r')
    rows=[];days=[]
    for method,pred in predictions.items():
        residual=np.asarray(pred,dtype=np.float64)-target
        method_rows,method_days=forecast_rows(meta,residual,groups,method)
        rows.extend(method_rows);days.extend(method_days)
    summary=pd.DataFrame(rows);daily=pd.concat(days,ignore_index=True)
    frame_json(out/'summary.json',summary);frame_json(out/'daily_scores.json',daily)
    result=dict(completed=True,seconds=time.monotonic()-start,summary_rows=len(summary),daily_rows=len(daily),methods=list(predictions),
        scales=summary.scale.unique().tolist(),metrics=['mse','mae','energy_ae_kwh'])
    dump(out/'scoring.json',result)
    print(json.dumps(dict(stage='scored',**result)),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','profile','run','score']);args=parser.parse_args()
    if args.stage=='prepare':prepare()
    else:
        packet=load()
        if args.stage=='profile':profile(packet)
        elif args.stage=='run':run(packet)
        else:score(packet)
