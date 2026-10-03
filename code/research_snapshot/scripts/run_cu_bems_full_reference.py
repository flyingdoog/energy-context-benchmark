"""All remaining original AC floors, with the completed Floor 2 pilot reused."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import json
from pathlib import Path
import time
import run_local
import numpy as np
import pandas as pd
from run_cu_bems_reference import OUT as PILOT, SPLITS, split_origins, complete_windows
from run_cu_bems_reference import dump, frame_json, fit_method, score, make_groups
from census_cu_bems_source import BASE as SOURCE_BASE, source_grid, cohorts

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/cu_bems_full_reference_v1'
NEW_FLOORS=[1,3,4,5,6,7]


def floor_output(floor):
    return PILOT if floor==2 else OUT/f'floor{floor}'


def prepare(floor):
    out=floor_output(floor)
    if floor==2:
        return json.loads((out/'preparation.json').read_text())
    receipt=out/'preparation.json'
    if receipt.exists() and json.loads(receipt.read_text()).get('completed'):
        return json.loads(receipt.read_text())
    started=time.monotonic()
    census=json.loads((SOURCE_BASE/'full_2019'/f'floor{floor}.json').read_text())
    devices=census['devices'];mapping=census['sensor_mapping']
    has_thermal=census['ac_with_thermal_fields']>0
    if has_thermal and census['ac_with_thermal_fields']!=len(devices):
        raise ValueError('Inspect partially absent per-floor sensors before using this packet')
    raw,clock,quality=source_grid(pd.read_csv(SOURCE_BASE/census['source']))
    arrays={name:[] for name in ['power','calendar','target']+(['thermal'] if has_thermal else [])}
    rows=[];training=[]
    for device_id,item in enumerate(devices):
        power=raw[item['column']].to_numpy(dtype=np.float32)
        sensor=raw[mapping[str(item['zone'])]].to_numpy(dtype=np.float32) if has_thermal else None
        for split,begin,end in SPLITS:
            q=split_origins(clock,begin,end)
            q=q[complete_windows(np.isfinite(power),q,60,150)]
            if not len(q):continue
            hist=q[:,None]+np.arange(-60,0)
            future=q[:,None]+np.arange(150)
            p=power[hist]
            y=power[future].reshape(-1,10,15).mean(axis=2,dtype=np.float64).astype(np.float32)
            stamps=clock[q];hours=stamps.hour.to_numpy()+stamps.minute.to_numpy()/60
            week=stamps.dayofweek.to_numpy()
            calendar=np.column_stack([np.sin(2*np.pi*hours/24),np.cos(2*np.pi*hours/24),
                np.sin(2*np.pi*week/7),np.cos(2*np.pi*week/7)]).astype(np.float32)
            arrays['power'].append(p);arrays['target'].append(y);arrays['calendar'].append(calendar)
            if has_thermal:arrays['thermal'].append(sensor[hist])
            rows.append(pd.DataFrame(dict(case_id=[f'cu_f{floor}_{device_id}_{i}' for i in q],
                floor=floor,device_index=device_id,device=item['column'],zone=item['zone'],split=split,
                query=stamps.strftime('%Y-%m-%d %H:%M:%S'),date=stamps.strftime('%Y-%m-%d'))))
            if split=='train':
                training.append(dict(device=item['column'],cases=len(q),mean_power=float(p.mean()),
                    history_zero_fraction=float((p==0).mean()),history_standard_deviation=float(p.std()),
                    thermal_observed_fraction=float(np.isfinite(sensor[hist]).mean()) if has_thermal else None))
    meta=pd.concat(rows,ignore_index=True)
    if meta.case_id.duplicated().any():raise ValueError('Repeated device case')
    if len(training)!=len(devices):raise ValueError('A source device has no training examples')
    definitions=cohorts(devices)
    groups=make_groups(meta,definitions)
    support=meta.groupby('split').agg(cases=('case_id','size'),dates=('date','nunique'),devices=('device_index','nunique')).reset_index()
    expected=[x for x in census['support'] if x['level']=='device']
    for split,_,_ in SPLITS:
        actual=int((meta.split==split).sum())
        if actual!=sum(x['power_windows'] for x in expected if x['split']==split):
            raise ValueError(f'Floor {floor} {split}: prepared cases differ from completed source support')
    group_support=[dict(split=s,group=g,windows=len(ids),members=ids.shape[1],dates=gmeta.date.nunique()) for s,g,gmeta,ids in groups]
    out.mkdir(parents=True,exist_ok=True)
    meta.to_json(out/'cases.jsonl.gz',orient='records',lines=True,compression='gzip')
    for name,values in arrays.items():np.save(out/f'{name}.npy',np.concatenate(values))
    result=dict(completed=True,floor=floor,seconds=time.monotonic()-started,cases=len(meta),devices=devices,
        device_count=len(devices),has_thermal=has_thermal,support=support.to_dict('records'),
        group_support=group_support,groups=definitions,splits=SPLITS,source_quality=quality,
        feature_recipe='Unchanged pilot power/calendar, optional real thermal history plus masks; actual device one-hot dimension',
        history_minutes=60,target_bins=10,bin_minutes=15,query_stride_minutes=30)
    dump(out/'training_channel_summary.json',training);dump(receipt,result)
    print(json.dumps(dict(stage='prepared',floor=floor,cases=len(meta),seconds=result['seconds'])),flush=True)
    return result


def load(floor):
    out=floor_output(floor)
    info=json.loads((out/'preparation.json').read_text())
    meta=pd.read_json(out/'cases.jsonl.gz',orient='records',lines=True,convert_dates=False)
    packet={n:np.load(out/f'{n}.npy',mmap_mode='r') for n in ('power','calendar','target')}
    packet['thermal']=np.load(out/'thermal.npy',mmap_mode='r') if info.get('has_thermal',True) else None
    packet.update(meta=meta,device_count=len(info['devices']),floor=floor)
    return packet,info


def run_floor(floor):
    started=time.monotonic();out=floor_output(floor)
    if (out/'full_floor_receipt.json').exists():
        saved=json.loads((out/'full_floor_receipt.json').read_text())
        if saved.get('completed'):return saved
    prepare(floor);packet,info=load(floor)
    conditions=('power','thermal') if info['has_thermal'] else ('power',)
    methods=[]
    # Floors are the two parallel workers; each default model uses two CPU threads.
    for family in ('tree','ridge'):
        for condition in conditions:
            result=fit_method(packet,family,condition,out=out);methods.append(result)
            print(json.dumps(dict(stage='method_complete',floor=floor,family=family,condition=condition,
                seconds=result['seconds'],new_fits=result['new_fits'])),flush=True)
    scoring=score(packet,out=out,definitions=info['groups'],conditions=conditions)
    result=dict(completed=True,floor=floor,seconds=time.monotonic()-started,methods=methods,scoring=scoring,gpu=False)
    dump(out/'full_floor_receipt.json',result)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','run']);args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.stage=='prepare':
        for floor in NEW_FLOORS:prepare(floor)
        return
    started=time.monotonic();results=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(run_floor,floor) for floor in NEW_FLOORS]
        for job in as_completed(jobs):
            results.append(job.result())
            dump(OUT/'progress.json',dict(completed_floors=sorted(r['floor'] for r in results),seconds=time.monotonic()-started))
    summary=[];daily=[]
    for floor in range(1,8):
        out=floor_output(floor)
        summary.append(pd.read_json(out/'summary.json').assign(floor=floor))
        daily.append(pd.read_json(out/'daily_scores.json',convert_dates=False).assign(floor=floor))
    combined=pd.concat(summary,ignore_index=True);all_days=pd.concat(daily,ignore_index=True)
    frame_json(OUT/'summary.json',combined);frame_json(OUT/'daily_scores.json',all_days)
    result=dict(completed=True,seconds=time.monotonic()-started,workers=2,new_floors=NEW_FLOORS,
        floor_receipts=results,reused_floor2=str(PILOT.relative_to(ROOT)),
        default_tree_heads=110,multi_output_ridge_fits=11,summary_rows=len(combined),daily_rows=len(all_days),
        gpu_used=False,validation_used_for_selection=False)
    dump(OUT/'receipt.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='floor_receipts'}),flush=True)


if __name__=='__main__':main()
