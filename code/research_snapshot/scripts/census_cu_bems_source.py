"""Full 2019 CU-BEMS support on the already selected pilot calendar; no fits."""
from __future__ import annotations

import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from census_cu_bems import parse_columns, complete_windows, longest_missing
from field_control.cu_bems_preparation import split_queries, sensor_columns, source_grid, cohorts

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/benchmark_source_feasibility_v1/cu_bems'
OUT = BASE / 'full_2019'
PILOT = ROOT / 'results/cu_bems_reference_pilot_v1'


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf8')


def support_rows(floor, split, stamps, devices, power, thermal):
    """power and thermal masks are per fixed query/device; absent sensor is None."""
    dates = np.asarray(stamps.strftime('%Y-%m-%d'))
    rows = []
    entities = [('device', x['column'], [i]) for i,x in enumerate(devices)]
    entities += [('group',name,members) for name,members in cohorts(devices)]
    for level, name, members in entities:
        mask = np.logical_and.reduce([power[i] for i in members])
        has_sensor = all(thermal[i] is not None for i in members)
        linked = mask & np.logical_and.reduce([thermal[i] for i in members]) if has_sensor else None
        rows.append(dict(floor=floor,split=split,level=level,entity=name,members=len(members),
            eligible_origins=len(stamps),power_windows=int(mask.sum()),
            power_dates=int(len(np.unique(dates[mask]))),
            thermal_fields='present' if has_sensor else 'absent_for_some_or_all_members',
            linked_windows=int(linked.sum()) if has_sensor else None,
            linked_dates=int(len(np.unique(dates[linked]))) if has_sensor else None))
    return rows


def inspect_floor(floor, splits):
    start = time.monotonic()
    path = BASE/f'2019Floor{floor}.csv'
    frame = pd.read_csv(path)
    frame,clock,quality = source_grid(frame)
    layout = parse_columns(frame.columns)
    devices = [x for x in layout if x['kind']=='AC']
    zones = sorted({x['zone'] for x in devices})
    mapping = {z:sensor_columns(layout,z) for z in zones}
    power = [np.isfinite(frame[x['column']].to_numpy(dtype=float)) for x in devices]
    thermal = {z:np.isfinite(frame[names].to_numpy(dtype=float)).all(axis=1) if names else None
               for z,names in mapping.items()}
    fields = []
    for x in layout:
        values = frame[x['column']].to_numpy(dtype=float)
        observed = np.isfinite(values)
        finite = values[observed]
        first = np.flatnonzero(observed)
        fields.append(dict(**x,observed=int(observed.sum()),missing=int((~observed).sum()),
            observed_fraction=float(observed.mean()),negative=int((finite<0).sum()),zero=int((finite==0).sum()),
            minimum=float(finite.min()) if len(finite) else None,maximum=float(finite.max()) if len(finite) else None,
            q50=float(np.quantile(finite,.5)) if len(finite) else None,q99=float(np.quantile(finite,.99)) if len(finite) else None,
            first_observed=str(clock[first[0]]) if len(first) else None,
            last_observed=str(clock[first[-1]]) if len(first) else None,
            humidity_outside_0_100=int(((finite<0)|(finite>100)).sum()) if x['unit']=='RH%' else None,
            longest_missing=longest_missing(observed,clock)))
    monthly = []
    for month in range(1,13):
        sel = clock.month==month
        for zone in zones:
            ids = [i for i,x in enumerate(devices) if x['zone']==zone]
            monthly.append(dict(floor=floor,month=f'2019-{month:02d}',zone=zone,ac_count=len(ids),
                power_fraction=float(np.mean([power[i][sel].mean() for i in ids])),
                thermal_fields='present' if mapping[zone] else 'absent',
                thermal_fraction=float(thermal[zone][sel].mean()) if mapping[zone] else None))
    support = []
    for split,begin,end in splits:
        q = split_queries(clock,begin,end)
        p = [complete_windows(v,q,60,150) for v in power]
        s = [complete_windows(thermal[x['zone']],q,60,0) if mapping[x['zone']] else None for x in devices]
        support += support_rows(floor,split,clock[q],devices,p,s)
    result = dict(floor=floor,completed=True,source=path.name,bytes=path.stat().st_size,rows=len(frame),
        measurements=len(layout),devices=devices,zones=zones,sensor_mapping=mapping,
        ac_devices=len(devices),ac_with_thermal_fields=sum(mapping[x['zone']] is not None for x in devices),
        duplicate_timestamps=quality['duplicate_timestamps'],missing_timestamp_rows=quality['missing_timestamp_rows'],
        source_quality=quality,fields=fields,monthly=monthly,support=support,
        seconds=time.monotonic()-start,reused_floor2=False,raw_modified=False,models_fitted=0)
    return result


def reuse_floor2(splits):
    start = time.monotonic()
    preparation = json.loads((PILOT/'preparation.json').read_text())
    devices = preparation['devices']
    meta = pd.read_json(PILOT/'cases.jsonl.gz',orient='records',lines=True,convert_dates=False)
    sensor = np.load(PILOT/'thermal.npy',mmap_mode='r')
    linked = np.isfinite(sensor).all(axis=(1,2))
    support = []
    for split,begin,end in splits:
        block = meta[meta.split==split].assign(linked=linked[meta.split==split])
        stamps = pd.DatetimeIndex(pd.date_range(begin,pd.Timestamp(end)-pd.Timedelta(minutes=150),freq='30min'))
        keys = stamps.strftime('%Y-%m-%d %H:%M:%S')
        power, thermal = [], []
        for i in range(len(devices)):
            subset = block[block.device_index==i]
            power.append(np.isin(keys,subset['query']))
            thermal.append(np.isin(keys,subset.loc[subset.linked,'query']))
        support += support_rows(2,split,stamps,devices,power,thermal)
    for item in preparation['support']:
        found = sum(x['power_windows'] for x in support if x['split']==item['split'] and x['level']=='device')
        if found != item['cases']:
            raise ValueError('Floor 2 support reconstruction changed completed cases')
    for item in preparation['group_support']:
        found = next(x for x in support if x['split']==item['split'] and x['level']=='group' and x['entity']==item['group'])
        if found['power_windows'] != item['windows'] or found['power_dates'] != item['dates']:
            raise ValueError('Floor 2 fixed group differs from completed pilot')
    old = json.loads((BASE/'census.json').read_text())
    return dict(floor=2,completed=True,source='2019Floor2.csv',bytes=old['bytes'],rows=old['rows'],
        measurements=old['measurements'],devices=devices,zones=old['zones_with_ac'],
        sensor_mapping={z:[f'z{z}_S1(degC)',f'z{z}_S1(RH%)'] for z in old['zones_with_ac']},
        ac_devices=len(devices),ac_with_thermal_fields=len(devices),duplicate_timestamps=0,missing_timestamp_rows=0,
        fields=json.loads((BASE/'field_summary.json').read_text()),
        monthly=json.loads((BASE/'monthly_coverage.json').read_text()),support=support,
        seconds=time.monotonic()-start,reused_floor2=True,raw_modified=False,models_fitted=0)


def main():
    started = time.monotonic()
    splits = json.loads((PILOT/'preparation.json').read_text())['splits']
    OUT.mkdir(parents=True,exist_ok=True)
    records = []
    for floor in range(1,8):
        path = OUT/f'floor{floor}.json'
        if path.exists() and json.loads(path.read_text()).get('completed'):
            result = json.loads(path.read_text())
        else:
            result = reuse_floor2(splits) if floor==2 else inspect_floor(floor,splits)
            dump(path,result)
        records.append(result)
        print(json.dumps(dict(stage='floor_complete',floor=floor,devices=result['ac_devices'],
            sensor_devices=result['ac_with_thermal_fields'],seconds=result['seconds'],reused_floor2=result['reused_floor2'])),flush=True)
    summaries=[]
    for r in records:
        for split,begin,end in splits:
            rows=[s for s in r['support'] if s['split']==split]
            devices=[s for s in rows if s['level']=='device']
            group=next(s for s in rows if s['level']=='group' and s['entity']=='floor')
            summaries.append(dict(floor=r['floor'],split=split,devices=r['ac_devices'],
                thermal_devices=r['ac_with_thermal_fields'],device_cases=sum(s['power_windows'] for s in devices),
                device_linked_cases=sum(s['linked_windows'] or 0 for s in devices) if r['ac_with_thermal_fields'] else None,
                devices_with_cases=sum(s['power_windows']>0 for s in devices),
                floor_windows=group['power_windows'],floor_dates=group['power_dates'],floor_linked_windows=group['linked_windows'],
                floor_linked_dates=group['linked_dates']))
    dump(OUT/'support.json',summaries)
    totals=[dict(split=split,device_cases=sum(s['device_cases'] for s in summaries if s['split']==split),
        floor_windows=sum(s['floor_windows'] for s in summaries if s['split']==split)) for split,_,_ in splits]
    result=dict(completed=True,seconds=time.monotonic()-started,source_files=7,new_csv_censuses=6,
        reused_floor2=True,source_buildings=1,ac_devices=sum(r['ac_devices'] for r in records),
        ac_with_thermal_fields=sum(r['ac_with_thermal_fields'] for r in records),splits=splits,
        totals=totals,negative_power_observations=sum(f['negative'] for r in records for f in r['fields'] if f['kind']=='AC'),
        models_fitted=0,gpu_used=False,raw_modified=False,
        clock='Source naive minute timestamps; no UTC or receipt-time assumption',
        history_minutes=60,future_minutes=150,query_stride_minutes=30,spans_minutes=[30,90,150])
    dump(OUT/'receipt.json',result)
    lines=['# CU-BEMS 2019: complete source support','',
        'Counts retain every original AC and complete 150-minute targets. Thermal completeness is descriptive, not an eligibility filter.',
        'One building. Floor 1 has no thermal fields. No model is fitted by this census.','',
        '| Floor | ACs / with thermal fields | Split | Device cases | Complete floor windows | Floor dates | Complete thermal floor windows |',
        '|---|---:|---|---:|---:|---:|---:|']
    for s in summaries:
        thermal='n.a.' if s['floor_linked_windows'] is None else f"{s['floor_linked_windows']:,}"
        lines.append(f"| {s['floor']} | {s['devices']} / {s['thermal_devices']} | {s['split']} | {s['device_cases']:,} | {s['floor_windows']:,} | {s['floor_dates']} | {thermal} |")
    lines += ['', 'Floor 6 ends on October 22. Its missing calendar tail remains unobserved, and later_2 support is zero. The single malformed trailing line is recorded separately, not interpreted as a measurement.',
        'A query uses [q-60,q) for history and [q,q+150) for targets, every 30 minutes. The whole target stays inside its calendar split.',
        'Train: Apr-Jun; validation: Jul-Aug; later_1: Sep-Oct; later_2: Nov-Dec 2019. Native spans are 30/90/150 minutes.',
        'Zone groups keep original membership; a zone identical to its floor is not duplicated. Per-device and zone details are in floor1.json through floor7.json.',
        'Floor 2 uses the completed pilot cases and thermal arrays, not a repeat raw census or training run.',
        'Source: [original CU-BEMS Figshare release](https://doi.org/10.6084/m9.figshare.11726517), recorded version 6, CC BY 4.0.','']
    (OUT/'support.md').write_text('\n'.join(lines),encoding='utf8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
