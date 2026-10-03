"""Build query-local record-link inputs and support, without fitting models."""
import argparse
import gc
import json
from pathlib import Path
import time
import run_local
import numpy as np
import pandas as pd
from field_control import AcnDataset, BpaDataset
from field_control.acn_record_inputs import recover_query_inputs
from field_control.record_linkage import donor_mapping, support_statistics

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/record_linkage_v1/support'
SEEDS=(2027,2028,2029)


def progress(value):
    OUT.mkdir(parents=True,exist_ok=True)
    value=dict(at=pd.Timestamp.now().isoformat(),**value)
    with (OUT/'progress.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(value)+'\n')
    print(json.dumps(value),flush=True)


def acn_columns():
    names=[];roles=[]
    for label,channels,role in [('power',['power_kw'],'protected'),
                               ('pilot',['pilot_a'],'commands'),
                               ('numeric_state',['current_a','voltage_v','cumulative_kwh'],'state'),
                               ('operating_state',['ADAPTIVE','NOT_CHARGING','READY','UNPLUGGED','IDLE','CHARGING','OTHER'],'state')]:
        for kind in ['value','observed']:
            for minute in range(-60,0):
                for channel in channels:
                    names.append(f'{label}:{minute}:{channel}:{kind}');roles.append(role)
    names += ['hour_sin','hour_cos','weekday_sin','weekday_cos','elapsed_hours']
    roles += ['protected']*5
    return names,roles


def save_support(folder,meta,x,names,roles,min_members,context):
    folder.mkdir(parents=True,exist_ok=True)
    if x.shape!=(len(meta),len(names)) or len(names)!=len(roles):
        raise ValueError('Feature layout mismatch')
    protected=np.array([i for i,role in enumerate(roles) if role=='protected'])
    blocks={name:np.array([i for i,role in enumerate(roles) if role==name])
            for name in ['state','commands']}
    moved=np.sort(np.r_[blocks['state'],blocks['commands']])
    if len(protected)+len(moved)!=x.shape[1]:
        raise ValueError('Unclassified columns')
    np.save(folder/'features.npy',x,allow_pickle=False)
    meta.to_csv(folder/'cases.csv.gz',index=False)
    pd.DataFrame(dict(column=np.arange(len(names)),name=names,role=roles)).to_csv(folder/'columns.csv',index=False)
    frames=[]
    for seed in SEEDS:
        donor,groups=donor_mapping(meta,min_members,seed)
        statistics,groups=support_statistics(meta,x,donor,groups,blocks)
        statistics['mapping_seed']=seed
        groups['mapping_seed']=seed
        frames.append(statistics)
        np.savez_compressed(folder/f'mapping_{seed}.npz',
            case_id=meta.case_id.to_numpy(dtype=str),donor_index=donor,
            donor_case_id=meta.iloc[donor].case_id.to_numpy(dtype=str),
            protected_columns=protected,moved_columns=moved)
        groups.to_csv(folder/f'groups_{seed}.csv.gz',index=False)
    summary=pd.concat(frames,ignore_index=True)
    summary.to_csv(folder/'support.csv',index=False)
    record=dict(completed=True,cases=len(meta),feature_columns=x.shape[1],
        protected_columns=len(protected),moved_columns=len(moved),
        cached_feature_bytes=x.nbytes,min_members=min_members,mapping_seeds=list(SEEDS),
        input_support_only=True,new_model_fits=0,gpu_used=False,**context)
    (folder/'receipt.json').write_text(json.dumps(record,indent=2)+'\n')
    progress(dict(stage='input_support_complete',name=folder.name,**record))
    return summary


def acn():
    folder=OUT/'acn'
    if (folder/'receipt.json').exists():
        progress(dict(stage='reuse_input_support',source='ACN'))
        return pd.read_csv(folder/'support.csv')
    started=time.monotonic()
    data=AcnDataset(ROOT/'results/acn_fleet_months_v1')
    x,recovery=recover_query_inputs(data.population,data.meta,data._x,
        ROOT/'data/acn_static_months_v1',progress)
    pop=data.population.reset_index(drop=True)
    original=pd.Index(data.meta.case_id).get_indexer(pop.case_id)
    if not np.array_equal(original>=0,pop.target_complete.to_numpy(dtype=bool)):
        raise ValueError('Query population and labeled cases disagree')
    meta=pd.DataFrame(dict(source='ACN',site=pop.site,period=pop.split,
        query_group=pop.query_utc,member=pop.station_id,case_id=pop.case_id,
        date=pd.to_datetime(pop.query_utc,utc=True,format='mixed').dt.tz_convert('America/Los_Angeles').dt.date.astype(str),
        target_complete=pop.target_complete,original_index=original))
    names,roles=acn_columns()
    result=save_support(folder,meta,x,names,roles,2,
        dict(seconds=time.monotonic()-started,recovery=recovery,
             population='All query-known histories; future completeness is used only for support reporting'))
    del data,x;gc.collect()
    return result


def bpa():
    from bpa_standard_refresh import all_features
    from bpa_telemetry_value import feature_names,field_group
    from bpa_update_validation_common import load_split
    train,validation=load_split()
    names=feature_names()
    roles=['protected' if field_group(name)=='power' else field_group(name) for name in names]
    tables=[]
    for season,period in [('winter','train'),('winter','validation'),('spring','spring'),('summer','summer')]:
        folder=OUT/f'bpa_{period}'
        if (folder/'receipt.json').exists():
            tables.append(pd.read_csv(folder/'support.csv'))
            progress(dict(stage='reuse_input_support',source='BPA',period=period))
            continue
        started=time.monotonic()
        data=BpaDataset(ROOT/'results/benchmark_bpa_sequences_v1',season)
        indices=train if period=='train' else validation if period=='validation' else np.arange(len(data))
        raw=data.meta.iloc[indices].reset_index(drop=True)
        progress(dict(stage='assemble_bpa_inputs',period=period,cases=len(indices)))
        x=all_features(data,indices)
        meta=pd.DataFrame(dict(source='BPA',site='Field study',period=period,
            query_group=raw.cohort_id.astype(str)+'|'+raw.command_type.astype(str),
            member=raw.alias.astype(str),case_id=raw.case_id,date=raw.local_date,
            target_complete=True,original_index=indices))
        tables.append(save_support(folder,meta,x,names,roles,10,
            dict(seconds=time.monotonic()-started,season=season,
                 population='Original label-eligible devices; unchanged training/evaluation split')))
        del data,x;gc.collect()
    return pd.concat(tables,ignore_index=True)


def write_table(summary):
    display=summary[summary.mapping_seed==2027]
    lines=['# Query-local record reassignment support','',
           'Counts use the original populations and calendar periods. Changed percentages',
           'describe reassigned input rows, including the associated masks; they are not',
           'prediction improvements. Mapping seed2027 is shown; all three mappings are saved.','',
           '| Source / site | Period | Query-known / labeled cases | Reassignable / scorable groups | Group dates | State changed | Control changed |',
           '|---|---|---:|---:|---:|---:|---:|']
    for row in display.to_dict('records'):
        lines.append(f"| {row['source']} / {row['site']} | {row['period']} | "
            f"{row['query_known_cases']:,} / {row['labeled_cases']:,} | "
            f"{row['reassignable_groups']:,} / {row['scored_groups']:,} | {row['scored_dates']:,} | "
            f"{row['state_changed_fraction_among_moved']:.1%} | {row['commands_changed_fraction_among_moved']:.1%} |")
    lines += ['','ACN donors include query-known sessions with incomplete future labels.',
              'Only complete distinct-station groups enter group scoring. BPA is restricted',
              'to its original label-eligible population. Groups below the source size',
              'minimum or with duplicate members are left unchanged. Own power histories,',
              'masks, calendar and lead-aligned historical profiles remain with the device.']
    (OUT/'support.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources',nargs='+',choices=['acn','bpa'],default=['acn','bpa'])
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    for source in args.sources:
        {'acn':acn,'bpa':bpa}[source]()
    tables=[]
    for name in ['acn','bpa_train','bpa_validation','bpa_spring','bpa_summer']:
        path=OUT/name/'support.csv'
        if path.exists():tables.append(pd.read_csv(path))
    summary=pd.concat(tables,ignore_index=True)
    summary.to_csv(OUT/'summary.csv',index=False)
    write_table(summary)
    record=dict(completed=True,sources=args.sources,summary_rows=len(summary),
        seconds=time.monotonic()-started,new_model_fits=0,gpu_used=False)
    (OUT/'batch_receipt.json').write_text(json.dumps(record,indent=2)+'\n')
    progress(dict(stage='batch_complete',**record))


if __name__=='__main__':
    main()
