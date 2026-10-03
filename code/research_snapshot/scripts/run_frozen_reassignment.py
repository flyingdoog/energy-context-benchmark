"""Evaluate saved full-input Trees under the existing query-local mappings."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import gc
import json
import os
from pathlib import Path
import time
import run_local
import numpy as np
import pandas as pd
from lightgbm import Booster
from run_record_linkage import load_inputs,normalize
from field_control.record_linkage import reassign_blocks
from field_control.acn_scoring import score_acn_residuals
from field_control.response import score_response_predictions,read_response_submission

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/frozen_reassignment_v1'


def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')


def note(folder,**value):
    value=dict(at=time.strftime('%Y-%m-%dT%H:%M:%S'),pid=os.getpid(),**value)
    with (folder/'progress.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(value)+'\n')
    print(json.dumps(value),flush=True)


def predict(models,x,donor,moved,ix,ids,path,folder):
    if path.exists():
        with np.load(path) as p:
            if not np.array_equal(p['case_id'].astype(str),np.asarray(ids,dtype=str)):raise ValueError('Prediction support differs')
            return p['prediction_kw']
    value=np.empty((len(ix),10),dtype=np.float32)
    for start in range(0,len(ix),4096):
        stop=min(start+4096,len(ix));z=reassign_blocks(x,donor,moved,ix[start:stop])
        value[start:stop]=np.column_stack([np.maximum(0,m.predict(z,num_threads=2)) for m in models])
        note(folder,stage='predict',packet=path.name,done=stop,total=len(ix))
    np.savez_compressed(path,case_id=np.asarray(ids,dtype=str),prediction_kw=value)
    return value


def run_site(site):
    folder=OUT/site;folder.mkdir(parents=True,exist_ok=True)
    rp=folder/'receipt.json'
    if rp.exists() and json.loads(rp.read_text()).get('completed'):return json.loads(rp.read_text())
    started=time.perf_counter();is_bpa=site=='bpa'
    base=ROOT/'results'/('benchmark_bpa_sequences_v1' if is_bpa else 'acn_fleet_months_v1')
    model_folder=base/'standard_refresh_v1' if is_bpa else base/'references'/site/'full'
    models=[Booster(model_file=str(model_folder/f'lead_{j:02d}.txt')) for j in range(10)]
    if not is_bpa:
        original_meta=pd.read_csv(base/'cases.csv.gz');population=pd.read_csv(base/'query_population.csv.gz')
        with np.load(base/'packet.npz') as p:targets=p['y']
        with np.load(base/'references/full.npz') as p:
            original_ids=pd.Index(p['case_id'].astype(str));original_prediction=p['prediction_kw']
    tables=[];days=[];support=[]
    periods=['validation','spring','summer'] if is_bpa else ['validation','autumn','winter']
    for period in periods:
        for seed in (2027,2028,2029):
            meta,x,donor,moved=load_inputs('bpa_'+period if is_bpa else 'acn',seed)
            if is_bpa:
                ix=np.arange(len(meta));season='winter' if period=='validation' else period
                indices=meta.original_index.to_numpy(dtype=int)
                native=pd.read_csv(base/season/'cases.csv.gz',dtype={'alias':str}).iloc[indices].reset_index(drop=True)
                target=np.load(base/season/'target_kw.npy',mmap_mode='r');y=np.asarray(target[indices,2:])
                saved='winter_validation' if period=='validation' else period
                original=read_response_submission(base/'standard_refresh_v1'/f'{saved}.npz',native.case_id)
            else:
                ix=np.flatnonzero((meta.site==site)&(meta.period==period)&(meta.original_index>=0))
                indices=meta.original_index.iloc[ix].to_numpy(dtype=int)
                native=original_meta.iloc[indices].reset_index(drop=True);y=targets[indices]
                order=original_ids.get_indexer(native.case_id)
                if (order<0).any():raise ValueError('Original model prediction missing')
                original=original_prediction[order]
            if seed==2027:
                probe=np.linspace(0,len(ix)-1,min(64,len(ix)),dtype=int)
                replay=np.column_stack([np.maximum(0,m.predict(x[ix[probe]],num_threads=2)) for m in models])
                np.testing.assert_allclose(replay,original[probe],rtol=1e-5,atol=1e-6)
                support.append(dict(period=period,cases=len(ix),unchanged_replay_max_abs=float(abs(replay-original[probe]).max())))
            value=predict(models,x,donor,moved,ix,native.case_id,folder/f'{period}_mapping{seed}.npz',folder)
            if is_bpa:
                table,daily=score_response_predictions(native,y,value,'Frozen reassigned',saved)
                table=normalize(table,'BPA','Field study',period);daily=normalize(daily,'BPA','Field study',period)
            else:
                table,daily=score_acn_residuals(native,value-y,'Frozen reassigned',population)
                table=normalize(table,'ACN',site,period);daily=normalize(daily,'ACN',site,period)
            tables.append(table.assign(mapping_seed=seed));days.append(daily.assign(mapping_seed=seed))
            note(folder,stage='scored',site=site,period=period,mapping_seed=seed,seconds=time.perf_counter()-started)
            del x,value,original,y;gc.collect()
    pd.concat(tables,ignore_index=True).to_json(folder/'summary.json',orient='records')
    pd.concat(days,ignore_index=True).to_json(folder/'daily.json.gz',orient='records',compression='gzip')
    result=dict(completed=True,site=site,seconds=time.perf_counter()-started,packets=9,support=support,new_fits=0,gpu_used=False)
    save(rp,result);note(folder,stage='site_complete',**result)
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True);started=time.perf_counter();receipts=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        tasks=[pool.submit(run_site,site) for site in ['caltech','office001','bpa']]
        for future in as_completed(tasks):receipts.append(future.result());save(OUT/'progress.json',receipts)
    result=dict(completed=True,seconds=time.perf_counter()-started,sites=receipts,new_fits=0,gpu_used=False)
    save(OUT/'receipt.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
