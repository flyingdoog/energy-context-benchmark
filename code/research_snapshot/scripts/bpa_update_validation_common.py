"""Environment-neutral input contract for temporal/cross-model replication.

Do not import run_local here: this module also runs inside .venv-neural.
"""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT/'src') not in sys.path:
    sys.path.insert(0,str(ROOT/'src'))
OUT=ROOT/'results/benchmark_bpa_sequences_v1'
DEST=OUT/'update_validation'
CONDITIONS=('initial','physical','command','full')


from field_control.bpa_preparation import split_winter


def prepare_split():
    DEST.mkdir(exist_ok=True)
    meta=pd.read_csv(OUT/'winter/cases.csv.gz',dtype={'alias':str})
    train,validation,cutoff=split_winter(meta)
    np.savez(DEST/'split.npz',train=train,validation=validation)
    origin=pd.to_datetime(meta.origin_utc,utc=True)
    info=dict(training_cases=len(train),validation_cases=len(validation),
        omitted_boundary_cases=len(meta)-len(train)-len(validation),cutoff_utc=cutoff,
        training_dates=int(meta.iloc[train].local_date.nunique()),
        validation_dates=int(meta.iloc[validation].local_date.nunique()),
        training_devices=int(meta.iloc[train].alias.nunique()),validation_devices=int(meta.iloc[validation].alias.nunique()),
        latest_training_target_end=(origin.iloc[train].max()+pd.Timedelta(minutes=180)).isoformat(),
        first_validation_origin=origin.iloc[validation].min().isoformat(),
        rule='Last14winter local dates validate. Training targets must end before the start of that block. No split chosen from prediction performance.')
    (DEST/'split.json').write_text(json.dumps(info,indent=2)+'\n')
    return info


def load_split():
    with np.load(DEST/'split.npz') as p:
        return p['train'],p['validation']


def load(season):
    folder=OUT/'model_packets'/season
    p={n:np.load(folder/f'{n}.npy',mmap_mode='r') for n in
       ['history','history_exog','static_exog','calendar','target']}
    p['feedback']=np.load(OUT/'fixed_target_update/feedback'/f'{season}.npy',mmap_mode='r')
    p['meta']=pd.read_csv(OUT/season/'cases.csv.gz',dtype={'alias':str})
    return p


def feedback_block(values,condition):
    if condition not in CONDITIONS:
        raise ValueError('Unspecified information condition')
    if condition=='initial':
        return np.zeros((len(values),690),dtype=np.float32)
    block=np.array(values,dtype=np.float32,copy=True)
    if block.shape[1:]!=(30,23):
        raise ValueError('Expected30minutes and23fields')
    if condition=='physical':block[:,:,12:]=0
    if condition=='command':block[:,:,:12]=0
    return block.reshape(len(block),690)


def static(p,indices,condition):
    return np.concatenate([np.asarray(p['static_exog'][indices]),feedback_block(p['feedback'][indices],condition)],axis=1)


def tabular(p,indices,condition):
    n=len(indices)
    return np.concatenate([p['history'][indices],p['history_exog'][indices].reshape(n,-1),
                           static(p,indices,condition),p['calendar'][indices,98:].reshape(n,-1)],axis=1)


def evaluation_parts(packets,validation):
    return [('winter_validation',packets['winter'],validation),
            ('spring',packets['spring'],np.arange(len(packets['spring']['meta']))),
            ('summer',packets['summer'],np.arange(len(packets['summer']['meta'])))]


def write_prediction(path,meta,pred):
    pred=np.asarray(pred,dtype=np.float32)
    if pred.shape!=(len(meta),10) or not np.isfinite(pred).all():
        raise ValueError('Missing/invalid prediction')
    np.savez_compressed(path,case_id=meta.case_id.to_numpy(dtype=str),absolute_bins=np.arange(2,12),prediction_kw=pred)


def blend_coefficient(initial,recent,target,weights=None):
    initial=np.asarray(initial,dtype=float);target=np.asarray(target,dtype=float)
    direction=np.asarray(recent,dtype=float)[:,None]-initial
    weight=np.ones((len(initial),1)) if weights is None else np.asarray(weights,dtype=float)[:,None]
    if weight.shape!=(len(initial),1) or not np.isfinite(weight).all() or (weight<0).any():
        raise ValueError('Invalid calibration weights')
    numerator=np.sum(weight*direction*(target-initial),axis=0)
    denominator=np.sum(weight*direction*direction,axis=0)
    return np.clip(np.divide(numerator,denominator,out=np.zeros_like(numerator),where=denominator>0),0,1)


def recent_power(p,indices):
    # Last15observedminutes, all available by the+30query; no target values.
    value=np.asarray(p['feedback'][indices,15:30,0])
    valid=np.isfinite(value);count=valid.sum(axis=1)
    return np.divide(np.where(valid,value,0).sum(axis=1),count,
                     out=np.array(p['static_exog'][indices,0],copy=True),where=count>0)


if __name__=='__main__':
    print(json.dumps(prepare_split()),flush=True)
