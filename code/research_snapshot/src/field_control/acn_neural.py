"""Typed ACN fields for direct multi-bin neural forecasts; NumPy only."""
import numpy as np
from .acn import STATES

CONDITIONS=('power','power_commands','power_state','full')
HISTORY_NAMES=('power_observed','pilot_a','pilot_observed','current_a','voltage_v','reported_energy_kwh',
               'current_observed','voltage_observed','energy_observed',
               *('state_'+s for s in STATES),*('state_observed_'+s for s in STATES))
STATIC_NAMES=('hour_sin','hour_cos','weekday_sin','weekday_cos','connected_hours')
NUMERIC_HISTORY=(1,3,4,5)


def typed_packet(inputs,target):
    history=np.asarray(inputs['power_kw'][...,0],dtype=np.float32)
    exog=np.concatenate([inputs['power_observed'],inputs['pilot_a'],inputs['pilot_observed'],
        inputs['numeric_state'],inputs['numeric_state_observed'],
        np.nan_to_num(inputs['operating_state_fractions'],nan=0),inputs['operating_state_observed']],axis=2).astype(np.float32)
    if exog.shape!=(len(history),60,len(HISTORY_NAMES)):raise ValueError('Unexpected ACN typed history dimensions')
    return dict(history=history,history_exog=exog,static_exog=np.asarray(inputs['calendar'],dtype=np.float32),
                target=np.asarray(target,dtype=np.float32))


def condition_mask(condition):
    if condition not in CONDITIONS:raise ValueError('Unknown information condition')
    mask=np.ones(len(HISTORY_NAMES),dtype=bool)
    if condition in ('power','power_state'):mask[1:3]=False
    if condition in ('power','power_commands'):mask[3:]=False
    return mask


def training_stats(packet,train):
    mean=np.zeros(len(HISTORY_NAMES),dtype=np.float32);scale=np.ones_like(mean)
    for column in NUMERIC_HISTORY:
        values=np.asarray(packet['history_exog'][train,:,column],dtype=np.float64)
        valid=values[np.isfinite(values)]
        if len(valid):
            mean[column]=valid.mean();sd=valid.std();scale[column]=sd if sd>=1e-6 else 1
    sm=np.zeros(len(STATIC_NAMES),dtype=np.float32);ss=np.ones_like(sm)
    elapsed=np.asarray(packet['static_exog'][train,-1],dtype=np.float64)
    sm[-1]=elapsed.mean();ss[-1]=max(float(elapsed.std()),1e-6)
    return dict(history_mean=mean,history_scale=scale,static_mean=sm,static_scale=ss,
                power_mean=float(np.nanmean(packet['history'][train])))


def normalized_numpy(packet,indices,stats,condition):
    raw=np.asarray(packet['history'][indices],dtype=np.float32)
    history=np.where(np.isfinite(raw),raw,stats['power_mean'])
    exog=np.nan_to_num((np.asarray(packet['history_exog'][indices],dtype=np.float32)-stats['history_mean'])/stats['history_scale'],nan=0,posinf=0,neginf=0)
    exog[:,:,~condition_mask(condition)]=0
    static=(np.asarray(packet['static_exog'][indices],dtype=np.float32)-stats['static_mean'])/stats['static_scale']
    return dict(insample_y=history[...,None],insample_mask=np.isfinite(raw)[...,None].astype(np.float32),
                hist_exog=exog,stat_exog=static,
                futr_exog=np.empty((len(indices),70,0),dtype=np.float32))
