"""CU-BEMS inputs for the existing native N-HiTS reference; NumPy only."""
import numpy as np

THERMAL_NAMES=('temperature','relative_humidity','temperature_observed','humidity_observed')
CALENDAR_NAMES=('hour_sin','hour_cos','weekday_sin','weekday_cos')


def training_stats(packet,train):
    sensor=packet.get('thermal')
    mean=[];scale=[]
    if sensor is not None:
        for channel in range(2):
            values=np.asarray(sensor[train,:,channel],dtype=np.float64)
            values=values[np.isfinite(values)]
            if not len(values):raise ValueError('No observed training values for a required thermal variable')
            mean.append(float(values.mean()));sd=float(values.std())
            scale.append(sd if sd>=1e-6 else 1.)
    return dict(thermal_mean=mean,thermal_scale=scale)


def normalized_inputs(packet,indices,stats,condition):
    if condition not in ('power','thermal'):raise ValueError('Unknown information condition')
    power=np.asarray(packet['power'][indices],dtype=np.float32)
    if not np.isfinite(power).all():raise ValueError('Prepared power histories must be complete')
    sensor=packet.get('thermal')
    if sensor is None:
        if condition!='power':raise ValueError('This floor has no thermal fields')
        exog=np.empty((len(indices),60,0),dtype=np.float32)
    else:
        raw=np.asarray(sensor[indices],dtype=np.float32)
        observed=np.isfinite(raw)
        scaled=(raw-np.asarray(stats['thermal_mean'],dtype=np.float32))/np.asarray(stats['thermal_scale'],dtype=np.float32)
        exog=np.concatenate([np.nan_to_num(scaled,nan=0,posinf=0,neginf=0),observed.astype(np.float32)],axis=2)
        if condition=='power':exog[:]=0
    calendar=np.asarray(packet['calendar'][indices],dtype=np.float32)
    identity=np.eye(packet['device_count'],dtype=np.float32)[packet['device_index'][indices]]
    static=np.concatenate([calendar,identity],axis=1)
    return dict(insample_y=power[...,None],insample_mask=np.ones_like(power[...,None]),
        hist_exog=exog,stat_exog=static,futr_exog=np.empty((len(indices),70,0),dtype=np.float32))
