"""Proper activity probability scores and date-weighted reliability bins."""
import numpy as np
import pandas as pd


def activity_probability_scores(dates, target, probability, bins):
    y=np.asarray(target,dtype=float);p=np.asarray(probability,dtype=float)
    if y.ndim!=2 or y.shape!=p.shape or len(y)!=len(dates) or len(y)==0:
        raise ValueError("Aligned nonempty case-bin arrays required")
    if bins<1 or bins>y.shape[1] or not np.isfinite(y).all() or (y<0).any():
        raise ValueError("Invalid target or forecast span")
    if not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError("Probabilities must be finite and within zero and one")
    a=(y[:,:bins]>0).astype(float);p=p[:,:bins]
    dates=np.asarray(dates,dtype=str)
    frame=pd.DataFrame(dict(date=dates,brier=np.mean((p-a)**2,axis=1)))
    daily=frame.groupby("date").brier.mean().reset_index()
    _,inverse,counts=np.unique(dates,return_inverse=True,return_counts=True)
    weights=np.repeat(1/(len(counts)*counts[inverse]*bins),bins)
    flat=p.ravel();actual=a.ravel()
    groups=np.minimum((flat*10).astype(int),9)
    mass=np.bincount(groups,weights=weights,minlength=10)
    p_sum=np.bincount(groups,weights=weights*flat,minlength=10)
    a_sum=np.bincount(groups,weights=weights*actual,minlength=10)
    reliability=pd.DataFrame(dict(bin=np.arange(10),lower=np.arange(10)/10,upper=np.arange(1,11)/10,
        mass=mass,predicted_sum=p_sum,observed_sum=a_sum,
        predicted=np.divide(p_sum,mass,out=np.full(10,np.nan),where=mass>0),
        observed=np.divide(a_sum,mass,out=np.full(10,np.nan),where=mass>0),
        observations=np.bincount(groups,minlength=10)))
    return daily,reliability
