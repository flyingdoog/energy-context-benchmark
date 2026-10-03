"""Input-only availability diagnostics; no labels or prediction metrics."""
import numpy as np


def last_available(values, observed, offsets, blackout_minutes=0):
    """Latest observed value per channel before an imposed query-time blackout.

    For D>0, reports with relative timestamp >= -D are withheld. D=0 retains
    all original query-available reports, including a source's t=0 snapshot.
    This is a synthetic availability cutoff, not a measured delivery delay.
    """
    values=np.asarray(values)
    observed=np.asarray(observed,dtype=bool)
    offsets=np.asarray(offsets,dtype=float)
    if (values.ndim!=3 or observed.shape!=values.shape or
            offsets.shape!=(values.shape[1],) or not len(offsets)):
        raise ValueError('Expected matching case/time/channel inputs and offsets')
    if (not np.isfinite(offsets).all() or np.any(np.diff(offsets)<=0) or
            np.any(offsets>0) or not np.isfinite(blackout_minutes) or blackout_minutes<0):
        raise ValueError('Require ordered past offsets and a nonnegative blackout')
    good=observed & np.isfinite(values)
    if blackout_minutes>0:
        good=good & (offsets[None,:,None] < -blackout_minutes)
    locations=np.max(np.where(good,np.arange(len(offsets))[None,:,None],-1),axis=1)
    available=locations>=0
    safe=np.maximum(locations,0)
    selected=np.take_along_axis(values,safe[:,None,:],axis=1)[:,0,:].astype(float)
    selected[~available]=np.nan
    age=np.where(available,-offsets[safe],np.nan)
    return selected,available,age


def compare_availability(values, observed, offsets, blackout_minutes):
    current,known,_=last_available(values,observed,offsets)
    past,retained,age=last_available(values,observed,offsets,blackout_minutes)
    paired=known & retained
    changed=paired & ~np.isclose(current,past,rtol=1e-5,atol=1e-6,equal_nan=True)
    comparable=paired.any(axis=1)
    age_values=age[retained]
    return dict(cases=len(values),channels=values.shape[-1],
        blackout_minutes=int(blackout_minutes),
        original_any_available_cases=int(known.any(axis=1).sum()),
        retained_any_available_cases=int(retained.any(axis=1).sum()),
        retained_all_channels_cases=int(retained.all(axis=1).sum()),
        comparable_cases=int(comparable.sum()),
        changed_cases=int(changed.any(axis=1).sum()),
        changed_fraction_among_comparable=(float(changed.any(axis=1).sum()/comparable.sum())
                                            if comparable.any() else None),
        lost_available_values=int((known & ~retained).sum()),
        median_last_report_age_minutes=float(np.median(age_values)) if len(age_values) else None,
        p90_last_report_age_minutes=float(np.quantile(age_values,.9)) if len(age_values) else None)


def mask_acn_auxiliary(x,blackout_minutes):
    """Native zero-filled values plus explicit observed masks; power unchanged."""
    out=np.array(x,copy=True)
    if out.ndim!=2 or out.shape[1]!=1445 or not 0<=blackout_minutes<=60:
        raise ValueError('Invalid ACN native layout or cutoff')
    if blackout_minutes==0:return out
    remove=np.arange(-60,0)>=-blackout_minutes
    for start,stop,channels in [(120,180,1),(180,240,1),(240,420,3),
                                (420,600,3),(600,1020,7),(1020,1440,7)]:
        out[:,start:stop].reshape(len(out),60,channels)[:,remove,:]=0
    return out


def mask_epfl_auxiliary(x,blackout_minutes):
    """Remove state/operational fields only; preserve actual power and calendar."""
    out=np.array(x,copy=True)
    if out.ndim!=2 or out.shape[1]!=94 or not 0<=blackout_minutes<=10:
        raise ValueError('Invalid EPFL native layout or cutoff')
    if blackout_minutes==0:return out
    remove=np.arange(-10,0)>=-blackout_minutes
    for start,stop,channels in [(14,64,5),(64,94,3)]:
        out[:,start:stop].reshape(len(out),10,channels)[:,remove,:]=np.nan
    return out
