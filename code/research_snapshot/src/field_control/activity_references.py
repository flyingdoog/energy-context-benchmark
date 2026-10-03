"""Training-only occurrence rules and existing-population activity scores."""
import numpy as np
from .response import score_response_predictions
from .acn_scoring import score_acn_residuals


def power_state(power,observed):
    power=np.asarray(power,dtype=float);observed=np.asarray(observed,dtype=bool)
    if power.ndim!=1 or power.shape!=observed.shape:raise ValueError('Aligned scalar observations required')
    if ((~np.isfinite(power)|(power<0))&observed).any():raise ValueError('Invalid observed power')
    state=np.full(len(power),-1,dtype=int)
    state[observed]=(power[observed]>0).astype(int)
    return state


def fit_state_frequency(power,observed,training_target):
    state=power_state(power,observed);y=np.asarray(training_target,dtype=float)
    if y.ndim!=2 or len(y)!=len(state) or not len(y) or y.shape[1]==0:
        raise ValueError('Nonempty aligned training targets required')
    if not np.isfinite(y).all() or (y<0).any():raise ValueError('Invalid training labels')
    a=(y>0).astype(float);prior=a.mean(axis=0);rates=[];counts=[]
    for s in [0,1]:
        ix=state==s;counts.append(int(ix.sum()))
        rates.append(a[ix].mean(axis=0) if ix.any() else prior)
    return dict(prevalence=prior.tolist(),rates=np.asarray(rates).tolist(),
        state_counts=counts,missing_training_state=int((state<0).sum()),training_cases=len(y))


def state_reference_predictions(power,observed,fitted):
    state=power_state(power,observed);prior=np.asarray(fitted['prevalence'])
    rates=np.asarray(fitted['rates']);known=state>=0
    prevalence=np.broadcast_to(prior,(len(state),len(prior))).copy()
    persistence=prevalence.copy();conditional=prevalence.copy()
    persistence[known]=state[known,None]
    conditional[known]=rates[state[known]]
    return {'Train prevalence':prevalence,'Current state':persistence,'State frequency':conditional}


def bpa_query_power(data,indices):
    """Exact current-power field used by the refreshed +30 full-input recipe."""
    rows=data.meta.iloc[np.atleast_1d(indices)]
    device=rows.device_index.to_numpy();minute=rows.origin_minute.to_numpy()+30
    values=np.asarray(data.grid['power_kw'][device,minute],dtype=float).copy()
    observed=np.asarray(data.grid['online'][device,minute],dtype=bool)&np.isfinite(values)
    values[~observed]=np.nan
    return values,observed


def score_activity_cohorts(source,meta,target,probability,method,period,population=None):
    """Reuse the source's membership/date rules; discard energy-specific scores.

    BPA groups contain its label-eligible devices. ACN additionally requires
    the complete query-known population and removes duplicate-station queries.
    Count means require no independence assumption. No count distribution is fit.
    """
    y=np.asarray(target,dtype=float);p=np.asarray(probability,dtype=float)
    if y.shape!=(len(meta),10) or p.shape!=y.shape:raise ValueError('Ten future bins required')
    if not np.isfinite(y).all() or (y<0).any() or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('Finite nonnegative targets and probabilities required')
    if source=='BPA':
        summary,daily=score_response_predictions(meta,(y>0).astype(float),p,method,period)
        summary=summary[summary.command_type=='all'].copy();daily=daily[daily.command_type=='all'].copy()
        for frame in [summary,daily]:frame['span_minutes']=frame.target_end_minutes-30
    elif source=='ACN':
        if population is None:raise ValueError('ACN requires the original query-known population')
        summary,daily=score_acn_residuals(meta,p-(y>0),method,population)
    else:raise ValueError('No supported cohorts for this source')
    summary=summary[summary.scale!='device'].copy();daily=daily[daily.scale!='device'].copy()
    mapping={'cohort_mean':'positive_fraction','cohort_total':'positive_count'}
    for frame in [summary,daily]:
        frame['scale']=frame.scale.map(mapping)
        frame['mse_unit']=frame.scale.map({'positive_count':'devices squared','positive_fraction':'fraction squared'})
        frame['mae_unit']=frame.scale.map({'positive_count':'devices','positive_fraction':'fraction'})
    common=['scale','span_minutes','method','mse','mae','mse_unit','mae_unit']
    return summary[common+['units','dates']],daily.rename(columns={'local_date':'date'})[common+['date']]
