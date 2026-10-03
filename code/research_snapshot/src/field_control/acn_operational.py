"""Input-only charging-demand rules and operational descriptors.

These rules predict recorded power; declared requests are not physical bounds
on realized charging. No fitted parameter or realized session endpoint is used.
"""
import numpy as np


def last_observed(values):
    values=np.asarray(values,dtype=float)
    if values.ndim!=2:raise ValueError('Expected cases by history')
    finite=np.isfinite(values)
    index=values.shape[1]-1-finite[:,::-1].argmax(axis=1)
    return np.where(finite.any(axis=1),values[np.arange(len(values)),index],np.nan)


def bounded_rate_forecast(rate,budget=None,stop_hours=None):
    """Integrate a constant rate exactly over ten 15-minute forecast bins."""
    rate=np.asarray(rate,dtype=float)
    if not np.isfinite(rate).all() or (rate<0).any():raise ValueError('Invalid power rate')
    end=np.arange(1,11,dtype=float)*.25
    cumulative=rate[:,None]*end[None,:]
    if stop_hours is not None:
        stop=np.asarray(stop_hours,dtype=float)
        if np.isnan(stop).any() or (stop<0).any():raise ValueError('Invalid stop time')
        cumulative=rate[:,None]*np.minimum(end[None,:],stop[:,None])
    if budget is not None:
        budget=np.asarray(budget,dtype=float)
        if np.isnan(budget).any() or (budget<0).any():raise ValueError('Invalid energy budget')
        cumulative=np.minimum(cumulative,budget[:,None])
    return np.diff(np.c_[np.zeros(len(rate)),cumulative],axis=1)/.25


def operational_inputs(history):
    power=last_observed(history['power_kw'][:,:,0])
    if not np.isfinite(power).all():raise ValueError('No observed history power')
    power=np.maximum(power,0)
    delivered=last_observed(history['numeric_state'][:,:,2])
    voltage=last_observed(history['numeric_state'][:,:,1])
    pilot=last_observed(history['pilot_a'][:,:,0])
    capacity=np.where((voltage>0)&(pilot>=0),voltage*pilot/1000,np.nan)
    request=history['declarations']['requested_kwh']
    hours=history['declarations']['requested_departure_hours']
    gap_known=np.isfinite(request)&np.isfinite(delivered)&(delivered>=0)
    gap=np.where(gap_known,np.maximum(request-delivered,0),np.nan)
    status=np.full(len(power),'missing_request',dtype='<U32')
    status[np.isfinite(request)]='missing_delivered'
    status[gap_known]='missing_departure'
    status[gap_known&np.isfinite(hours)&(hours>2.5)]='departure_beyond_target'
    status[gap_known&np.isfinite(hours)&(hours>0)&(hours<=2.5)]='departure_within_target'
    status[gap_known&np.isfinite(hours)&(hours<=0)]='declared_departure_passed'
    status[gap_known&(gap==0)]='request_already_delivered'
    return dict(power_kw=power,delivered_kwh=delivered,pilot_capacity_kw=capacity,
        requested_kwh=request,departure_hours=hours,outstanding_kwh=gap,status=status)


def operational_rules(history):
    state=operational_inputs(history)
    p=state['power_kw'];gap=state['outstanding_kwh'];hours=state['departure_hours']
    capacity=state['pilot_capacity_kw']
    budget=np.where(np.isfinite(gap),gap,np.inf)
    spread=p.copy();stop=np.full(len(p),np.inf)
    eligible=np.isfinite(gap)&np.isfinite(hours)&(hours>0)&np.isfinite(capacity)
    spread[eligible]=np.minimum(gap[eligible]/hours[eligible],capacity[eligible])
    stop[eligible]=hours[eligible]
    return dict(current_power=bounded_rate_forecast(p),
        requested_budget=bounded_rate_forecast(p,budget),
        requested_deadline=bounded_rate_forecast(spread,budget,stop))
