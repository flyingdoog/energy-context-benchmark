"""Model-neutral synthetic auxiliary-report availability, with native clocks."""
import numbers
import numpy as np


def validate_blackout(source,minutes):
    limits={'BPA':60,'ACN':60,'EPFL':10,'CU-BEMS':60}
    if source not in limits:raise ValueError('Unknown operational source')
    if (not isinstance(minutes,numbers.Real) or isinstance(minutes,(bool,np.bool_)) or
            not np.isfinite(minutes) or minutes!=int(minutes) or not 0<=minutes<=limits[source]):
        raise ValueError(f'{source} auxiliary blackout must be integer minutes from 0 to {limits[source]}')
    return int(minutes)


def blackout_native_inputs(native,source,minutes):
    """Return an input view with recent auxiliary reports unknown, never shifted.

    D>0 removes auxiliary report timestamps >= query-D. Zero is the unchanged
    native view. Floating unknowns are NaN; categorical/bool placeholders must
    be read with their false observed masks. Input arrays are not modified.
    """
    minutes=validate_blackout(source,minutes)
    if minutes==0:return native
    out=dict(native)

    def hide_pair(value_key,mask_key,remove,unknown=np.nan):
        if value_key not in out:return
        values=np.array(out[value_key],copy=True)
        observed=np.array(out[mask_key],dtype=bool,copy=True)
        values[:,remove,...]=False if values.dtype.kind=='b' else unknown
        observed[:,remove,...]=False
        out[value_key]=values;out[mask_key]=observed

    if source=='BPA':
        # Common BPA inputs use 96 bins of 15 minutes. State/control is the
        # point report at each bin's final minute, not its start or mean time.
        n=native['past_power_kw'].shape[1]
        offsets=15*np.arange(-n,0)+14
        remove=offsets>=-minutes
        hide_pair('past_state','past_state_observed',remove)
        hide_pair('past_command_type','past_command_observed',remove,unknown=-1)
        if 'past_command_parameter' in out:
            out['past_command_parameter']=np.array(out['past_command_parameter'],copy=True)
            out['past_command_parameter'][:,remove]=np.nan
        if 'current_state' in out:
            out['current_state']=np.array(out['current_state'],copy=True)
            out['current_state_observed']=np.array(out['current_state_observed'],copy=True)
            # Column zero is the still-observed power snapshot.
            out['current_state'][:,1:]=np.nan
            out['current_state_observed'][:,1:]=False
        if 'current_command' in out:
            out['current_command']=np.full_like(out['current_command'],np.nan,dtype=float)
            out['current_command_observed']=np.zeros_like(out['current_command_observed'],dtype=bool)
    elif source=='ACN':
        remove=np.asarray(native['history_intervals_minutes'])[:,0]>=-minutes
        for values,mask in [('pilot_a','pilot_observed'),('numeric_state','numeric_state_observed'),
                            ('operating_state_fractions','operating_state_observed')]:
            hide_pair(values,mask,remove)
    else:
        remove=np.asarray(native['history_offsets_minutes'])>=-minutes
        for values,mask in [('numeric_state','numeric_state_observed'),('setpoint_kw','setpoint_observed'),
                            ('requested_kw','requested_observed'),('controlled','controlled_observed')]:
            hide_pair(values,mask,remove)
    return out
