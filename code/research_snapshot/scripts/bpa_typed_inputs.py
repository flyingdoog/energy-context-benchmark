"""Typed exogenous fields; categorical codes are never physical magnitudes.

The BPA report's operating-state table (printed p156, PDF p171) defines
0..12 as states,13..125 unused and126..255 manufacturer-reserved. Preserve
reserved observations as a category; do not relabel255 as missing without a
vendor definition. Original raw arrays, cases and target values remain intact.
"""
from functools import lru_cache
import json
import numpy as np
from bpa_update_validation_common import OUT,CONDITIONS

PHYSICAL_WIDTH=30
FEEDBACK_WIDTH=41


def category_names(name):
    if name=='oper_state':return [f'oper_state_is_{i}' for i in range(13)]+['oper_state_manufacturer_reserved','oper_state_other']
    if name in ('running','opt_out'):return [f'{name}_is_0',f'{name}_is_1',f'{name}_other']
    return [name]


def encode(values,names):
    blocks=[]
    for j,name in enumerate(names):
        x=np.asarray(values[...,j],dtype=np.float32)
        if name=='oper_state':
            normal=(x[...,None]==np.arange(13));reserved=np.isfinite(x)&(x>=126)&(x<=255)
            other=np.isfinite(x)&~normal.any(axis=-1)&~reserved
            blocks.extend([normal,reserved[...,None],other[...,None]])
        elif name in ('running','opt_out'):
            blocks.append(np.stack([x==0,x==1,np.isfinite(x)&(x!=0)&(x!=1)],axis=-1))
        else:blocks.append(x[...,None])
    return np.concatenate(blocks,axis=-1).astype(np.float32)


def expanded(names):return [new for name in names for new in category_names(name)]


@lru_cache(maxsize=1)
def description():
    raw=json.loads((OUT/'model_packets/winter/receipt.json').read_text())
    early=(['power_kw','power_observed','present','capacity','oper_state','running','opt_out']+
           [n+'_observed' for n in ['present','capacity','oper_state','running','opt_out']]+
           [f'command_is_{i}' for i in [0,1,2,3,5,7,8,9,13]]+['command_numeric','command_observed'])
    initial=expanded(raw['static_exog_names']);feedback=expanded(early)
    return dict(raw=raw,early=early,initial_static=initial,
        history_exog_names=expanded(raw['history_exog_names']),
        static_exog_names=initial+[f'minute{m}:{n}' for m in range(1,31) for n in feedback],
        calendar_names=raw['calendar_names'])


def history_exog(p,ix):return encode(p['history_exog'][ix],description()['raw']['history_exog_names'])


def feedback_values(values,condition):
    if condition not in CONDITIONS:raise ValueError('Unknown information condition')
    if condition=='initial':return np.zeros((len(values),30*FEEDBACK_WIDTH),dtype=np.float32)
    block=encode(values,description()['early'])
    if condition=='physical':block[:,:,PHYSICAL_WIDTH:]=0
    if condition=='command':block[:,:,:PHYSICAL_WIDTH]=0
    return block.reshape(len(values),-1)


def static(p,ix,condition):
    first=encode(p['static_exog'][ix],description()['raw']['static_exog_names'])
    return np.concatenate([first,feedback_values(p['feedback'][ix],condition)],axis=1)


def tabular(p,ix,condition):
    n=len(ix)
    return np.concatenate([p['history'][ix],history_exog(p,ix).reshape(n,-1),
                           static(p,ix,condition),p['calendar'][ix,98:].reshape(n,-1)],axis=1)


def numeric_fields(names):
    numeric={'power_kw','present','capacity','current_command_parameter','past_command_parameter','command_numeric'}
    return np.array([n.split(':')[-1] in numeric or
        (n.startswith('previous_') and not n.endswith('_fraction')) for n in names])


def numeric_only_stats(stats,names):
    mean,scale=[np.array(v,copy=True) for v in stats]
    mask=numeric_fields(names);mean[~mask]=0;scale[~mask]=1
    return mean,scale
