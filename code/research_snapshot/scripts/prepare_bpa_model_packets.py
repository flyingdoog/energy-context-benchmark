"""Export a shared case interface for tabular and native-exogenous neural models."""
import gc
import json
import time
import run_local
import numpy as np
from numpy.lib.format import open_memmap
from field_control import BpaDataset
from field_control.dataset import STATE_NAMES, HISTORY_STATE_NAMES, CALENDAR_NAMES
from build_bpa_sequence_release import OUT

CODES = (0,1,2,3,5,7,8,9,13)
PACKETS = OUT/'model_packets'
HIST_NAMES = (['power_observed_fraction'] + list(HISTORY_STATE_NAMES) +
              [n+'_observed' for n in HISTORY_STATE_NAMES] +
              [f'past_command_is_{c}' for c in CODES] + ['past_command_parameter','past_command_observed'])
STATIC_NAMES = (list(STATE_NAMES) + [n+'_observed' for n in STATE_NAMES] +
                [f'current_command_is_{c}' for c in CODES] + ['current_command_parameter'] +
                [f'previous_{lag}_lead_{j}' for lag in ['day','week'] for j in range(12)] +
                [f'previous_{lag}_lead_{j}_fraction' for lag in ['day','week'] for j in range(12)])


def arrays_for(data, ix):
    p = data.inputs(ix)
    hist = np.concatenate([p['past_power_observed_fraction'][...,None], p['past_state'],
        p['past_state_observed'].astype(np.float32),
        (p['past_command_type'][...,None] == np.array(CODES)).astype(np.float32),
        p['past_command_parameter'][...,None],p['past_command_observed'][...,None]],axis=-1)
    # Every lead-aligned day/week value is historical, including the last one.
    day, df = data.power_windows(ix,1+15*np.arange(12)-1440)
    week, wf = data.power_windows(ix,1+15*np.arange(12)-10080)
    static = np.c_[p['current_state'],p['current_state_observed'],
        (p['current_command'][:,:1] == np.array(CODES)).astype(np.float32),
        p['current_command'][:,1],day,week,df,wf]
    return dict(history=p['past_power_kw'], history_exog=hist.astype(np.float32),
        static_exog=static.astype(np.float32),
        calendar=np.concatenate([p['past_calendar'],p['future_calendar']],axis=1),
        target=np.asarray(data.target[ix],dtype=np.float32))


def main():
    PACKETS.mkdir(parents=True,exist_ok=True)
    for season in ['winter','spring','summer']:
        folder=PACKETS/season;folder.mkdir(exist_ok=True)
        if (folder/'receipt.json').exists():
            print(json.dumps({'packet_reused':season}),flush=True);continue
        start=time.perf_counter();data=BpaDataset(OUT,season);outputs={}
        for begin in range(0,len(data),1024):
            ix=np.arange(begin,min(begin+1024,len(data)));batch=arrays_for(data,ix)
            if not outputs:
                for name,value in batch.items():
                    outputs[name]=open_memmap(folder/f'{name}.npy',mode='w+',dtype=np.float32,shape=(len(data),*value.shape[1:]))
            for name,value in batch.items():outputs[name][ix]=value
        for value in outputs.values():value.flush()
        receipt=dict(season=season,cases=len(data),history_bins=96,future_bins=12,
            history_exog_names=HIST_NAMES,static_exog_names=STATIC_NAMES,calendar_names=list(CALENDAR_NAMES),
            shapes={n:list(a.shape) for n,a in outputs.items()},seconds=time.perf_counter()-start,
            command_availability='Only command code/parameter at current report. Historical commands are observed; future calendar is known. No future realized commands.',
            static_interpretation='Static within a forecast case, not an immutable property of a physical device. Includes current snapshot and historical day/week profiles.')
        (folder/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps({'packet_complete':season,'cases':len(data),'seconds':receipt['seconds']}),flush=True)
        del outputs,data,batch;gc.collect()


if __name__=='__main__':main()
