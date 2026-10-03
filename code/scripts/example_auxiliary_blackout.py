"""Inspect auxiliary availability through the common API; no model or labels."""
import argparse
import json
from pathlib import Path
import sys
import time
import run_local
import numpy as np
from field_control import OperationalForecastTask


def observed_counts(native,source):
    keys={'BPA':['past_state_observed','past_command_observed','current_command_observed'],
          'ACN':['numeric_state_observed','operating_state_observed','pilot_observed'],
          'EPFL':['numeric_state_observed','setpoint_observed','requested_observed','controlled_observed']}[source]
    counts={key:int(np.count_nonzero(native[key])) for key in keys if key in native}
    if source=='BPA' and 'current_state_observed' in native:
        counts['current_auxiliary_state_observed']=int(np.count_nonzero(native['current_state_observed'][:,1:]))
    return counts


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',choices=['BPA','ACN','EPFL'],required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--season');parser.add_argument('--site')
    parser.add_argument('--split',required=True)
    parser.add_argument('--condition',choices=['power','power_state','full'],default='full')
    parser.add_argument('--blackout-minutes',type=int,default=5)
    parser.add_argument('--sample-cases',type=int,default=16)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.sample_cases<1:parser.error('--sample-cases must be positive')
    started=time.monotonic()
    task=OperationalForecastTask(args.data,args.source,season=args.season,site=args.site)
    indices=task.indices(args.split)
    indices=indices[np.linspace(0,len(indices)-1,min(args.sample_cases,len(indices)),dtype=int)]
    timely=task.inputs(indices,args.condition)
    limited=task.inputs(indices,args.condition,auxiliary_blackout_minutes=args.blackout_minutes)
    before=observed_counts(timely['native_inputs'],args.source)
    after=observed_counts(limited['native_inputs'],args.source)
    receipt=dict(completed=True,source=args.source,site=task.site,split=args.split,
        condition=args.condition,sampled_cases=len(indices),blackout_minutes=args.blackout_minutes,
        observed_auxiliary_values_before=before,observed_auxiliary_values_after=after,
        withheld_auxiliary_values={key:before[key]-after[key] for key in before},
        availability=limited.get('availability',{'auxiliary_blackout_minutes':0,'applied':False}),
        target_values_used=False,predictions_made=False,new_ml_fits=0,gpu_used=False,
        model_libraries_loaded=sorted(n for n in ['torch','lightgbm','sklearn'] if n in sys.modules),
        seconds=round(time.monotonic()-started,3))
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
