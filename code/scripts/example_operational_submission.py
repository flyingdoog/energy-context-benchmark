"""Model-library-free example for the shared power and occurrence interface."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from field_control import OperationalForecastTask
from field_control.activity_references import fit_state_frequency,state_reference_predictions


def main(argv=None):
    started = time.monotonic()
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',choices=['BPA','ACN','EPFL','CU-BEMS'],required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--season');parser.add_argument('--site');parser.add_argument('--split',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--submission',type=Path,help='Read an existing submission instead of making the example forecasts')
    args=parser.parse_args(argv);task=OperationalForecastTask(args.data,args.source,season=args.season,site=args.site)
    args.output.mkdir(parents=True,exist_ok=True);ix=task.indices(args.split)
    if args.submission is None:
        training=OperationalForecastTask(args.data,'BPA',season='winter') if args.source=='BPA' else task
        train=training.indices('train');past,observed=training.current_power(train);y=training.targets(train)
        fit=fit_state_frequency(past,observed,y)
        now,known=task.current_power(ix);prob=state_reference_predictions(now,known,fit)['State frequency']
        power=np.repeat(now[:,None],task.width,axis=1);power[~known]=y.mean(axis=0)
        path=args.output/'example_submission.npz'
        task.submission(path,split=args.split,prediction_kw=power,positive_probability=prob)
    else:path=args.submission
    values=task.read_submission(path,split=args.split);scores=task.score(split=args.split,**values,method='example' if args.submission is None else 'submission')
    for track,(summary,daily) in scores.items():
        summary.to_json(args.output/f'{track}_scores.json',orient='records',indent=2)
    receipt=dict(completed=True,source=args.source,split=args.split,cases=len(ix),tracks=list(scores),
        fitted_training_frequency=args.submission is None,new_ml_fits=0,gpu_used=False,
        seconds=time.monotonic()-started,
        model_libraries_loaded=sorted(name for name in ['torch','lightgbm','sklearn'] if name in sys.modules))
    (args.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)
    close = getattr(task.base, 'close', None)
    if close is not None:
        close()


if __name__=='__main__':main()
