"""Fixed native N-HiTS CU-BEMS reference. Run with .venv-neural Python."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(name,'2')
import numpy as np
import pandas as pd
import torch
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import MSE
from field_control.cu_bems_neural import training_stats,normalized_inputs,THERMAL_NAMES,CALENDAR_NAMES

OUT=ROOT/'results/cu_bems_neural_reference_v1'


def write_json(path,value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,default=str,allow_nan=False)+'\n',encoding='utf8')
    temp.replace(path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--floor',type=int,choices=range(1,8),required=True)
    parser.add_argument('--condition',choices=['power','thermal'],required=True)
    parser.add_argument('--seed',type=int,choices=[1,2,3],required=True)
    parser.add_argument('--stop-after',type=int,default=1000)
    args=parser.parse_args()
    if not 1<=args.stop_after<=1000:raise ValueError('Stop is a retained segment, at most the fixed 1000 updates')
    if args.floor==1 and args.condition=='thermal':raise ValueError('Floor 1 has no thermal arm')
    torch.set_num_threads(2)
    folder=OUT/f'floor{args.floor}'/f'nhits_seed{args.seed}_{args.condition}'
    folder.mkdir(parents=True,exist_ok=True)
    receipt=folder/'receipt.json'
    if receipt.exists() and json.loads(receipt.read_text()).get('completed'):
        print(json.dumps(dict(reused_completed=True,floor=args.floor,condition=args.condition,seed=args.seed)),flush=True)
        return
    base=ROOT/'results'/('cu_bems_reference_pilot_v1' if args.floor==2 else f'cu_bems_full_reference_v1/floor{args.floor}')
    meta=pd.read_json(base/'cases.jsonl.gz',orient='records',lines=True,convert_dates=False)
    info=json.loads((base/'preparation.json').read_text())
    packet={name:np.load(base/f'{name}.npy',mmap_mode='r') for name in ['power','calendar','target']}
    packet.update(thermal=np.load(base/'thermal.npy',mmap_mode='r') if args.floor!=1 else None,
                  device_count=len(info['devices']),device_index=meta.device_index.to_numpy(dtype=np.int64))
    train=np.flatnonzero(meta.split.to_numpy()=='train')
    stats=training_stats(packet,train);write_json(folder/'preprocessing.json',stats)
    def batch(indices):
        arrays=normalized_inputs(packet,indices,stats,args.condition)
        return {name:torch.from_numpy(value) for name,value in arrays.items()},torch.from_numpy(np.array(packet['target'][indices],copy=True)[...,None])
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    history_names=list(THERMAL_NAMES) if args.floor!=1 else []
    static_names=[*CALENDAR_NAMES,*[f'device_{i}' for i in range(packet['device_count'])]]
    net=NHITS(h=10,input_size=60,hist_exog_list=history_names,stat_exog_list=static_names,
              loss=MSE(),random_seed=args.seed)
    config=net.configure_optimizers();optimizer=config['optimizer'];scheduler=config['lr_scheduler']['scheduler']
    if net.max_steps!=1000:raise ValueError('Installed N-HiTS recipe differs from the existing 1000-step baseline')
    rng=np.random.default_rng(args.seed)
    groups=[train[ix] for ix in meta.iloc[train].groupby('device_index',sort=True).indices.values()]
    def draw():
        chosen=rng.choice(len(groups),min(net.batch_size,len(groups)),replace=False)
        pool=np.concatenate([groups[i] for i in chosen])
        return rng.choice(pool,net.windows_batch_size,replace=len(pool)<net.windows_batch_size)
    checkpoint=folder/'resume.pt';start_step=0;previous_seconds=0.;losses=[]
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        net.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        rng.bit_generator.state=saved['rng'];torch.set_rng_state(saved['torch_rng'])
        start_step=saved['steps'];previous_seconds=saved['seconds'];losses=saved['losses']
    parameters=sum(p.numel() for p in net.parameters())
    recipe=dict(floor=args.floor,condition=args.condition,seed=args.seed,parameters=parameters,
        start_step=start_step,stop_after=args.stop_after,fixed_total_steps=1000,
        training_cases=len(train),cases=len(meta),devices=packet['device_count'],
        cpu_threads=2,gpu_used=False,hyperparameters=dict(net.hparams),pid=os.getpid(),
        versions={name:importlib.metadata.version(name) for name in ('torch','neuralforecast','numpy')})
    write_json(folder/'run.json',recipe)
    print(json.dumps({k:v for k,v in recipe.items() if k!='hyperparameters'}),flush=True)
    started=time.perf_counter();step_times=[];net.train()
    for step in range(start_step,args.stop_after):
        tick=time.perf_counter();inputs,y=batch(draw());optimizer.zero_grad(set_to_none=True)
        prediction=net(inputs);loss=net.loss(y=y,y_hat=prediction)
        if not torch.isfinite(loss):raise ValueError('Nonfinite loss: inspect before drawing a model conclusion')
        loss.backward();optimizer.step();scheduler.step()
        losses.append(float(loss.detach()));step_times.append(time.perf_counter()-tick)
        if (step+1)%25==0 or step+1==args.stop_after:
            seconds=previous_seconds+time.perf_counter()-started
            progress=dict(floor=args.floor,condition=args.condition,seed=args.seed,steps=step+1,
                total=1000,seconds=seconds,loss=losses[-1],pid=os.getpid())
            print(json.dumps(progress),flush=True)
            temp=folder/'resume.partial.pt'
            torch.save(dict(model=net.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),steps=step+1,seconds=seconds,losses=losses),temp)
            temp.replace(checkpoint);write_json(folder/'progress.json',progress)
    fit_seconds=previous_seconds+time.perf_counter()-started
    net.eval()
    if args.stop_after<1000:
        # Timing only, training inputs only. No partial-model evaluation result.
        indices=train[:min(len(train),512)];inputs,_=batch(indices)
        with torch.inference_mode():
            net(inputs)
            tick=time.perf_counter();net(inputs);inference_seconds=time.perf_counter()-tick
        result=dict(segment_complete=True,full_run_complete=False,retained_steps=max(start_step,args.stop_after),
            fit_seconds=fit_seconds,mean_update_seconds=float(np.mean(step_times)) if step_times else None,
            median_update_seconds=float(np.median(step_times[1:])) if len(step_times)>1 else None,
            estimated_1000_update_seconds=fit_seconds/max(start_step,args.stop_after)*1000,
            prediction_batch_cases=len(indices),prediction_batch_seconds=inference_seconds,
            estimated_full_prediction_seconds=inference_seconds*len(meta)/len(indices),
            checkpoint=str(checkpoint.relative_to(ROOT)),gpu_used=False,partial_model_test_scores=False)
        write_json(folder/'resource_measurement.json',result);print(json.dumps(result),flush=True)
        return
    torch.save(net.state_dict(),folder/'weights.pt')
    prediction=np.empty((len(meta),10),dtype=np.float32);predict_start=time.perf_counter()
    with torch.inference_mode():
        for begin in range(0,len(meta),512):
            indices=np.arange(begin,min(begin+512,len(meta)));inputs,_=batch(indices)
            prediction[indices]=np.maximum(net(inputs).squeeze(-1).numpy(),0)
    if not np.isfinite(prediction).all():raise ValueError('Nonfinite completed predictions')
    np.save(folder/'prediction.npy',prediction)
    result=dict(completed=True,floor=args.floor,condition=args.condition,seed=args.seed,steps=1000,
        parameters=parameters,fit_seconds=fit_seconds,prediction_seconds=time.perf_counter()-predict_start,
        training_cases=len(train),cases=len(meta),source_packet=str(base.relative_to(ROOT)),
        recipe='Existing native N-HiTS defaults, same information slots/case draws across paired conditions; no HPO/validation selection',
        interface='60 one-minute history samples to ten 15-minute power means; historical thermal fields/masks only',
        gpu_used=False,losses=losses)
    write_json(receipt,result);print(json.dumps({k:v for k,v in result.items() if k!='losses'}),flush=True)


if __name__=='__main__':main()
