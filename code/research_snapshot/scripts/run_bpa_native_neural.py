"""Native NHITS/TiDE exogenous interfaces on case-aligned field-control data.

Run with .venv-neural. Architectures, optimizer, scheduler, window batch size
and training steps use installed library defaults. MSE is explicitly the task
training objective. The wrapper supplies eligible cases instead of sampling
non-command windows from a continuous-series DataFrame.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('RAY_USAGE_STATS_ENABLED','0')
import numpy as np
import pandas as pd
import torch
from neuralforecast.models import NHITS,TiDE,BiTCN
from neuralforecast.losses.pytorch import MSE
from field_control import BpaDataset,score_predictions,read_submission

OUT=ROOT/'results/benchmark_bpa_sequences_v1'
PACKETS=OUT/'model_packets'


def load(season):
    p={name:np.load(PACKETS/season/f'{name}.npy',mmap_mode='r') for name in
       ['history','history_exog','static_exog','calendar','target']}
    p['meta']=pd.read_csv(OUT/season/'cases.csv.gz',dtype={'alias':str})
    return p


def moments(array):
    width=array.shape[-1];count=np.zeros(width);total=np.zeros(width);total2=np.zeros(width)
    for start in range(0,len(array),1024):
        x=np.asarray(array[start:start+1024],dtype=np.float64).reshape(-1,width)
        good=np.isfinite(x);v=np.where(good,x,0)
        count+=good.sum(axis=0);total+=v.sum(axis=0);total2+=(v*v).sum(axis=0)
    mean=total/np.maximum(count,1)
    scale=np.sqrt(np.maximum(total2/np.maximum(count,1)-mean**2,0))
    scale[scale<1e-6]=1
    return mean.astype(np.float32),scale.astype(np.float32)


def normalize(x,stats):
    mean,scale=stats
    return np.nan_to_num((np.asarray(x,dtype=np.float32)-mean)/scale,nan=0,posinf=0,neginf=0)


def batch(packet,ix,stats):
    raw=np.asarray(packet['history'][ix],dtype=np.float32)
    history=np.where(np.isfinite(raw),raw,stats['power_mean'])
    inputs=dict(insample_y=torch.from_numpy(history[...,None]),
        insample_mask=torch.from_numpy(np.isfinite(raw)[...,None].astype(np.float32)),
        hist_exog=torch.from_numpy(normalize(packet['history_exog'][ix],stats['history'])),
        stat_exog=torch.from_numpy(normalize(packet['static_exog'][ix],stats['static'])),
        futr_exog=torch.from_numpy(np.array(packet['calendar'][ix],copy=True)))
    y=torch.from_numpy(np.array(packet['target'][ix],copy=True)[...,None])
    return inputs,y


def write_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,default=str)+'\n');temporary.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['nhits','tide','bitcn'],required=True)
    p.add_argument('--seed',type=int,default=1);p.add_argument('--profile',action='store_true')
    args=p.parse_args();torch.set_num_threads(2)
    folder=OUT/'references'/f'{args.model}_mse_seed{args.seed}';folder.mkdir(parents=True,exist_ok=True)
    result_path=folder/('profile.json' if args.profile else 'receipt.json')
    if result_path.exists():print('Completed run reused',flush=True);return
    packet=load('winter');description=json.loads((PACKETS/'winter'/'receipt.json').read_text())
    stats=dict(history=moments(packet['history_exog']),static=moments(packet['static_exog']),
               power_mean=float(np.nanmean(packet['history'])))
    write_json(folder/'preprocessing.json',dict(history=[a.tolist() for a in stats['history']],
        static=[a.tolist() for a in stats['static']],power_mean=stats['power_mean'],fit_season='winter'))
    cls={'nhits':NHITS,'tide':TiDE,'bitcn':BiTCN}[args.model]
    # Seed initialization explicitly, since native forward bypasses Trainer.fit.
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    net=cls(h=12,input_size=96,hist_exog_list=description['history_exog_names'],
        stat_exog_list=description['static_exog_names'],futr_exog_list=description['calendar_names'],
        loss=MSE(),random_seed=args.seed)
    config=net.configure_optimizers();optimizer=config['optimizer'];scheduler=config['lr_scheduler']['scheduler']
    rng=np.random.default_rng(args.seed)
    indices=[np.asarray(ix) for ix in packet['meta'].groupby('alias',sort=True).indices.values()]
    def draw():
        chosen=rng.choice(len(indices),size=min(net.batch_size,len(indices)),replace=False)
        pool=np.concatenate([indices[i] for i in chosen])
        return rng.choice(pool,size=net.windows_batch_size,replace=len(pool)<net.windows_batch_size)
    # Confirm that the installed native static-covariate branch actually receives
    # the command. This is a wiring test, not an efficacy claim.
    probe,_=batch(packet,np.arange(8),stats);net.eval()
    command_column=description['static_exog_names'].index('current_command_is_1')
    with torch.no_grad():
        a=net(probe);altered={k:v.clone() for k,v in probe.items()}
        altered['stat_exog'][:,command_column]+=1
        command_response=float((net(altered)-a).abs().max())
    if not np.isfinite(command_response) or command_response<=0:
        write_json(folder/'wiring_issue.json',dict(model=args.model,library_version=importlib.metadata.version('neuralforecast'),
            command_response=command_response,steps_trained=0,
            diagnosis='Inspect native implementation before interpreting model performance. Installed TiDE uses LayerNorm(1) in its scalar temporal decoder with default layernorm=True, making that branch input-independent.',
            next='Use a functioning native-exogenous reference now; resolve TiDE implementation separately. Do not report this failed wiring check as a weak forecasting baseline.'))
        raise ValueError('Command input has no numerical path to model output')
    checkpoint=folder/'resume.pt';start_step=0;elapsed=0;losses=[]
    if checkpoint.exists() and not args.profile:
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        net.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        rng.bit_generator.state=saved['rng'];torch.set_rng_state(saved['torch_rng'])
        start_step=saved['steps'];elapsed=saved['seconds'];losses=saved['losses']
    steps=5 if args.profile else net.max_steps
    net.train();start=time.perf_counter();step_times=[]
    print(json.dumps({'model':args.model,'seed':args.seed,'start_step':start_step,'steps':steps,
        'parameters':sum(v.numel() for v in net.parameters()),'command_wiring_response':command_response}),flush=True)
    for step in range(start_step,steps):
        tick=time.perf_counter();inputs,y=batch(packet,draw(),stats)
        optimizer.zero_grad(set_to_none=True);prediction=net(inputs);loss=net.loss(y=y,y_hat=prediction)
        if not torch.isfinite(loss):raise ValueError('Nonfinite loss; inspect inputs and scaling')
        loss.backward();optimizer.step();scheduler.step()
        step_times.append(time.perf_counter()-tick);losses.append(float(loss.detach()))
        if (step+1)%100==0 or step+1==steps:
            seconds=elapsed+time.perf_counter()-start
            progress=dict(model=args.model,seed=args.seed,steps=step+1,total=steps,seconds=seconds,loss=losses[-1],pid=os.getpid())
            print(json.dumps(progress),flush=True)
            if not args.profile:
                temporary=folder/'resume.partial.pt'
                torch.save(dict(model=net.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                    rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),steps=step+1,seconds=seconds,losses=losses),temporary)
                temporary.replace(checkpoint);write_json(folder/'progress.json',progress)
    fit_seconds=elapsed+time.perf_counter()-start
    receipt=dict(model=args.model,seed=args.seed,profile_only=args.profile,steps=steps,fit_seconds=fit_seconds,
        estimated_1000_step_seconds=float(np.median(step_times[1:])*1000) if len(step_times)>1 else None,
        parameters=sum(v.numel() for v in net.parameters()),model_hyperparameters=dict(net.hparams),
        command_wiring_response=command_response,losses=losses,
        training='Winter-only eligible command cases; up to32devices and default1024windows per update; MSE objective explicitly chosen for conditional-mean forecasting. Native architecture/optimizer/scheduler/default1000steps unchanged; no HPO.',
        preprocessing='Training-only channelwise centering/scaling for covariates, missing values set to their training means with input masks. Power stays in kW. Current snapshot/command routed through native static-within-case exogenous input.',
        versions={s:importlib.metadata.version(s) for s in ['neuralforecast','torch','numpy']})
    if args.profile:
        write_json(result_path,receipt);print(json.dumps({'profile_complete':args.model,'estimated_seconds':receipt['estimated_1000_step_seconds']}),flush=True);return
    torch.save(net.state_dict(),folder/'weights.pt');net.eval();tables=[];days=[]
    for season in ['spring','summer']:
        evaluation=load(season);forecast=np.empty_like(evaluation['target'])
        with torch.inference_mode():
            for begin in range(0,len(forecast),512):
                ix=np.arange(begin,min(begin+512,len(forecast)));inputs,_=batch(evaluation,ix,stats)
                forecast[ix]=np.maximum(net(inputs).squeeze(-1).numpy(),0)
        ds=BpaDataset(OUT,season);path=folder/f'{season}_{args.model}.npz';ds.submission(path,forecast)
        aligned=read_submission(path,ds.meta.case_id)
        table,daily=score_predictions(ds.meta,ds.target,aligned,f'{args.model}_mse_seed{args.seed}',season)
        tables.append(table);days.append(daily)
    pd.concat(tables).to_csv(folder/'scores.csv',index=False);pd.concat(days).to_csv(folder/'daily_scores.csv.gz',index=False)
    receipt['complete_with_scored_submissions']=True;write_json(result_path,receipt)
    rows=pd.concat(tables);show=rows[(rows.scale=='cohort_mean')&(rows.command_type=='all')]
    print(show[['season','method','horizon_minutes','mse','mae']].to_string(index=False),flush=True)


if __name__=='__main__':main()
