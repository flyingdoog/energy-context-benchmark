"""Native N-HiTS telemetry-value comparison; run only in .venv-neural."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import time
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import numpy as np
import pandas as pd
import torch
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import MSE
from run_bpa_native_neural import moments,normalize,write_json
from bpa_update_validation_common import OUT,load_split,write_prediction
from bpa_typed_inputs import numeric_only_stats

CONDITIONS=('power','power_commands','power_state','full')
PACKETS=OUT/'refreshed_model_packets';DEST=OUT/'telemetry_value_v1'


def load(season):
    folder=PACKETS/season
    p={n:np.load(folder/f'{n}.npy',mmap_mode='r') for n in ['history','history_exog','static_exog','calendar','target']}
    p['meta']=pd.read_csv(OUT/season/'cases.csv.gz',dtype={'alias':str})
    return p


def batch(p,ix,stats,masks):
    raw=np.asarray(p['history'][ix],dtype=np.float32)
    hist=normalize(p['history_exog'][ix],stats['history']);hist[:,:,~masks['history']]=0
    static=normalize(p['static_exog'][ix],stats['static']);static[:,~masks['static']]=0
    inputs=dict(insample_y=torch.from_numpy(np.where(np.isfinite(raw),raw,stats['power_mean'])[...,None]),
        insample_mask=torch.from_numpy(np.isfinite(raw)[...,None].astype(np.float32)),
        hist_exog=torch.from_numpy(hist),stat_exog=torch.from_numpy(static),
        futr_exog=torch.from_numpy(np.array(p['calendar'][ix],copy=True)))
    return inputs,torch.from_numpy(np.array(p['target'][ix],copy=True)[...,None])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--condition',choices=CONDITIONS,required=True)
    parser.add_argument('--seed',type=int,default=1);parser.add_argument('--profile',action='store_true');args=parser.parse_args()
    torch.set_num_threads(2);folder=DEST/f'nhits_seed{args.seed}_{args.condition}';folder.mkdir(exist_ok=True)
    receipt_path=folder/('profile.json' if args.profile else 'receipt.json')
    if receipt_path.exists() and json.loads(receipt_path.read_text()).get('completed'):
        print('Completed run reused',flush=True);return
    desc=json.loads((PACKETS/'winter/receipt.json').read_text());p=load('winter');train,validation=load_split()
    stats=dict(history=numeric_only_stats(moments(p['history_exog'][train]),desc['history_exog_names']),
        static=numeric_only_stats(moments(p['static_exog'][train]),desc['static_exog_names']),
        power_mean=float(np.nanmean(p['history'][train])))
    masks={n:np.array(v,dtype=bool) for n,v in desc['condition_masks'][args.condition].items()}
    write_json(folder/'preprocessing.json',dict(history=[a.tolist() for a in stats['history']],static=[a.tolist() for a in stats['static']],
        power_mean=stats['power_mean'],training_cases=len(train),condition=args.condition,
        masking='After training-only normalization, inactive exogenous columns are zero; categorical indicators remain unscaled.'))
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    net=NHITS(h=10,input_size=96,hist_exog_list=desc['history_exog_names'],stat_exog_list=desc['static_exog_names'],
        futr_exog_list=desc['calendar_names'],loss=MSE(),random_seed=args.seed)
    config=net.configure_optimizers();optimizer=config['optimizer'];scheduler=config['lr_scheduler']['scheduler']
    rng=np.random.default_rng(args.seed);groups=[train[ix] for ix in p['meta'].iloc[train].groupby('alias',sort=True).indices.values()]
    def draw():
        chosen=rng.choice(len(groups),min(net.batch_size,len(groups)),replace=False)
        pool=np.concatenate([groups[i] for i in chosen]);return rng.choice(pool,net.windows_batch_size,replace=len(pool)<net.windows_batch_size)
    checkpoint=folder/'resume.pt';start_step=0;elapsed=0.;losses=[]
    if checkpoint.exists() and not args.profile:
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        net.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        rng.bit_generator.state=saved['rng'];torch.set_rng_state(saved['torch_rng']);start_step=saved['steps'];elapsed=saved['seconds'];losses=saved['losses']
    steps=5 if args.profile else net.max_steps;parameters=sum(a.numel() for a in net.parameters());step_times=[]
    print(json.dumps(dict(condition=args.condition,seed=args.seed,steps=steps,start_step=start_step,parameters=parameters,pid=os.getpid())),flush=True)
    net.train();start=time.perf_counter()
    for step in range(start_step,steps):
        tick=time.perf_counter();inputs,y=batch(p,draw(),stats,masks)
        optimizer.zero_grad(set_to_none=True);prediction=net(inputs);loss=net.loss(y=y,y_hat=prediction)
        if not torch.isfinite(loss):raise ValueError('Nonfinite loss; inspect inputs before interpretation')
        loss.backward();optimizer.step();scheduler.step();losses.append(float(loss.detach()));step_times.append(time.perf_counter()-tick)
        if (step+1)%100==0 or step+1==steps:
            seconds=elapsed+time.perf_counter()-start
            progress=dict(condition=args.condition,seed=args.seed,steps=step+1,total=steps,seconds=seconds,loss=losses[-1],pid=os.getpid())
            print(json.dumps(progress),flush=True)
            if not args.profile:
                tmp=folder/'resume.partial.pt'
                torch.save(dict(model=net.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),rng=rng.bit_generator.state,
                    torch_rng=torch.get_rng_state(),steps=step+1,seconds=seconds,losses=losses),tmp)
                tmp.replace(checkpoint);write_json(folder/'progress.json',progress)
    fit_seconds=elapsed+time.perf_counter()-start
    if args.profile:
        write_json(receipt_path,dict(completed=True,profile_only=True,steps=steps,parameters=parameters,
            estimated_1000_steps_seconds=float(np.median(step_times[1:])*1000),losses=losses));return
    torch.save(net.state_dict(),folder/'weights.pt');net.eval()
    for season in ['winter','spring','summer']:
        evaluation=p if season=='winter' else load(season);ix=validation if season=='winter' else np.arange(len(evaluation['meta']))
        period='winter_validation' if season=='winter' else season;forecast=np.empty((len(ix),10),dtype=np.float32)
        with torch.inference_mode():
            for begin in range(0,len(ix),512):
                ids=ix[begin:begin+512];inputs,_=batch(evaluation,ids,stats,masks)
                forecast[begin:begin+len(ids)]=np.maximum(net(inputs).squeeze(-1).numpy(),0)
        write_prediction(folder/f'{period}.npz',evaluation['meta'].iloc[ix],forecast)
        print(json.dumps(dict(scored_predictions=period,condition=args.condition,seed=args.seed)),flush=True)
        del evaluation,forecast
    write_json(receipt_path,dict(completed=True,family='nhits',seed=args.seed,condition=args.condition,steps=steps,parameters=parameters,
        fit_seconds=fit_seconds,train_cases=len(train),history_exog_width=40,static_exog_width=88,
        recipe='Native N-HiTS default architecture/Adam/scheduler/1000steps; MSE task loss. Same model dimensions and case draws across information conditions. No HPO.',
        targets='Same original minute31..180 targets; refreshed query at+30,24hhistory and query-time state/commands.',
        hyperparameters=dict(net.hparams),losses=losses,versions={k:importlib.metadata.version(k) for k in ['torch','neuralforecast','numpy']}))


if __name__=='__main__':main()
