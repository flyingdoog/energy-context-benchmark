"""Native N-HiTS replication on the shared chronological, fixed-target task.

Run in .venv-neural. No project .deps overlay: keep the neural runtime intact.
The four conditions share architecture, cases, draws and default training recipe.
"""
import argparse
import importlib.metadata
import os
import time
import json
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import numpy as np
import torch
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import MSE
from run_bpa_native_neural import moments,normalize,write_json
from bpa_update_validation_common import (OUT,DEST,CONDITIONS,load,load_split,
    static,evaluation_parts,write_prediction)
import bpa_typed_inputs as typed_inputs


def training_stats(packet,train,condition,typed=False):
    raw=packet['history'][train]
    if typed:
        desc=typed_inputs.description()
        return dict(history=typed_inputs.numeric_only_stats(moments(typed_inputs.history_exog(packet,train)),desc['history_exog_names']),
            static=typed_inputs.numeric_only_stats(moments(typed_inputs.static(packet,train,condition)),desc['static_exog_names']),
            power_mean=float(np.nanmean(raw)),typed=True)
    return dict(typed=False,history=moments(packet['history_exog'][train]),
                static=moments(static(packet,train,condition)),
                power_mean=float(np.nanmean(raw)))


def batch(packet,indices,stats,condition):
    raw=np.asarray(packet['history'][indices],dtype=np.float32)
    power=np.where(np.isfinite(raw),raw,stats['power_mean'])
    calendar=packet['calendar'][indices]
    calendar=np.concatenate([calendar[:,:96],calendar[:,98:]],axis=1)
    hist=typed_inputs.history_exog(packet,indices) if stats['typed'] else packet['history_exog'][indices]
    stat=typed_inputs.static(packet,indices,condition) if stats['typed'] else static(packet,indices,condition)
    inputs=dict(insample_y=torch.from_numpy(power[...,None]),
        insample_mask=torch.from_numpy(np.isfinite(raw)[...,None].astype(np.float32)),
        hist_exog=torch.from_numpy(normalize(hist,stats['history'])),
        stat_exog=torch.from_numpy(normalize(stat,stats['static'])),
        futr_exog=torch.from_numpy(calendar))
    return inputs,torch.from_numpy(np.array(packet['target'][indices,2:],copy=True)[...,None])


def main():
    global DEST
    parser=argparse.ArgumentParser()
    parser.add_argument('--condition',choices=CONDITIONS,required=True)
    parser.add_argument('--seed',type=int,default=1)
    parser.add_argument('--encoding',choices=['raw','typed'],default='raw')
    args=parser.parse_args();torch.set_num_threads(2)
    typed=args.encoding=='typed'
    if typed:DEST=DEST.parent/'update_validation_typed';DEST.mkdir(exist_ok=True)
    folder=DEST/f'nhits_seed{args.seed}_{args.condition}';folder.mkdir(exist_ok=True)
    receipt_path=folder/'receipt.json'
    if receipt_path.exists() and json.loads(receipt_path.read_text()).get('completed'):
        print('Completed run reused',flush=True);return
    train,validation=load_split();packet=load('winter')
    stats=training_stats(packet,train,args.condition,typed)
    write_json(folder/'preprocessing.json',dict(history=[v.tolist() for v in stats['history']],
        static=[v.tolist() for v in stats['static']],power_mean=stats['power_mean'],
        fitted_cases=len(train),training_only=True))
    desc=typed_inputs.description() if typed else json.loads((OUT/'model_packets/winter/receipt.json').read_text())
    static_names=desc['static_exog_names'] if typed else desc['static_exog_names']+[f'observed_minute_{m}_field_{f}' for m in range(1,31) for f in range(23)]
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    net=NHITS(h=10,input_size=96,hist_exog_list=desc['history_exog_names'],
        stat_exog_list=static_names,futr_exog_list=desc['calendar_names'],loss=MSE(),random_seed=args.seed)
    config=net.configure_optimizers();optimizer=config['optimizer'];scheduler=config['lr_scheduler']['scheduler']
    rng=np.random.default_rng(args.seed)
    groups=[train[ix] for ix in packet['meta'].iloc[train].groupby('alias',sort=True).indices.values()]
    def draw():
        chosen=rng.choice(len(groups),min(net.batch_size,len(groups)),replace=False)
        pool=np.concatenate([groups[i] for i in chosen])
        return rng.choice(pool,net.windows_batch_size,replace=len(pool)<net.windows_batch_size)
    probe,_=batch(packet,train[:8],stats,args.condition);net.eval()
    response={}
    columns={'initial_command':desc['static_exog_names'].index('current_command_is_1')}
    initial_width=len(desc['initial_static']) if typed else 70
    physical_width=typed_inputs.PHYSICAL_WIDTH if typed else 12
    if args.condition in ('physical','full'):columns['feedback_power']=initial_width
    if args.condition in ('command','full'):columns['feedback_command']=initial_width+physical_width
    with torch.no_grad():
        base=net(probe)
        for name,col in columns.items():
            altered={k:v.clone() for k,v in probe.items()};altered['stat_exog'][:,col]+=1
            response[name]=float((net(altered)-base).abs().max())
    if any(not np.isfinite(v) or v<=0 for v in response.values()):
        write_json(folder/'wiring_issue.json',response)
        raise ValueError('Inspect native covariate wiring before interpreting performance')
    checkpoint=folder/'resume.pt';start_step=0;elapsed=0.;losses=[]
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        net.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler']);rng.bit_generator.state=saved['rng']
        torch.set_rng_state(saved['torch_rng']);start_step=saved['steps']
        elapsed=saved['seconds'];losses=saved['losses']
    parameters=sum(p.numel() for p in net.parameters())
    print(json.dumps(dict(condition=args.condition,seed=args.seed,start_step=start_step,
                         total_steps=net.max_steps,parameters=parameters,wiring=response,pid=os.getpid())),flush=True)
    net.train();start=time.perf_counter()
    for step in range(start_step,net.max_steps):
        inputs,y=batch(packet,draw(),stats,args.condition)
        optimizer.zero_grad(set_to_none=True);pred=net(inputs);loss=net.loss(y=y,y_hat=pred)
        if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
        loss.backward();optimizer.step();scheduler.step();losses.append(float(loss.detach()))
        if (step+1)%100==0 or step+1==net.max_steps:
            seconds=elapsed+time.perf_counter()-start
            progress=dict(condition=args.condition,seed=args.seed,steps=step+1,total=net.max_steps,
                          seconds=seconds,loss=losses[-1],pid=os.getpid())
            tmp=folder/'resume.partial.pt'
            torch.save(dict(model=net.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),steps=step+1,seconds=seconds,losses=losses),tmp)
            tmp.replace(checkpoint);write_json(folder/'progress.json',progress)
            print(json.dumps(progress),flush=True)
    fit_seconds=elapsed+time.perf_counter()-start;torch.save(net.state_dict(),folder/'weights.pt')
    net.eval();packets={'winter':packet,'spring':load('spring'),'summer':load('summer')}
    for period,p,indices in evaluation_parts(packets,validation):
        forecast=np.empty((len(indices),10),dtype=np.float32)
        with torch.inference_mode():
            for begin in range(0,len(indices),512):
                ix=indices[begin:begin+512];inputs,_=batch(p,ix,stats,args.condition)
                forecast[begin:begin+len(ix)]=np.maximum(net(inputs).squeeze(-1).numpy(),0)
        write_prediction(folder/f'{period}.npz',p['meta'].iloc[indices],forecast)
        print(json.dumps(dict(predictions_complete=period,cases=len(indices),condition=args.condition)),flush=True)
    write_json(receipt_path,dict(completed=True,family='nhits',seed=args.seed,condition=args.condition,
        train_cases=len(train),fit_seconds=fit_seconds,steps=net.max_steps,parameters=parameters,
        history_exog_width=len(desc['history_exog_names']),static_exog_width=len(static_names),future_exog_width=4,wiring=response,
        encoding=args.encoding,
        recipe=f'Native N-HiTS default architecture, Adam/scheduler, 1000 steps, seed{args.seed}, MSE. No HPO. Same architecture and case draws across conditions; inactive feedback columns zero.',
        target='Ten15-minute bins from original minute31 to180; original96-bin history retained. Intermediate calendar bins excluded from future input.',
        preprocessing='Training-only moments and power imputation. Typed mode leaves categorical one-hots and masks in[0,1], standardizing numeric quantities only. No validation labels used in fit.',
        hyperparameters=dict(net.hparams),losses=losses,
        versions={k:importlib.metadata.version(k) for k in ['neuralforecast','torch','numpy']}))
    print(json.dumps(dict(completed=args.condition,fit_seconds=fit_seconds)),flush=True)


if __name__=='__main__':main()
