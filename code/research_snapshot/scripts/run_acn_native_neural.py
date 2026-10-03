"""Fixed native N-HiTS recipe for ACN; run only in the neural environment."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(name,'2')
import numpy as np
import pandas as pd
import torch
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import MSE
from field_control.acn_neural import HISTORY_NAMES,STATIC_NAMES,CONDITIONS,training_stats,normalized_numpy
from field_control.acn_requests import DECLARATION_NAMES,declaration_features,declaration_stats,normalized_declarations

DATA=ROOT/'results/acn_fleet_months_v1';PACKETS=DATA/'neural_packets_v1';DEST=DATA/'neural_references_v1'


def write_json(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2,default=str)+'\n',encoding='utf-8');temporary.replace(path)


def main():
    global DEST
    parser=argparse.ArgumentParser();parser.add_argument('--site',choices=['caltech','office001'],required=True)
    parser.add_argument('--condition',choices=CONDITIONS,required=True);parser.add_argument('--seed',type=int,default=1)
    parser.add_argument('--profile',action='store_true')
    parser.add_argument('--declarations',choices=['none','padded','raw','gap'],default='none');args=parser.parse_args()
    suffix='' if args.declarations=='none' else '_'+args.declarations
    if suffix:DEST=DATA/'declared_demand_neural_v1'
    torch.set_num_threads(2);folder=DEST/args.site/f'nhits_seed{args.seed}_{args.condition}{suffix}';folder.mkdir(parents=True,exist_ok=True)
    receipt=folder/('profile.json' if args.profile else 'receipt.json')
    if receipt.exists() and json.loads(receipt.read_text()).get('completed'):
        print('Completed run reused',flush=True);return
    packet={name:np.load(PACKETS/(name+'.npy'),mmap_mode='r') for name in ['history','history_exog','static_exog','target']}
    meta=pd.read_csv(PACKETS/'cases.csv.gz');train=np.flatnonzero((meta.site==args.site)&(meta.split=='train'))
    stats=training_stats(packet,train)
    declaration=None;static_names=list(STATIC_NAMES)
    if args.declarations!='none':
        requests=pd.read_csv(DATA/'request_metadata_v1/query_requests.csv.gz').set_index('case_id').reindex(meta.case_id)
        if requests.site.isna().any():raise ValueError('Missing declaration input rows')
        raw,gap=declaration_features(requests.requested_kwh,requests.requested_departure_hours,packet['history_exog'][:,-1,5])
        extra=np.concatenate([raw,gap],axis=1);extra_stats=declaration_stats(extra,train)
        declaration=normalized_declarations(extra,extra_stats,args.declarations)
        static_names+=list(DECLARATION_NAMES)
        write_json(folder/'declaration_preprocessing.json',dict(mean=extra_stats[0].tolist(),scale=extra_stats[1].tolist(),view=args.declarations,
            information='All three views use the same six static slots; missing and excluded entries are zero after scaling. Numeric stats use observed training values only.'))
    write_json(folder/'preprocessing.json',{name:value.tolist() if isinstance(value,np.ndarray) else value for name,value in stats.items()})
    def batch(indices):
        arrays=normalized_numpy(packet,indices,stats,args.condition)
        if declaration is not None:arrays['stat_exog']=np.concatenate([arrays['stat_exog'],declaration[indices]],axis=1)
        return {name:torch.from_numpy(value) for name,value in arrays.items()},torch.from_numpy(np.array(packet['target'][indices],copy=True)[...,None])
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    net=NHITS(h=10,input_size=60,hist_exog_list=list(HISTORY_NAMES),stat_exog_list=static_names,
              loss=MSE(),random_seed=args.seed)
    optimizer_config=net.configure_optimizers();optimizer=optimizer_config['optimizer'];scheduler=optimizer_config['lr_scheduler']['scheduler']
    rng=np.random.default_rng(args.seed)
    groups=[train[ix] for ix in meta.iloc[train].groupby('station_id',sort=True).indices.values()]
    def draw():
        chosen=rng.choice(len(groups),min(net.batch_size,len(groups)),replace=False)
        pool=np.concatenate([groups[i] for i in chosen]);return rng.choice(pool,net.windows_batch_size,replace=len(pool)<net.windows_batch_size)
    checkpoint=folder/'resume.pt';start_step=0;elapsed=0.;losses=[]
    if checkpoint.exists() and not args.profile:
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        net.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        rng.bit_generator.state=saved['rng'];torch.set_rng_state(saved['torch_rng'])
        start_step=saved['steps'];elapsed=saved['seconds'];losses=saved['losses']
    steps=5 if args.profile else net.max_steps;parameters=sum(p.numel() for p in net.parameters())
    print(json.dumps(dict(site=args.site,condition=args.condition,seed=args.seed,steps=steps,start_step=start_step,parameters=parameters,pid=os.getpid())),flush=True)
    step_times=[];net.train();start=time.perf_counter()
    for step in range(start_step,steps):
        tick=time.perf_counter();inputs,y=batch(draw());optimizer.zero_grad(set_to_none=True)
        prediction=net(inputs);loss=net.loss(y=y,y_hat=prediction)
        if not torch.isfinite(loss):raise ValueError('Nonfinite loss: inspect implementation and data before drawing conclusions')
        loss.backward();optimizer.step();scheduler.step();losses.append(float(loss.detach()));step_times.append(time.perf_counter()-tick)
        if (step+1)%100==0 or step+1==steps:
            seconds=elapsed+time.perf_counter()-start
            progress=dict(site=args.site,condition=args.condition,seed=args.seed,steps=step+1,total=steps,seconds=seconds,loss=losses[-1],pid=os.getpid())
            print(json.dumps(progress),flush=True)
            if not args.profile:
                tmp=folder/'resume.partial.pt'
                torch.save(dict(model=net.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                    rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),steps=step+1,seconds=seconds,losses=losses),tmp)
                tmp.replace(checkpoint);write_json(folder/'progress.json',progress)
    fit_seconds=elapsed+time.perf_counter()-start
    if args.profile:
        write_json(receipt,dict(completed=True,profile_only=True,steps=steps,parameters=parameters,losses=losses,
            estimated_1000_steps_seconds=float(np.median(step_times[1:])*1000)));return
    torch.save(net.state_dict(),folder/'weights.pt');net.eval()
    indices=np.flatnonzero(meta.site==args.site);prediction=np.empty((len(indices),10),dtype=np.float32)
    with torch.inference_mode():
        for begin in range(0,len(indices),512):
            ids=indices[begin:begin+512];inputs,_=batch(ids)
            prediction[begin:begin+len(ids)]=np.maximum(net(inputs).squeeze(-1).numpy(),0)
    if not np.isfinite(prediction).all():raise ValueError('Nonfinite saved forecast')
    np.savez_compressed(folder/'predictions.npz',case_id=meta.case_id.to_numpy(dtype=str)[indices],prediction_kw=prediction,target_minutes=np.arange(0,151,15))
    write_json(receipt,dict(completed=True,site=args.site,condition=args.condition,declarations=args.declarations,seed=args.seed,steps=steps,
        parameters=parameters,fit_seconds=fit_seconds,training_cases=len(train),training_stations=len(groups),
        recipe='Native N-HiTS defaults:architecture,Adam,scheduler,1000steps,batch32stations/1024windows. MSE objective. Same architecture/case draws across information conditions. No HPO or validation-based selection.',
        interface='60one-minute historical inputs to10direct fifteen-minute average outputs; h denotes output bins. Current query calendar/elapsed time only. Numeric exogenous scaling fitted on this site training split; masks/categories unscaled.',
        hyperparameters=dict(net.hparams),losses=losses,versions={name:importlib.metadata.version(name) for name in ['torch','neuralforecast','numpy']}))
    print(json.dumps(dict(completed=True,site=args.site,condition=args.condition,seed=args.seed,fit_seconds=fit_seconds)),flush=True)


if __name__=='__main__':main()
