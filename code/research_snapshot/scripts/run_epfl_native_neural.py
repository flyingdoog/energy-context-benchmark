"""Fixed N-HiTS reference for actual EPFL power; run in .venv-neural only."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
for variable in ["OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"]:
    os.environ.setdefault(variable,"2")

OUT=ROOT/"results/epfl_neural_references_v1"
BASE=ROOT/"results/epfl_operational_pilot_v1"


def write_json(path,value):
    temporary=path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value,indent=2,default=str)+"\n",encoding="utf-8")
    temporary.replace(path)


def worker(condition,seed,profile=False):
    import numpy as np
    import pandas as pd
    import torch
    from neuralforecast.models import NHITS
    from neuralforecast.losses.pytorch import MSE
    from field_control.epfl_neural import HISTORY_NAMES,STATIC_NAMES,training_stats,normalized_inputs
    torch.set_num_threads(2)
    folder=OUT/f"nhits_seed{seed}_{condition}"
    folder.mkdir(parents=True,exist_ok=True)
    receipt=folder/("profile.json" if profile else "receipt.json")
    if receipt.exists() and json.loads(receipt.read_text()).get("completed"):
        print("Completed run reused",flush=True)
        return
    meta=pd.read_csv(BASE/"cases.csv")
    with np.load(BASE/"packet.npz",allow_pickle=False) as packet:
        if not np.array_equal(packet["case_id"].astype(str),meta.case_id.to_numpy(dtype=str)):
            raise ValueError("EPFL packet/case ordering differs")
        x=packet["x"];y=packet["y"][:,:,0]
    train=np.flatnonzero(meta.split.to_numpy()=="train")
    stats=training_stats(x,train)
    write_json(folder/"preprocessing.json",{k:v.tolist() for k,v in stats.items()})
    def batch(ids):
        arrays=normalized_inputs(x,ids,stats,condition)
        return {k:torch.from_numpy(v) for k,v in arrays.items()},torch.from_numpy(y[ids,:,None].astype(np.float32))
    torch.manual_seed(seed);np.random.seed(seed)
    net=NHITS(h=6,input_size=10,hist_exog_list=list(HISTORY_NAMES),
              stat_exog_list=list(STATIC_NAMES),loss=MSE(),random_seed=seed)
    configured=net.configure_optimizers()
    optimizer=configured["optimizer"];scheduler=configured["lr_scheduler"]["scheduler"]
    rng=np.random.default_rng(seed)
    groups=[train[ix] for ix in meta.iloc[train].groupby("plug",sort=True).indices.values()]
    def draw():
        selected=rng.choice(len(groups),min(net.batch_size,len(groups)),replace=False)
        pool=np.concatenate([groups[i] for i in selected])
        return rng.choice(pool,net.windows_batch_size,replace=len(pool)<net.windows_batch_size)
    checkpoint=folder/"resume.pt";start_step=0;elapsed=0.;losses=[]
    if checkpoint.exists() and not profile:
        saved=torch.load(checkpoint,map_location="cpu",weights_only=True)
        net.load_state_dict(saved["model"]);optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        rng.bit_generator.state=saved["rng"];torch.set_rng_state(saved["torch_rng"])
        start_step=saved["steps"];elapsed=saved["seconds"];losses=saved["losses"]
    steps=5 if profile else net.max_steps
    parameters=sum(p.numel() for p in net.parameters())
    print(json.dumps(dict(condition=condition,seed=seed,steps=steps,start_step=start_step,
                          parameters=parameters,pid=os.getpid())),flush=True)
    net.train();start=time.perf_counter();durations=[]
    for step in range(start_step,steps):
        tick=time.perf_counter()
        inp,target=batch(draw());optimizer.zero_grad(set_to_none=True)
        prediction=net(inp);loss=net.loss(y=target,y_hat=prediction)
        if not torch.isfinite(loss):raise ValueError("Nonfinite N-HiTS loss")
        loss.backward();optimizer.step();scheduler.step()
        losses.append(float(loss.detach()));durations.append(time.perf_counter()-tick)
        if (step+1)%100==0 or step+1==steps:
            progress=dict(condition=condition,seed=seed,steps=step+1,total=steps,
                          seconds=elapsed+time.perf_counter()-start,loss=losses[-1],pid=os.getpid())
            print(json.dumps(progress),flush=True)
            if not profile:
                tmp=folder/"resume.partial.pt"
                torch.save(dict(model=net.state_dict(),optimizer=optimizer.state_dict(),
                                scheduler=scheduler.state_dict(),rng=rng.bit_generator.state,
                                torch_rng=torch.get_rng_state(),steps=step+1,
                                seconds=progress["seconds"],losses=losses),tmp)
                tmp.replace(checkpoint);write_json(folder/"progress.json",progress)
    seconds=elapsed+time.perf_counter()-start
    if profile:
        write_json(receipt,dict(completed=True,profile_only=True,steps=steps,
                               parameters=parameters,losses=losses,
                               estimated_1000_steps_seconds=float(np.median(durations[1:])*1000)))
        return
    torch.save(net.state_dict(),folder/"weights.pt")
    net.eval();prediction=np.empty_like(y,dtype=np.float32)
    with torch.inference_mode():
        for begin in range(0,len(meta),512):
            ids=np.arange(begin,min(begin+512,len(meta)));inp,_=batch(ids)
            prediction[ids]=np.maximum(net(inp).squeeze(-1).numpy(),0)
    if not np.isfinite(prediction).all():raise ValueError("Nonfinite EPFL prediction")
    np.savez_compressed(folder/"predictions.npz",case_id=meta.case_id.to_numpy(dtype=str),
                        prediction_kw=prediction,target="actual_power")
    write_json(receipt,dict(completed=True,condition=condition,seed=seed,steps=steps,
        parameters=parameters,fit_seconds=seconds,training_cases=len(train),training_plugs=len(groups),
        recipe="Native N-HiTS defaults: architecture, Adam, scheduler, 1000steps, batch32plugs/1024windows. MSE objective. All two training plugs are available each draw; session windows sampled. No HPO.",
        interface="Ten one-minute actual-power inputs, eight typed historical covariates and four query-calendar values to six direct five-minute mean-power outputs. Excluded fields zeroed after train-only scaling; identical dimensions/init/case draws across conditions.",
        hyperparameters=dict(net.hparams),losses=losses,
        versions={name:importlib.metadata.version(name) for name in ["torch","neuralforecast","numpy"]}))
    print(json.dumps(dict(completed=True,condition=condition,seed=seed,fit_seconds=seconds)),flush=True)


def launch(condition,seed):
    folder=OUT/f"nhits_seed{seed}_{condition}"
    folder.mkdir(parents=True,exist_ok=True)
    receipt=folder/"receipt.json"
    if receipt.exists() and json.loads(receipt.read_text()).get("completed"):
        return dict(condition=condition,seed=seed,completed=True,reused=True)
    start=time.monotonic();last_change=start;stamp=None
    with (folder/"controller.log").open("a",encoding="utf-8") as log:
        process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),
                                  "--condition",condition,"--seed",str(seed)],cwd=ROOT,
                                  stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                  creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
        write_json(folder/"worker.json",dict(pid=process.pid,controller_pid=os.getpid(),status="running"))
        print(json.dumps(dict(started=condition,seed=seed,pid=process.pid)),flush=True)
        while process.poll() is None:
            progress=folder/"progress.json"
            # Observe both log and checkpoint progression, including initial load.
            current=(progress.stat().st_mtime if progress.exists() else None,
                     (folder/"controller.log").stat().st_size)
            if current!=stamp:stamp=current;last_change=time.monotonic()
            if time.monotonic()-last_change>600:
                write_json(folder/"worker.json",dict(pid=process.pid,status="needs_diagnosis",
                           reason="No checkpoint or log advancement for ten minutes"))
                print(json.dumps(dict(needs_diagnosis=condition,seed=seed,pid=process.pid)),flush=True)
                last_change=time.monotonic()
            time.sleep(5)
    complete=process.returncode==0 and receipt.exists() and json.loads(receipt.read_text()).get("completed")
    result=dict(condition=condition,seed=seed,completed=bool(complete),
                returncode=process.returncode,seconds=time.monotonic()-start,pid=process.pid)
    write_json(folder/"worker.json",dict(result,status="completed" if complete else "failed"))
    print(json.dumps(result),flush=True)
    if not complete:raise RuntimeError(f"EPFL {condition}/{seed} failed; inspect controller.log")
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--condition",choices=["power","power_state","full"])
    parser.add_argument("--seed",type=int,default=1)
    parser.add_argument("--profile",action="store_true")
    parser.add_argument("--batch",action="store_true")
    parser.add_argument("--workers",type=int,choices=[1,2,3],default=3)
    args=parser.parse_args()
    if args.batch:
        if args.profile or args.condition:parser.error("--batch cannot be combined with --condition/--profile")
        OUT.mkdir(exist_ok=True);start=time.monotonic();results=[]
        jobs=[(condition,seed) for seed in [1,2,3] for condition in ["power","power_state","full"]]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(launch,*job) for job in jobs]
            for future in as_completed(futures):
                results.append(future.result())
                write_json(OUT/"repeat_progress.json",dict(completed=len(results),total=9,
                    results=results,seconds=time.monotonic()-start,controller_pid=os.getpid()))
        write_json(OUT/"repeat_receipt.json",dict(completed=True,results=results,seeds=[1,2,3],
                   workers=args.workers,seconds=time.monotonic()-start,recipe="Fixed native N-HiTS; no HPO; CPU only."))
    else:
        if args.condition is None:parser.error("--condition required outside --batch")
        worker(args.condition,args.seed,args.profile)


if __name__=="__main__":
    main()

