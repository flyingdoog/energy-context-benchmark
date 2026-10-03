"""Four owned local workers for the fixed three-seed ACN neural comparison."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor,as_completed

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/acn_fleet_months_v1/neural_references_v1'
PYTHON=ROOT/'.venv-neural'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')


def write_json(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8');temporary.replace(path)


def run(seed,site,condition,declarations=None):
    suffix='_'+declarations if declarations else ''
    folder=OUT/site/f'nhits_seed{seed}_{condition}{suffix}';folder.mkdir(parents=True,exist_ok=True);receipt=folder/'receipt.json'
    if receipt.exists() and json.loads(receipt.read_text()).get('completed'):
        return dict(seed=seed,site=site,condition=condition,completed=True,reused=True)
    args=[str(PYTHON),str(ROOT/'scripts/run_acn_native_neural.py'),'--seed',str(seed),'--site',site,'--condition',condition]
    if declarations:args+=['--declarations',declarations]
    start=time.monotonic();last_progress=start;stamp=None
    with (folder/'controller.log').open('a',encoding='utf-8') as log:
        process=subprocess.Popen(args,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        write_json(folder/'worker.json',dict(pid=process.pid,controller_pid=os.getpid(),site=site,condition=condition,seed=seed,status='running'))
        print(json.dumps(dict(started=condition,declarations=declarations,site=site,seed=seed,pid=process.pid)),flush=True)
        while process.poll() is None:
            progress=folder/'progress.json'
            if progress.exists() and progress.stat().st_mtime!=stamp:
                stamp=progress.stat().st_mtime;last_progress=time.monotonic()
            if time.monotonic()-last_progress>600:
                if os.name=='nt':
                    stopped=subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True,text=True,
                        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                    if stopped.returncode:raise RuntimeError(f'Cannot stop owned stalled process {process.pid}')
                else:process.terminate()
                process.wait(timeout=20)
                write_json(folder/'worker.json',dict(pid=process.pid,site=site,condition=condition,seed=seed,status='stopped_stalled'))
                raise RuntimeError(f'{site}/{seed}/{condition}: no progress for ten minutes; inspect before resume')
            time.sleep(5)
    completed=process.returncode==0 and receipt.exists() and json.loads(receipt.read_text()).get('completed')
    info=dict(seed=seed,site=site,condition=condition,declarations=declarations,completed=bool(completed),returncode=process.returncode,seconds=time.monotonic()-start)
    write_json(folder/'worker.json',dict(info,pid=process.pid,status='completed' if completed else 'failed'))
    print(json.dumps(info),flush=True)
    if not completed:raise RuntimeError(f'{site}/{seed}/{condition} failed; inspect its controller.log')
    return info


def main():
    global PYTHON,OUT
    parser=argparse.ArgumentParser();parser.add_argument('--python',type=Path,default=PYTHON)
    parser.add_argument('--workers',type=int,choices=[1,2,3,4],default=4)
    parser.add_argument('--declared',action='store_true');args=parser.parse_args();PYTHON=args.python.resolve()
    if args.declared:OUT=ROOT/'results/acn_fleet_months_v1/declared_demand_neural_v1'
    if not PYTHON.is_file():raise ValueError('Select an installed neural-environment interpreter with --python')
    OUT.mkdir(exist_ok=True);start=time.monotonic();results=[]
    jobs=[(seed,site,condition) for seed in [1,2,3] for condition in ['power','full','power_commands','power_state'] for site in ['caltech','office001']]
    if args.declared:jobs=[(seed,'caltech','full',view) for seed in [1,2,3] for view in ['padded','raw','gap']]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run,*job) for job in jobs]
        for future in as_completed(futures):
            results.append(future.result());write_json(OUT/'repeat_progress.json',dict(completed=len(results),total=len(jobs),results=results,seconds=time.monotonic()-start,controller_pid=os.getpid()))
    write_json(OUT/'repeat_receipt.json',dict(completed=True,seeds=[1,2,3],sites=['caltech'] if args.declared else ['caltech','office001'],results=results,
        seconds=time.monotonic()-start,workers=args.workers,declared_study=args.declared,
        recipe='Owned local CPU workers,two threads each. Fixed information conditions,three matched seeds. No HPO or GPU.'))


if __name__=='__main__':main()
