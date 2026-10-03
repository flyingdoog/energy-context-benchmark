"""Bounded local repeat batch for the already specified typed-input comparison."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/benchmark_bpa_sequences_v1/update_validation_typed'
PYTHON=ROOT/'.venv-neural/Scripts/python.exe'
CONDITIONS=('initial','physical','command','full')


def write_json(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)


def run(seed,condition):
    folder=OUT/f'nhits_seed{seed}_{condition}';folder.mkdir(exist_ok=True)
    receipt=folder/'receipt.json'
    if receipt.exists() and json.loads(receipt.read_text()).get('completed'):
        return dict(seed=seed,condition=condition,completed=True,reused=True)
    args=[str(PYTHON),str(ROOT/'scripts/run_bpa_update_validation_neural.py'),
          '--encoding','typed','--seed',str(seed),'--condition',condition]
    start=time.monotonic();last_progress=start;stamp=None
    with (folder/'controller.log').open('a',encoding='utf-8') as log:
        process=subprocess.Popen(args,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        write_json(folder/'worker.json',dict(pid=process.pid,controller_pid=os.getpid(),seed=seed,
            condition=condition,started_unix=time.time(),status='running'))
        print(json.dumps(dict(started=condition,seed=seed,pid=process.pid)),flush=True)
        while process.poll() is None:
            progress=folder/'progress.json'
            if progress.exists() and progress.stat().st_mtime!=stamp:
                stamp=progress.stat().st_mtime;last_progress=time.monotonic()
            if time.monotonic()-last_progress>600:
                process.terminate()
                try:process.wait(timeout=20)
                except subprocess.TimeoutExpired:process.kill();process.wait()
                write_json(folder/'worker.json',dict(pid=process.pid,seed=seed,condition=condition,
                    status='stopped_stalled',seconds=time.monotonic()-start))
                raise RuntimeError(f'{seed}/{condition}: no step progress for ten minutes; inspect before resuming')
            time.sleep(5)
    completed=process.returncode==0 and receipt.exists() and json.loads(receipt.read_text()).get('completed')
    result=dict(seed=seed,condition=condition,completed=bool(completed),returncode=process.returncode,
                seconds=time.monotonic()-start,pid=process.pid)
    write_json(folder/'worker.json',dict(result,status='completed' if completed else 'failed'))
    if not completed:raise RuntimeError(f'{seed}/{condition} failed; see {folder}/controller.log')
    print(json.dumps(result),flush=True);return result


def main():
    start=time.monotonic();results=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(run,seed,condition) for seed in (2,3) for condition in CONDITIONS]
        for future in as_completed(futures):
            results.append(future.result())
            write_json(OUT/'repeat_progress.json',dict(completed=len(results),total=8,
                controller_pid=os.getpid(),results=results,seconds=time.monotonic()-start))
    write_json(OUT/'repeat_receipt.json',dict(completed=True,results=results,
        wall_seconds=time.monotonic()-start,recipe='Only random seeds2/3 added; no hyperparameter changes, no new information conditions. Four local workers, two neural threads each.'))


if __name__=='__main__':main()
