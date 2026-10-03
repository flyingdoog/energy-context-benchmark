"""Two-process controller for the fixed 39-run CU-BEMS N-HiTS comparison."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/cu_bems_neural_reference_v1'
LOCK=threading.Lock()


def write_json(path,value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n',encoding='utf8');temp.replace(path)


def worker(job):
    floor,condition,seed=job
    folder=OUT/f'floor{floor}'/f'nhits_seed{seed}_{condition}'
    folder.mkdir(parents=True,exist_ok=True)
    receipt=folder/'receipt.json'
    identity=dict(floor=floor,condition=condition,seed=seed)
    if receipt.exists() and json.loads(receipt.read_text()).get('completed'):
        return dict(**identity,completed=True,reused=True)
    started=time.monotonic();last_change=started;stamp=None
    with (folder/'controller.log').open('a',encoding='utf8') as log:
        process=subprocess.Popen([sys.executable,str(ROOT/'scripts/run_cu_bems_native_neural.py'),
            '--floor',str(floor),'--condition',condition,'--seed',str(seed)],cwd=ROOT,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        write_json(folder/'worker.json',dict(**identity,pid=process.pid,controller_pid=__import__('os').getpid(),status='running'))
        print(json.dumps(dict(started=True,**identity,pid=process.pid)),flush=True)
        while process.poll() is None:
            progress=folder/'progress.json'
            current=(progress.stat().st_mtime if progress.exists() else None,(folder/'controller.log').stat().st_size)
            if current!=stamp:stamp=current;last_change=time.monotonic()
            if time.monotonic()-last_change>600:
                write_json(folder/'attention.json',dict(**identity,pid=process.pid,
                    issue='No progress/log change for 10 minutes; supervisor should inspect the current stage',
                    seconds_without_change=time.monotonic()-last_change))
            time.sleep(5)
    completed=process.returncode==0 and receipt.exists() and json.loads(receipt.read_text()).get('completed',False)
    result=dict(**identity,completed=bool(completed),returncode=process.returncode,wall_seconds=time.monotonic()-started,pid=process.pid)
    write_json(folder/'worker.json',dict(**result,status='complete' if completed else 'failed'))
    print(json.dumps(result),flush=True)
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    jobs=[(floor,condition,seed) for floor in range(1,8)
        for seed in (1,2,3) for condition in (('power',) if floor==1 else ('power','thermal'))]
    # Continue the already measured run first and pair it with its thermal arm.
    first=[(3,'power',1),(3,'thermal',1)]
    jobs=first+[job for job in jobs if job not in first]
    start=time.monotonic();results=[]
    write_json(OUT/'batch_progress.json',dict(status='running',planned=len(jobs),completed=0,failed=0,
        workers=2,started_at_utc=datetime.now(timezone.utc).isoformat(),pid=__import__('os').getpid()))
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending=[pool.submit(worker,job) for job in jobs]
        for future in as_completed(pending):
            results.append(future.result())
            with LOCK:
                write_json(OUT/'batch_progress.json',dict(status='running',planned=len(jobs),
                    completed=sum(r['completed'] for r in results),failed=sum(not r['completed'] for r in results),
                    workers=2,seconds=time.monotonic()-start,results=results))
    completed=all(r['completed'] for r in results)
    result=dict(completed=completed,planned=len(jobs),workers=2,seconds=time.monotonic()-start,
        finished_at_utc=datetime.now(timezone.utc).isoformat(),results=results,gpu_used=False)
    write_json(OUT/'batch_receipt.json',result)
    write_json(OUT/'batch_progress.json',dict(status='complete' if completed else 'needs_attention',**result))
    print(json.dumps(dict(completed=completed,runs=len(results),seconds=result['seconds'])),flush=True)


if __name__=='__main__':main()
