"""Prepare original ACN session files without historical research outputs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import run_local
import numpy as np
import pandas as pd
from field_control.acn_preparation import build_acn_packet, read_manifest


def prepare(source, output, workers=4, timeout_seconds=900):
    source, output = Path(source).resolve(), Path(output).resolve()
    if not 1 <= workers <= 4:
        raise ValueError('Use one to four preparation workers')
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('Keep source and output directories separate')
    manifest = read_manifest(source)
    identity = dict(source_directory=str(source), workers=workers)
    plan = output / 'preparation.json'
    if plan.exists():
        if json.loads(plan.read_text()) != identity:
            raise ValueError('Preparation directory belongs to another source or worker layout')
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError('Use an empty preparation directory')
        output.mkdir(parents=True, exist_ok=True)
        plan.write_text(json.dumps(identity, indent=2) + '\n')
    receipt_path = output / 'packet_receipt.json'
    required = ['cases.csv.gz', 'query_population.csv.gz', 'packet.npz', 'source_coverage.csv', 'support.csv']
    if receipt_path.exists():
        record = json.loads(receipt_path.read_text())
        if not record.get('completed') or not all((output / name).is_file() for name in required):
            raise ValueError('Incomplete recorded packet')
        return record
    started = time.monotonic()
    logs = output / 'logs'
    logs.mkdir(exist_ok=True)
    jobs = []
    try:
        for shard in range(workers):
            folder = output / 'shards' / f'{shard:02d}'
            if (folder / 'packet_receipt.json').exists():
                if not all((folder / name).is_file() for name in required):
                    raise ValueError('Completed shard has missing outputs')
                continue
            stream = (logs / f'{shard:02d}.log').open('w')
            command = [sys.executable, str(Path(__file__).resolve()),
                       '--source', str(source), '--output', str(folder),
                       '--shards', str(workers), '--shard-index', str(shard)]
            try:
                process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            except BaseException:
                stream.close()
                raise
            jobs.append((shard, process, stream))
        while any(process.poll() is None for _, process, _ in jobs):
            state = {str(shard): process.poll() for shard, process, _ in jobs}
            progress = dict(shards=state, seconds=round(time.monotonic()-started, 2))
            (output / 'progress.json').write_text(json.dumps(progress) + '\n')
            print(json.dumps(progress), flush=True)
            if any(code not in (None, 0) for code in state.values()):
                raise RuntimeError(f'Preparation failed; inspect logs in {logs}')
            if time.monotonic()-started > timeout_seconds:
                raise TimeoutError('Preparation exceeded its 15-minute execution bound; inspect logs')
            time.sleep(5)
        if any(process.returncode != 0 for _, process, _ in jobs):
            raise RuntimeError(f'Preparation failed; inspect logs in {logs}')
    finally:
        for _, process, stream in jobs:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            stream.close()
    metadata, population, coverage, arrays, targets, receipts = [], [], [], [], [], []
    for shard in range(workers):
        folder = output / 'shards' / f'{shard:02d}'
        receipts.append(json.loads((folder / 'packet_receipt.json').read_text()))
        metadata.append(pd.read_csv(folder / 'cases.csv.gz'))
        population.append(pd.read_csv(folder / 'query_population.csv.gz'))
        coverage.append(pd.read_csv(folder / 'source_coverage.csv'))
        with np.load(folder / 'packet.npz', allow_pickle=False) as packet:
            arrays.append(packet['x'])
            targets.append(packet['y'])
    meta = pd.concat(metadata, ignore_index=True)
    pop = pd.concat(population, ignore_index=True)
    if not len(meta) or not meta.case_id.is_unique or not pop.case_id.is_unique:
        raise ValueError('Expected nonempty unique cases across calendar contexts')
    if any(r['feature_groups'] != receipts[0]['feature_groups'] for r in receipts):
        raise ValueError('Feature layout differs across shards')
    order = np.argsort(meta.case_id.to_numpy(), kind='stable')
    meta = meta.iloc[order].reset_index(drop=True)
    x, y = np.concatenate(arrays)[order], np.concatenate(targets)[order]
    meta.to_csv(output / 'cases.csv.gz', index=False)
    pop.sort_values('case_id').to_csv(output / 'query_population.csv.gz', index=False)
    np.savez_compressed(output / 'packet.npz', x=x, y=y)
    pd.concat(coverage, ignore_index=True).to_csv(output / 'source_coverage.csv', index=False)
    support = meta.groupby(['site', 'split']).agg(cases=('case_id', 'size'),
        sessions=('filename', 'nunique'), dates=('local_date', 'nunique'),
        stations=('station_id', 'nunique')).reset_index()
    support.to_csv(output / 'support.csv', index=False)
    record = dict(receipts[0], source_directory=str(source),
        source_files=int(manifest.filename.nunique()), manifest_contexts=len(manifest),
        cases=len(meta), query_known_cases=len(pop), seconds=time.monotonic()-started,
        preparation_workers=workers, case_order='case_id',
        split_policy='Explicit local calendar bounds contain the query and all 150 forecast minutes.',
        new_model_fits=0, gpu_used=False)
    receipt_path.write_text(json.dumps(record, indent=2) + '\n')
    print(support.to_string(index=False), flush=True)
    print(json.dumps(record), flush=True)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--shards', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--shard-index', type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.shards is not None:
        return build_acn_packet(args.source, args.output, args.shards, args.shard_index)
    return prepare(args.source, args.output, args.workers)


if __name__ == '__main__':
    main()
