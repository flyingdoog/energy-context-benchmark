"""Original ACN sessions -> shared task -> simple device/cohort submission."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import run_local
from field_control import OperationalForecastTask
import prepare_acn_dataset
import example_operational_submission


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True,
                        help='Existing manifest.csv and unmodified sessions/*.csv.gz')
    parser.add_argument('--workdir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args(argv)
    source, work = args.source.resolve(), args.workdir.resolve()
    if source == work or source in work.parents or work in source.parents:
        raise ValueError('Keep the original source outside the quickstart output')
    identity = dict(source_directory=str(source), workers=args.workers)
    plan = work / 'workflow.json'
    if plan.exists():
        if json.loads(plan.read_text()) != identity:
            raise ValueError('Quickstart directory belongs to another source or worker layout')
    else:
        if work.exists() and any(work.iterdir()):
            raise ValueError('Use an empty quickstart directory')
        prepare_acn_dataset.read_manifest(source)
        if not 1 <= args.workers <= 4:
            raise ValueError('Use one to four preparation workers')
        work.mkdir(parents=True, exist_ok=True)
        plan.write_text(json.dumps(identity, indent=2) + '\n')
    receipt_path = work / 'quickstart_receipt.json'
    if receipt_path.exists():
        record = json.loads(receipt_path.read_text())
        if not record.get('completed'):
            raise ValueError('Incomplete recorded quickstart')
        required = ['dataset/packet_receipt.json', 'dataset/packet.npz']
        for key in record['examples']:
            required += [f'example/{key}/receipt.json', f'example/{key}/example_submission.npz']
        if not all((work / name).is_file() for name in required):
            raise ValueError('Completed quickstart has missing outputs')
        print(json.dumps(dict(reused_completed_run=True, **record)), flush=True)
        return record
    started = time.monotonic()
    packet = work / 'dataset'
    prepared = prepare_acn_dataset.prepare(source, packet, args.workers)
    examples, support = {}, {}
    for site in ['caltech', 'office001']:
        task = OperationalForecastTask(packet, 'ACN', site=site)
        support[site] = {split: len(task.indices(split)) for split in ['train', 'validation', 'autumn', 'winter']}
        for split in ['autumn', 'winter']:
            key = f'{site}_{split}'
            example = work / 'example' / key
            if not (example / 'receipt.json').exists():
                example_operational_submission.main(['--source', 'ACN', '--data', str(packet),
                    '--site', site, '--split', split, '--output', str(example)])
            submission = task.read_submission(example / 'example_submission.npz', split=split)
            scores = task.score(split=split, **submission, method='quickstart')
            examples[key] = {track: summary.to_dict(orient='records')
                             for track, (summary, _) in scores.items()}
    record = dict(completed=True, **identity, cases=support,
        source_files=prepared['source_files'], manifest_contexts=prepared['manifest_contexts'],
        query_known_cases=prepared['query_known_cases'], examples=examples,
        acquisition='Existing original session files; no network access',
        source_clock='UTC source timestamps; America/Los_Angeles calendar splits',
        group_scope='Complete query-known cohorts of at least two distinct stations; no future arrivals',
        new_model_fits=0, fitted_training_frequency=True, gpu_used=False,
        model_libraries_loaded=[name for name in ['torch', 'lightgbm', 'sklearn'] if name in sys.modules],
        versions={name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'scipy']},
        seconds=time.monotonic()-started)
    receipt_path.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(dict(completed=True, cases=support, examples=list(examples),
                         seconds=record['seconds'], gpu_used=False)), flush=True)
    return record


if __name__ == '__main__':
    main()
