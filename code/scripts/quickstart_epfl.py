"""Public workbooks -> native task -> simple submission -> scores, without ML."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import run_local
from field_control.epfl_source import download_source, extract_source, REVISION
from field_control import OperationalForecastTask
import prepare_epfl_dataset
import example_operational_submission


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    origin = parser.add_mutually_exclusive_group(required=True)
    origin.add_argument('--download', action='store_true', help='Acquire the fixed public EPFL release')
    origin.add_argument('--raw', type=Path, help='Read already acquired unmodified source workbooks')
    parser.add_argument('--workdir', type=Path, required=True, help='New or previously completed EPFL quickstart folder')
    args = parser.parse_args(argv)
    work = args.workdir.resolve()
    raw = work / 'source' if args.download else args.raw.resolve()
    if args.raw and (raw == work or work in raw.parents or raw in work.parents):
        raise ValueError('Keep an existing source folder separate from the quickstart work directory')
    identity = dict(source_directory=str(raw), downloaded=bool(args.download),
                    revision=REVISION if args.download else None)
    plan_path = work / 'workflow.json'
    if plan_path.exists():
        prior = json.loads(plan_path.read_text())
        if any(prior.get(k) != v for k, v in identity.items()):
            raise ValueError('This work directory was used with a different source')
    else:
        if work.exists() and any(work.iterdir()):
            raise ValueError('Use an empty directory or a previous quickstart directory')
        work.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(json.dumps(identity, indent=2) + '\n', encoding='utf-8')
    record_path = work / 'quickstart_receipt.json'
    if record_path.exists():
        record = json.loads(record_path.read_text())
        if not record.get('completed') or any(record.get(k) != v for k, v in identity.items()):
            raise ValueError('This work directory was used with a different source')
        for relative in ['cache/source_receipt.json', 'dataset/receipt.json',
                         'dataset/cases.csv', 'dataset/packet.npz',
                         'example/example_submission.npz', 'example/receipt.json']:
            if not (work / relative).is_file():
                raise ValueError('Completed quickstart has missing outputs')
        print(json.dumps(dict(reused_completed_run=True, **record)), flush=True)
        return record
    started = time.monotonic()
    if args.download:
        download_source(raw)
    cache = work / 'cache'
    extracted = extract_source(raw, cache)
    packet = work / 'dataset'
    if not (packet / 'receipt.json').exists():
        prepare_epfl_dataset.main(['--source', str(cache), '--output', str(packet)])
    task = OperationalForecastTask(packet, 'EPFL')
    example = work / 'example'
    if not (example / 'receipt.json').exists():
        example_operational_submission.main(['--source', 'EPFL', '--data', str(packet),
            '--split', 'later', '--output', str(example)])
    submission = task.read_submission(example / 'example_submission.npz', split='later')
    scores = task.score(split='later', **submission, method='quickstart')
    record = dict(completed=True, **identity,
        active_source_rows=extracted['active_rows'], source_sessions=extracted['sessions'],
        cases={split: len(task.indices(split)) for split in ['train', 'validation', 'later']},
        source_clock='Native recorded timestamps; physical timezone unresolved',
        target_time_semantics=task.contract['native_time_semantics'],
        scores={name: summary.to_dict(orient='records') for name, (summary, _) in scores.items()},
        versions={name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'scipy', 'openpyxl']},
        new_model_fits=0, fitted_training_frequency=True, gpu_used=False,
        model_libraries_loaded=[name for name in ['torch', 'lightgbm', 'sklearn'] if name in sys.modules],
        seconds=time.monotonic() - started)
    record_path.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(record), flush=True)
    return record


if __name__ == '__main__':
    main()
