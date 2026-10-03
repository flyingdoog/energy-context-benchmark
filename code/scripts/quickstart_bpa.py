"""Original BPA seasonal archives -> shared task -> device/cohort scores."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import run_local
from field_control import OperationalForecastTask
import example_operational_submission


SEASONS = ('winter', 'spring', 'summer')
EVALUATIONS = ('spring', 'summer')
TRACKS = ('power', 'probability', 'group_activity')


def prepare_dataset(source, output, workers):
    # Keep the wrapper independently testable without reading original archives.
    from field_control.bpa_preparation import prepare
    return prepare(source, output, seasons=SEASONS, workers=workers)


def check_example(folder, season):
    receipt = json.loads((folder / 'receipt.json').read_text(encoding='utf-8'))
    if not receipt.get('completed') or receipt.get('source') != 'BPA' or receipt.get('split') != season:
        raise ValueError('Incomplete or mismatched BPA example receipt')
    required = ['example_submission.npz'] + [f'{track}_scores.json' for track in TRACKS]
    if not all((folder / name).is_file() for name in required):
        raise ValueError('Completed quickstart has missing example outputs')


def check_completed_outputs(work):
    packet = work / 'dataset'
    required = [packet / 'update_validation' / name for name in ('split.npz', 'split.json')]
    for season in SEASONS:
        folder = packet / season
        receipt_path = folder / 'receipt.json'
        if not receipt_path.is_file():
            raise ValueError('Completed quickstart has missing dataset outputs')
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        required += [folder / name for name in
                     ('cases.csv.gz', 'target_kw.npy', 'target_observed_fraction.npy')]
        required += [folder / 'grid' / f'{name}.npy' for name in receipt['grid_fields']]
    if not all(path.is_file() for path in required):
        raise ValueError('Completed quickstart has missing dataset outputs')
    for season in EVALUATIONS:
        folder = work / 'example' / season
        if not (folder / 'receipt.json').is_file():
            raise ValueError('Completed quickstart has missing example outputs')
        check_example(folder, season)


def run(source, workdir, workers=2):
    source, work = Path(source).resolve(), Path(workdir).resolve()
    if workers not in (1, 2):
        raise ValueError('Use one or two preparation workers')
    if source == work or source in work.parents or work in source.parents:
        raise ValueError('Keep the original source outside the quickstart output')
    identity = dict(source='BPA', source_directory=str(source), workers=workers,
                    workflow_version='bpa-quickstart-v1')
    plan = work / 'workflow.json'
    if plan.exists():
        if json.loads(plan.read_text(encoding='utf-8')) != identity:
            raise ValueError('Quickstart directory belongs to another source or worker layout')
    else:
        if work.exists() and (not work.is_dir() or any(work.iterdir())):
            raise ValueError('Use an empty quickstart directory')
        missing = [f'{season}2018.zip' for season in SEASONS
                   if not (source / f'{season}2018.zip').is_file()]
        if missing:
            raise ValueError('Missing original BPA source archives: ' + ', '.join(missing))
        work.mkdir(parents=True, exist_ok=True)
        plan.write_text(json.dumps(identity, indent=2) + '\n', encoding='utf-8')
    receipt_path = work / 'quickstart_receipt.json'
    if receipt_path.exists():
        record = json.loads(receipt_path.read_text(encoding='utf-8'))
        if not record.get('completed') or any(record.get(k) != v for k, v in identity.items()):
            raise ValueError('Incomplete or mismatched quickstart receipt')
        if set(record.get('examples', {})) != set(EVALUATIONS):
            raise ValueError('Completed quickstart is missing a seasonal example')
        check_completed_outputs(work)
        print(json.dumps(dict(reused_completed_run=True, **record)), flush=True)
        return record

    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    packet = work / 'dataset'
    prepared = prepare_dataset(source, packet, workers)
    if not prepared.get('completed'):
        raise ValueError('BPA preparation did not complete')
    winter = OperationalForecastTask(packet, 'BPA', season='winter')
    support = {'winter': {split: len(winter.indices(split)) for split in ('train', 'validation')}}
    del winter
    examples = {}
    for season in EVALUATIONS:
        task = OperationalForecastTask(packet, 'BPA', season=season)
        support[season] = {season: len(task.indices(season))}
        example = work / 'example' / season
        if not (example / 'receipt.json').exists():
            example_operational_submission.main(['--source', 'BPA', '--data', str(packet),
                '--season', season, '--split', season, '--output', str(example)])
        check_example(example, season)
        submission = task.read_submission(example / 'example_submission.npz', split=season)
        scores = task.score(split=season, **submission, method='quickstart', require_groups=True)
        examples[season] = {track: summary.to_dict(orient='records')
                            for track, (summary, _) in scores.items()}
        del task
    record = dict(completed=True, **identity, cases=support, examples=examples,
        preparation=prepared, source_files=[f'{season}2018.zip' for season in SEASONS],
        acquisition='Existing original seasonal archives; no network access',
        source_clock='UTC source timestamps; America/Los_Angeles calendar splits',
        query_shift_from_original_minutes=30, target_bins=10, bin_minutes=15,
        target_report_offsets_minutes=[[1+15*i, 15+15*i] for i in range(10)],
        group_scope='Original label-eligible synchronous groups of at least ten devices',
        new_model_fits=0, new_ml_fits=0, fitted_training_frequency=True, gpu_used=False,
        model_libraries_loaded=[name for name in ('torch', 'lightgbm', 'sklearn') if name in sys.modules],
        versions=dict(python=sys.version.split()[0], **{name: importlib.metadata.version(name)
                      for name in ('numpy', 'pandas', 'scipy', 'inflate64')}),
        started_at_utc=started_at, finished_at_utc=datetime.now(timezone.utc).isoformat(),
        seconds=time.monotonic()-started)
    receipt_path.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(completed=True, cases=support, examples=list(examples),
                         seconds=record['seconds'], gpu_used=False)), flush=True)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True,
                        help='Existing winter2018.zip, spring2018.zip and summer2018.zip')
    parser.add_argument('--workdir', type=Path, required=True)
    parser.add_argument('--workers', type=int, choices=(1, 2), default=2)
    args = parser.parse_args(argv)
    return run(args.source, args.workdir, args.workers)


if __name__ == '__main__':
    main()
