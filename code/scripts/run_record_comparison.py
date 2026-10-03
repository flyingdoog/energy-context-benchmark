"""Prepared ACN/BPA data -> default four-arm record comparison; preview by default."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import time
import run_local
import numpy as np
import pandas as pd
from field_control.record_reference_inputs import describe_inputs, prepare_inputs, task_for, SEEDS
from field_control.record_linkage import reassign_blocks
from field_control.record_reference_scores import aggregate_errors, paired_effects, KEYS, METRICS


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')


def input_view(x, rows, moved, donor=None, power_only=False):
    value = reassign_blocks(x, donor, moved, rows) if donor is not None else np.array(x[rows], copy=True)
    if power_only:
        value[:, moved] = 0
    return value


def load_period(cache, period):
    folder = cache/period
    meta = pd.read_csv(folder/'metadata.csv.gz', dtype={'case_id': str, 'member': str})
    x = np.load(folder/'features.npy', mmap_mode='r', allow_pickle=False)
    with np.load(folder/'labeled.npz', allow_pickle=False) as p:
        rows, y, moved = p['rows'], p['target_kw'], p['moved']
    donors = {seed: np.load(folder/f'donor_{seed}.npy', allow_pickle=False) for seed in SEEDS}
    return meta, x, rows, y, moved, donors


def execute(args):
    data = args.data.resolve()
    description = describe_inputs(data, args.source, args.site)
    if not args.run and not args.prepare_only:
        print(json.dumps(dict(preview=True, **description)), flush=True)
        return description
    if args.output is None:
        raise ValueError('--output is required for execution')
    if not 1 <= args.workers <= 4 or not 1 <= args.threads <= 8 or args.smoke_train_cases < 0:
        raise ValueError('Use 1-4 workers, 1-8 threads and a nonnegative smoke limit')
    output = args.output.resolve()
    cache = args.inputs_cache.resolve() if args.inputs_cache else output/'inputs'
    original = args.original_source.resolve() if args.original_source else None
    for root in [data] + ([original] if original else []):
        if output == root or output in root.parents or root in output.parents:
            raise ValueError('Keep run outputs separate from source/prepared directories')
    plan = dict(data_root=str(data), source=args.source, site=args.site,
        original_source=str(original) if original else None, inputs_cache=str(cache),
        recipe='default-tree-record-comparison-v1', smoke_train_cases=args.smoke_train_cases,
        n_estimators=100, num_leaves=31, objective='regression', model_seed=2027,
        mapping_seeds=list(SEEDS), workers=args.workers, threads=args.threads, hpo=False)
    if (output/'protocol.json').exists():
        if json.loads((output/'protocol.json').read_text()) != plan:
            raise ValueError('Output belongs to another run; choose a new directory')
    elif output.exists() and any(output.iterdir()):
        raise ValueError('Use an empty output directory')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'protocol.json', plan)
    if (output/'complete.json').exists():
        record = json.loads((output/'complete.json').read_text())
        if not all((output/name).is_file() for name in ['raw_daily.json', 'daily.json', 'summary.json', 'effects.json']):
            raise ValueError('Completed run has missing scores')
        print(json.dumps(dict(reused=True, **record)), flush=True)
        return record
    started = time.perf_counter()

    def note(value):
        row = dict(value, at=time.strftime('%Y-%m-%dT%H:%M:%S'), pid=os.getpid(),
                   seconds=round(time.perf_counter()-started, 3))
        with (output/'progress.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row)+'\n')
        print(json.dumps(row), flush=True)

    note(dict(stage='prepare_inputs', **description))
    prepare_inputs(data, args.source, args.site, cache, original_source=original, progress=note)
    if args.prepare_only:
        record = dict(inputs_complete=True, model_fits=0, seconds=time.perf_counter()-started, **description)
        write_json(output/'input_receipt.json', record)
        return record

    # Preparation and preview do not load a model library.
    try:
        import sklearn
        from lightgbm import LGBMRegressor, Booster
    except ImportError as error:
        raise RuntimeError('Install requirements-record.txt in the active Python environment before --run') from error
    meta, x, train_rows, y, moved, donors = load_period(cache, 'train')
    original_training_cases = len(train_rows)
    if args.smoke_train_cases:
        train_rows, y = train_rows[:args.smoke_train_cases], y[:args.smoke_train_cases]
    training_cases = len(train_rows)
    if training_cases < 2:
        raise ValueError('At least two labeled training cases are required')
    model_specs = [('power', None, True), ('paired', None, False)] + [(f'retrained_{s}', s, False) for s in SEEDS]
    fitted = 0
    for name, seed, power in model_specs:
        folder = output/'models'/name
        folder.mkdir(parents=True, exist_ok=True)
        values = input_view(x, train_rows, moved, donors.get(seed), power)

        def fit(lead):
            target = folder/f'lead_{lead:02d}.txt'
            fresh = not target.exists()
            if fresh:
                model = LGBMRegressor(n_estimators=100, num_leaves=31, objective='regression',
                                     random_state=2027, n_jobs=args.threads, verbosity=-1)
                model.fit(values, y[:, lead])
                model.booster_.save_model(str(target))
            return lead, fresh

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for done in as_completed([pool.submit(fit, lead) for lead in range(10)]):
                lead, fresh = done.result()
                fitted += int(fresh)
                note(dict(stage='fit_head', arm=name, lead=lead, fitted_this_run=fresh))
        del values
        gc.collect()
    del x, meta, y, donors
    gc.collect()
    days = []
    for period in [r['period'] for r in description['support'] if r['period'] != 'train']:
        meta, x, rows, _, moved, donors = load_period(cache, period)
        task = task_for(data, args.source, args.site, period)
        expected = task.meta.case_id.iloc[task.indices(period)].to_numpy(dtype=str)
        if not np.array_equal(meta.case_id.iloc[rows].to_numpy(dtype=str), expected):
            raise ValueError('Cached and native evaluation case order differ')
        out = output/'predictions'/period
        out.mkdir(parents=True, exist_ok=True)
        specs = [('Power only', 'power', None, True), ('Paired', 'paired', None, False)]
        specs += [(f'Fixed {s}', 'paired', s, False) for s in SEEDS]
        specs += [(f'Retrained {s}', f'retrained_{s}', s, False) for s in SEEDS]
        for method, model_folder, seed, power in specs:
            path = out/(method.replace(' ', '_')+'.npz')
            if path.exists():
                value = task.read_submission(path, split=period)['prediction_kw']
            else:
                models = [Booster(model_file=str(output/'models'/model_folder/f'lead_{j:02d}.txt')) for j in range(10)]
                value = np.empty((len(rows), 10), dtype=np.float32)
                for start in range(0, len(rows), 4096):
                    ix = rows[start:start+4096]
                    features = input_view(x, ix, moved, donors.get(seed), power)
                    value[start:start+len(ix)] = np.column_stack([np.maximum(0, model.predict(features, num_threads=args.threads)) for model in models])
                task.submission(path, split=period, prediction_kw=value)
                del models
            summary, daily = task.score(split=period, prediction_kw=value, method=method)['power']
            days.append(daily)
            note(dict(stage='scored', period=period, method=method, cases=len(rows)))
        del task, x, meta, donors, value
        gc.collect()
    raw = pd.concat(days, ignore_index=True)
    raw.to_json(output/'raw_daily.json', orient='records')
    daily = aggregate_errors(raw)
    daily.to_json(output/'daily.json', orient='records')
    summary = daily.groupby([k for k in KEYS if k != 'date']+['method'], as_index=False)[METRICS].mean()
    summary.to_json(output/'summary.json', orient='records', indent=2)
    effects = paired_effects(daily)
    effects.to_json(output/'effects.json', orient='records', indent=2)
    record = dict(completed=True, purpose='training-subset smoke' if args.smoke_train_cases else 'full default reconstruction',
        paper_result_eligible=not bool(args.smoke_train_cases), source=args.source, site=args.site,
        training_cases=training_cases, original_training_cases=original_training_cases,
        fitted_heads_this_run=fitted, total_fitted_heads=50, new_mapping_choices=0,
        raw_score_rows=len(raw), aggregate_score_rows=len(summary), effect_rows=len(effects),
        seconds=time.perf_counter()-started, gpu_used=False, hpo=False,
        versions={name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'lightgbm', 'scikit-learn']},
        mapping='Complete auxiliary blocks and masks within original query populations; power/calendar protected',
        uncertainty='1500 paired calendar-week draws; conditional on fits; null if fewer than two weeks or zero denominator')
    write_json(output/'complete.json', record)
    note(dict(stage='complete', result=record))
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--source', choices=['ACN', 'BPA'], required=True)
    parser.add_argument('--site', choices=['caltech', 'office001'])
    parser.add_argument('--original-source', type=Path, help='ACN manifest and sessions for query-known input-only donors')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--inputs-cache', type=Path, help='Reusable cache created by this entry, not a historical experiment')
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--run', action='store_true', help='Fit default models and score all supported evaluation splits')
    actions.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--smoke-train-cases', type=int, default=0, help='Explicit training-subset smoke; never a paper result')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--threads', type=int, default=2)
    return execute(parser.parse_args(argv))


if __name__ == '__main__':
    main()
