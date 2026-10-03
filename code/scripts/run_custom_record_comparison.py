"""Run the record comparison with a supplied fit/predict model; preview by default."""
import argparse
import gc
import json
import os
from pathlib import Path
import runpy
import time
import run_local
import numpy as np
import pandas as pd
from field_control.record_comparison import RecordInputView, fit_record_models, predict_record_models, load_record_cache
from field_control.record_reference_inputs import task_for
from field_control.record_reference_scores import aggregate_errors, paired_effects, KEYS, METRICS
from summarize_record_comparison import render_report


def execute(args):
    cache, data = args.inputs_cache.resolve(), args.data.resolve()
    description = load_record_cache(cache, data, args.source, args.site)
    if not args.run:
        result = dict(preview=True, source=args.source, site=args.site, support=description['support'],
            feature_columns=description['feature_columns'], fitted_model_objects=5,
            evaluation_streams_per_split=8, model_imported=False,
            note='Factory models receive tabular native-reference inputs; no automatic reshape or tuning')
        print(json.dumps(result))
        return result
    if args.output is None or args.model_script is None or not args.model_name:
        raise ValueError('--run needs --output, --model-script and --model-name')
    if args.batch_size < 1:
        raise ValueError('Batch size must be positive')
    output, script = args.output.resolve(), args.model_script.resolve()
    if not script.is_file():
        raise ValueError('The supplied model script does not exist')
    source_root = json.loads((cache/'plan.json').read_text()).get('original_source')
    for root in [cache, data] + ([Path(source_root).resolve()] if source_root else []):
        if root == output or root in output.parents or output in root.parents:
            raise ValueError('Keep output separate from source and prepared input directories')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new or empty output directory; arbitrary user models are not automatically resumed')
    namespace = runpy.run_path(str(script))
    factory = namespace.get('make_model')
    if not callable(factory):
        raise TypeError('The model script must define make_model(seed=2027)')
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    def write(name, value):
        (output/name).write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')

    def note(value):
        row = dict(value, at=time.strftime('%Y-%m-%dT%H:%M:%S'), pid=os.getpid(),
                   seconds=round(time.perf_counter()-started, 3))
        with (output/'progress.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row)+'\n')
        print(json.dumps(row), flush=True)

    write('protocol.json', dict(source=args.source, site=args.site, data_root=str(data),
        input_cache=str(cache), model_script=str(script), model_name=args.model_name,
        model_seed=2027, clipping=args.clip_nonnegative, batch_size=args.batch_size,
        automatic_hyperparameter_search=False, training_objects=5,
        fixed_arm='Reuses the paired fit; predict must not adapt its fitted state'))
    train = RecordInputView.from_cache(cache, 'train')
    models = fit_record_models(train, factory, progress=note)
    training_cases = len(train)
    del train
    gc.collect()
    daily_parts, packets = [], 0
    for item in description['support']:
        period = item['period']
        if period == 'train':
            continue
        view = RecordInputView.from_cache(cache, period)
        task = task_for(data, args.source, args.site, period)
        expected = task.meta.case_id.iloc[task.indices(period)].to_numpy(dtype=str)
        if not np.array_equal(expected, view.case_ids):
            raise ValueError('Recipient order differs from the complete native task split')
        folder = output/'predictions'/period
        folder.mkdir(parents=True, exist_ok=True)
        for method, prediction in predict_record_models(models, view, args.batch_size, args.clip_nonnegative):
            task.submission(folder/(method.replace(' ', '_')+'.npz'), split=period, prediction_kw=prediction)
            _, daily = task.score(split=period, prediction_kw=prediction, method=method)['power']
            daily_parts.append(daily)
            packets += 1
            note(dict(stage='scored', period=period, method=method, cases=len(view)))
        del view, task
        gc.collect()
    raw = pd.concat(daily_parts, ignore_index=True)
    raw.to_json(output/'raw_daily.json', orient='records')
    daily = aggregate_errors(raw)
    daily.to_json(output/'daily.json', orient='records')
    summary = daily.groupby([k for k in KEYS if k != 'date']+['method'], as_index=False)[METRICS].mean()
    summary.to_json(output/'summary.json', orient='records', indent=2)
    effects = paired_effects(daily)
    effects.to_json(output/'effects.json', orient='records', indent=2)
    receipt = dict(completed=True, purpose='Full-training user-model comparison',
        paper_result_eligible=True, source=args.source, site=args.site,
        model_name=args.model_name, training_recipe='User-supplied model; no automatic parameter search',
        training_cases=training_cases, original_training_cases=training_cases,
        fitted_models=5, prediction_packets=packets, model_seed=2027,
        raw_score_rows=len(raw), aggregate_score_rows=len(summary), effect_rows=len(effects),
        gpu_used='Determined by supplied model; the runner does not choose a device',
        seconds=time.perf_counter()-started)
    (output/'comparison.md').write_text(render_report(receipt, json.loads(effects.to_json(orient='records'))), encoding='utf-8')
    write('complete.json', receipt)
    note(dict(stage='complete', result=receipt))
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--inputs-cache', type=Path, required=True)
    parser.add_argument('--source', choices=['ACN', 'BPA'], required=True)
    parser.add_argument('--site', choices=['caltech', 'office001'])
    parser.add_argument('--model-script', type=Path)
    parser.add_argument('--model-name')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--clip-nonnegative', action='store_true', help='Explicit output clipping; disabled by default')
    return execute(parser.parse_args(argv))


if __name__ == '__main__':
    main()
