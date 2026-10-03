"""Evaluate the group-level control on saved four-arm packets; preview by default."""
import argparse
import json
from pathlib import Path
import time
import run_local
from field_control.group_error_level import group_residuals
from field_control.record_level_diagnostic import analyze_groups, write_result
from field_control.record_reference_inputs import task_for
from field_control.record_reference_scores import RAW_ARMS


def inspect_run(run, data, source, site):
    if (source == 'ACN' and site not in ('caltech','office001')) or (source == 'BPA' and site is not None):
        raise ValueError('ACN needs its site; BPA does not use a site argument')
    if not data.is_dir():
        raise ValueError('The prepared dataset directory does not exist')
    protocol = json.loads((run/'protocol.json').read_text(encoding='utf-8'))
    complete = json.loads((run/'complete.json').read_text(encoding='utf-8'))
    if not complete.get('completed'):
        raise ValueError('The prediction run has not completed')
    if protocol.get('source') != source or protocol.get('site') != site:
        raise ValueError('Requested source/site differs from the prediction run')
    periods = ['validation']+(['spring','summer'] if source == 'BPA' else ['autumn','winter'])
    missing = []
    for period in periods:
        for method in RAW_ARMS:
            path = run/'predictions'/period/(method.replace(' ','_')+'.npz')
            if not path.is_file():
                missing.append(f'{period}/{path.name}')
    if missing:
        raise ValueError(f'Missing required prediction packets: {missing}')
    return periods, dict(input_run_purpose=complete.get('purpose','Completed record comparison'),
        model_name=complete.get('model_name',protocol.get('recipe','supplied predictor')),
        original_training_cases=complete.get('original_training_cases'),
        training_cases=complete.get('training_cases'))


def packet_groups(run, data, source, site, periods):
    for period in periods:
        task = task_for(data, source, site, period)
        ix = task.indices(period)
        meta, truth = task.meta.iloc[ix], task.targets(ix).astype(float)
        for method in RAW_ARMS:
            path = run/'predictions'/period/(method.replace(' ','_')+'.npz')
            packet = task.read_submission(path, split=period)
            if 'prediction_kw' not in packet:
                raise ValueError(f'Power predictions are missing: {period}/{method}')
            groups = group_residuals(source, meta, packet['prediction_kw']-truth,
                                    task.base.population if source == 'ACN' else None)
            yield period, method, groups


def execute(args):
    run, data = args.run_directory.resolve(), args.data.resolve()
    periods, metadata = inspect_run(run, data, args.source, args.site)
    if not args.run:
        preview = dict(preview=True, source=args.source, site=args.site, periods=periods,
            prediction_packets=len(periods)*len(RAW_ARMS), model_imported=False,
            packet_values_read=False, output_created=False,
            note='Replay only; explicit --data replaces historical data-root paths')
        print(json.dumps(preview), flush=True)
        return preview
    if args.output is None:
        raise ValueError('--run requires --output')
    output = args.output.resolve()
    for root in (run, data):
        if output == root or root in output.parents or output in root.parents:
            raise ValueError('Keep output separate from predictions and prepared inputs')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('Use a new or empty output directory')
    started = time.perf_counter()
    result = analyze_groups(packet_groups(run,data,args.source,args.site,periods),
        source=args.source, site=args.site or 'Field study', periods=periods)
    result['receipt']['seconds'] = time.perf_counter()-started
    metadata.update(purpose='Group-level replay of a supplied completed prediction run',
                    run_directory=str(run), data_root=str(data))
    receipt = write_result(output, result, metadata)
    print(json.dumps(receipt), flush=True)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-directory', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--source', choices=['BPA','ACN'], required=True)
    parser.add_argument('--site', choices=['caltech','office001'])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--run', action='store_true')
    return execute(parser.parse_args(argv))


if __name__ == '__main__':
    main()
