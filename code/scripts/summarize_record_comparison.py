"""Write readable four-arm results from one complete portable comparison."""
import argparse
import json
import math
from pathlib import Path

FIELDS = ['power_error', 'paired_error', 'frozen_error', 'retrained_error']
CONTRASTS = ['frozen_penalty', 'retrained_penalty', 'relearning_recovery']


def cell(value):
    return str(value).replace('|', '\\|').replace('\n', ' ').replace('\r', ' ')


def render_report(receipt, effects):
    if not receipt.get('completed'):
        raise ValueError('The comparison must complete before generating its report')
    if not effects:
        raise ValueError('The comparison has no scored effects')
    site = receipt.get('site') or ('Field study' if receipt['source'] == 'BPA' else None)
    scores = {}
    for row in effects:
        if row['source'] != receipt['source'] or row['site'] != site:
            raise ValueError('Score rows belong to a different source or site')
        key = (row['scale'], row['period'], row['span_minutes'], row['metric'])
        if key in scores:
            raise ValueError('Duplicate score row')
        if row['metric'] not in ['mse', 'mae'] or row['span_minutes'] not in [0, 30, 90, 150]:
            raise ValueError('Unexpected metric or prediction span')
        if any(not isinstance(row[f], (int, float)) or not math.isfinite(row[f]) for f in FIELDS):
            raise ValueError('Nonfinite or missing comparison errors')
        scores[key] = row
    groups = sorted({(r['scale'], r['period']) for r in effects})
    for scale, period in groups:
        for span in [0, 30, 90, 150]:
            if any((scale, period, span, metric) not in scores for metric in ['mse', 'mae']):
                raise ValueError('A period/scale is missing an MSE/MAE or native-span pair')
    full = receipt.get('paper_result_eligible', False)
    title = f"# Record-information comparison: {cell(receipt['source'])} / {cell(site)}"
    lines = [title, '',
        ('Full training run: '+cell(receipt['model_name'])+'.' if receipt.get('model_name') else 'Full default training run.')
        if full else '**Training-subset smoke: pipeline check only; do not use as a full-training result.**',
        f"Training cases: {int(receipt['training_cases']):,} of {int(receipt['original_training_cases']):,}.", '',
        'Each error cell is **MSE / MAE**, in squared kW / kW. Lower is better.',
        'Fixed model uses the paired fit with reassigned records. Retrained uses',
        'the same model recipe with reassigned records at training and',
        'evaluation. Reassignment errors are averaged across three mappings;',
        'their forecasts are not ensembled. Average gives equal weight to the',
        '30-, 90- and 150-minute scores. Dates follow the source calendar.', '']
    for scale in sorted({g[0] for g in groups}):
        lines += [f'## {cell(scale)}', '',
            '| Period | Span (min) | Power only | Paired records | Fixed model, reassigned | Retrained, reassigned |',
            '|---|---:|---:|---:|---:|---:|']
        periods = sorted({p for s, p in groups if s == scale}, key=lambda p: (p != 'validation', p))
        for period in periods:
            for span in [0, 30, 90, 150]:
                mse, mae = [scores[scale, period, span, metric] for metric in ['mse', 'mae']]
                values = [f'{mse[f]:.3f} / {mae[f]:.3f}' for f in FIELDS]
                lines.append('| '+ ' | '.join([cell(period), 'Average' if span == 0 else str(span)]+values)+' |')
        lines.append('')
    lines += ['## Average-span contrasts', '',
        'Percent differences use paired-record error as the denominator.',
        'A positive fixed or retrained penalty means higher error under reassignment.',
        'Positive recovery means retraining reduces the fixed-model penalty.',
        'Intervals are 95% paired calendar-week bootstrap intervals, conditional',
        'on these fitted models. A missing interval is shown as unavailable.', '',
        '| Scale | Period | Metric | Fixed penalty, % [95% CI] | Retrained penalty, % [95% CI] | Recovery, % [95% CI] |',
        '|---|---|---|---:|---:|---:|']
    for scale, period in groups:
        for metric in ['mse', 'mae']:
            row = scores[scale, period, 0, metric]
            values = []
            for contrast in CONTRASTS:
                value, low, high = [row.get(contrast+suffix) for suffix in ['_pct', '_low', '_high']]
                if value is None:
                    values.append('unavailable')
                elif low is None or high is None:
                    values.append(f'{value:.2f} [unavailable]')
                else:
                    values.append(f'{value:.2f} [{low:.2f}, {high:.2f}]')
            lines.append('| '+' | '.join([cell(scale), cell(period), metric.upper()]+values)+' |')
    lines += ['', 'Full native-span intervals and date/week counts are in `effects.json`.',
        'The controlled reassignment does not estimate the natural frequency',
        'or financial cost of record-linkage failures.', '']
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='Completed run directory')
    parser.add_argument('--output', type=Path, help='New Markdown file; default: print to stdout')
    args = parser.parse_args(argv)
    receipt = json.loads((args.run/'complete.json').read_text(encoding='utf-8'))
    effects = json.loads((args.run/'effects.json').read_text(encoding='utf-8'))
    report = render_report(receipt, effects)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(report)
        print(json.dumps(dict(report=str(args.output), score_rows=len(effects))))
    else:
        print(report)


if __name__ == '__main__':
    main()
