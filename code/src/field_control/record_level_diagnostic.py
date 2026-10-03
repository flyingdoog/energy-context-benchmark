"""Replay the fixed group-level control on supplied four-arm predictions."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .group_error_level import (LEADS, equal_date_mean, lead_decomposition,
    daily_errors, span_decomposition, paired_week_difference)
from .record_reference_scores import aggregate_errors, paired_effects, RAW_ARMS, METRICS

SCALES = ('cohort_mean', 'cohort_total')
FIELDS = ('power_error', 'paired_error', 'frozen_error', 'retrained_error')


def average_mappings(frame, keys, metrics):
    frame = frame.copy()
    frame['method'] = frame.method.str.replace(r' (2027|2028|2029)$', '', regex=True)
    return frame.groupby(keys+['method'], as_index=False)[metrics].mean()


def contrast_table(daily, draws=1500):
    mean = daily.groupby(['source','site','period','scale','date','method','calibration'], as_index=False)[METRICS].mean()
    mean['span_minutes'] = 0
    values = pd.concat([daily, mean], ignore_index=True)
    result = []
    group = ['source','site','period','scale','span_minutes']
    for key, part in values.groupby(group):
        for metric in ['mse', 'mae']:
            p = part.pivot(index='date', columns=['calibration','method'], values=metric).sort_index()
            if p.isna().any().any():
                raise ValueError('Calibration contrast has different date support')
            for arm in ['Fixed', 'Retrained']:
                uncorrected = p['raw', arm]-p['raw', 'Paired']
                corrected = p['validation_offset', arm]-p['validation_offset', 'Paired']
                for name, first, second in [
                    ('raw', p['raw', arm], p['raw', 'Paired']),
                    ('validation_offset', p['validation_offset', arm], p['validation_offset', 'Paired']),
                    ('change_in_penalty', corrected, uncorrected)]:
                    interval = paired_week_difference(p.index, first, second, draws=draws)
                    result.append(dict(zip(group, key), metric=metric, method=arm,
                                       calibration=name, dates=len(p), **interval))
    return pd.DataFrame(result)


def _validated_groups(groups):
    if set(groups) != set(SCALES):
        raise ValueError('Each packet needs both native group-mean and group-total residuals')
    result = {}
    for scale, original in groups.items():
        frame = original.copy()
        if frame.empty or not frame.columns.is_unique or not set(LEADS+['date']) <= set(frame):
            raise ValueError('Group residuals need identities, date and ten finite lead columns')
        identity = [column for column in frame if column not in LEADS]
        if len(identity) < 2 or frame[identity].isna().any().any():
            raise ValueError('Group identities and dates must be present')
        frame['date'] = pd.to_datetime(frame.date, errors='raise').dt.strftime('%Y-%m-%d')
        if frame.duplicated(identity).any():
            raise ValueError('Repeated group identity')
        if not np.isfinite(frame[LEADS].to_numpy(dtype=float)).all():
            raise ValueError('Group residuals must be finite')
        frame = frame.sort_values(identity).reset_index(drop=True)
        result[scale] = frame
    first, second = [result[scale].drop(columns=LEADS) for scale in SCALES]
    if not first.equals(second):
        raise ValueError('Group scales use different memberships')
    return result


def analyze_groups(stream, *, source, site, periods, draws=1500):
    """Analyze (period, raw-arm, group-residual-frames) records, without models.

    Frames come from group_error_level.group_residuals. All eight mapping
    streams are required. Validation is calibrated first, regardless of input
    stream order; later-period labels only score the unchanged offsets.
    """
    periods = tuple(periods)
    if len(periods) < 2 or periods[0] != 'validation' or len(set(periods)) != len(periods):
        raise ValueError('Name validation followed by each later period exactly once')
    if not isinstance(draws, int) or draws < 1:
        raise ValueError('Bootstrap draws must be a positive integer')
    packets, identities = {}, {}
    for period, method, groups in stream:
        key = (period, method)
        if period not in periods or method not in RAW_ARMS or key in packets:
            raise ValueError(f'Unexpected or duplicate group packet: {key}')
        groups = _validated_groups(groups)
        for scale, frame in groups.items():
            identity = frame.drop(columns=LEADS)
            reference = identities.setdefault((period, scale), identity)
            if not identity.equals(reference):
                raise ValueError(f'Arms use different group memberships in {period}/{scale}')
        packets[key] = groups
    missing = [(p, a) for p in periods for a in RAW_ARMS if (p, a) not in packets]
    if missing:
        raise ValueError(f'Missing required prediction packets (including validation): {missing}')
    days, leads, spans, offsets = [], [], [], []
    validation_offsets = {}
    for period in periods:
        for method in RAW_ARMS:
            for scale, frame in packets[period, method].items():
                info = dict(source=source, site=site, period=period, scale=scale, method=method)
                if period == 'validation':
                    validation_offsets[method, scale] = equal_date_mean(frame)
                    offsets.append(dict(**info, intercept_kw=validation_offsets[method, scale].tolist(),
                                        dates=int(frame.date.nunique()), groups=len(frame)))
                for control, offset in [('raw', None), ('validation_offset', validation_offsets[method, scale])]:
                    days.append(daily_errors(frame, offset).assign(**info, calibration=control))
                    decomposition = lead_decomposition(frame, offset)
                    leads.append(decomposition.assign(**info, calibration=control))
                    spans.append(span_decomposition(decomposition).assign(**info, calibration=control))
    raw = pd.concat(days, ignore_index=True)
    lead_keys = ['source','site','period','scale','calibration','lead']
    metrics = ['mse','mae','bias_mse','centered_mse']
    lead_values = average_mappings(pd.concat(leads, ignore_index=True), lead_keys, metrics)
    span_keys = ['source','site','period','scale','calibration','span_minutes']
    span_values = average_mappings(pd.concat(spans, ignore_index=True), span_keys, metrics)
    averages, effects = [], []
    for control, part in raw.groupby('calibration'):
        daily = aggregate_errors(part)
        averages.append(daily.assign(calibration=control))
        effects.append(paired_effects(daily, draws=draws).assign(calibration=control))
    daily = pd.concat(averages, ignore_index=True)
    tables = dict(raw_daily=raw, daily=daily, lead_decomposition=lead_values,
                  decomposition=span_values, effects=pd.concat(effects, ignore_index=True),
                  penalty_contrasts=contrast_table(daily, draws=draws))
    receipt = dict(source=source, site=site, periods=list(periods), prediction_packets=len(packets),
        validation_offsets=len(offsets), raw_daily_rows=len(raw), averaged_daily_rows=len(daily),
        lead_decomposition_rows=len(lead_values), span_decomposition_rows=len(span_values),
        effect_rows=len(tables['effects']), paired_contrasts=len(tables['penalty_contrasts']),
        calibration='Ten validation-only intercepts per arm/mapping and group scale; no search',
        uncertainty='Pointwise calendar-week intervals conditional on trained models and validation offsets',
        bootstrap_draws=draws, model_fits=0, inference_calls=0, gpu_used=False)
    return dict(tables=tables, validation_offsets=offsets, receipt=receipt)


def render_report(result, metadata=None):
    receipt, tables = result['receipt'], result['tables']
    lines = ['# Group-level control on a supplied prediction run', '',
        (metadata or {}).get('purpose', 'Replay of saved predictions; no model fitting or inference.'), '',
        f'Source/site: {receipt["source"]} / {receipt["site"]}.',
        'Cells show MSE / MAE, in squared kW / kW. Original retains the submitted',
        'predictions. Offset subtracts a validation-fitted signed mean per lead,',
        'group scale and arm/mapping. The same offsets are transferred unchanged',
        'to every later period; no clipping or parameter search is used.', '',
        'This is a group-output control, not a corrected device submission.',
        'Offset validation results are calibration in-sample readouts.', '',
        '## Group-total summary', '',
        '| Period | Control | Power only | Paired | Fixed | Retrained |',
        '|---|---|---:|---:|---:|---:|']
    effects = tables['effects']

    def metric_cells(period, scale, span, control):
        selected = effects[(effects.period == period)&(effects.scale == scale)&
            (effects.span_minutes == span)&(effects.calibration == control)].set_index('metric')
        return [f'{selected.loc["mse",field]:.3f} / {selected.loc["mae",field]:.3f}' for field in FIELDS]

    for period in receipt['periods']:
        for control in ('raw','validation_offset'):
            name = 'Original' if control == 'raw' else 'Offset'
            lines.append('| '+' | '.join([period,name]+metric_cells(period,'cohort_total',0,control))+' |')
    lines += ['', 'Summary errors equally average the three native spans after the original',
        'date-balanced scoring. Reassignment averages mapping errors, not forecasts.', '',
        '## All native spans and group scales', '',
        '| Period | Scale | Span (min) | Control | Power only | Paired | Fixed | Retrained |',
        '|---|---|---:|---|---:|---:|---:|---:|']
    for period in receipt['periods']:
        for scale in SCALES:
            for span in (30,90,150):
                for control in ('raw','validation_offset'):
                    name = 'Original' if control == 'raw' else 'Offset'
                    lines.append('| '+' | '.join([period,scale.replace('cohort_',''),str(span),name]+
                        metric_cells(period,scale,span,control))+' |')
    lines += ['', '## Group-total penalty and its components', '',
        'Original MSE differences are decomposed by lead and mapping before averaging.',
        'Centered group-error variation is distinct from centered device covariance.', '',
        '| Period | Arm | MSE penalty | Squared-mean contribution | Centered-variation contribution |',
        '|---|---|---:|---:|---:|']
    values = tables['decomposition']
    for period in receipt['periods']:
        part = values[(values.period == period)&(values.scale == 'cohort_total')&
            (values.span_minutes == 0)&(values.calibration == 'raw')].set_index('method')
        for arm in ('Fixed','Retrained'):
            diff = part.loc[arm,['mse','bias_mse','centered_mse']]-part.loc['Paired',['mse','bias_mse','centered_mse']]
            lines.append('| '+' | '.join([period,arm]+[f'{v:.3f}' for v in diff])+' |')
    lines += ['', '## Later group-total penalties after correction', '',
        '| Period | Metric | Arm | Original penalty [95% CI] | Offset penalty [95% CI] |',
        '|---|---|---|---:|---:|']
    contrasts = tables['penalty_contrasts']
    for period in receipt['periods'][1:]:
        for metric in ('mse','mae'):
            for arm in ('Fixed','Retrained'):
                rows = contrasts[(contrasts.period == period)&(contrasts.scale == 'cohort_total')&
                    (contrasts.span_minutes == 0)&(contrasts.metric == metric)&(contrasts.method == arm)].set_index('calibration')
                intervals = []
                for control in ('raw','validation_offset'):
                    row = rows.loc[control]
                    ci = 'unavailable' if pd.isna(row.low) else f'{row.low:.3f}, {row.high:.3f}'
                    intervals.append(f'{row.difference:.3f} [{ci}]')
                lines.append('| '+' | '.join([period,metric.upper(),arm]+intervals)+' |')
    lines += ['', 'Intervals are pointwise paired calendar-week intervals conditional on',
        'the fitted predictors and validation offsets; they omit calibration-estimation',
        'uncertainty. Penalty changes compare each arm with its paired reference',
        'under the same control, not just the arm\'s absolute improvement.', '',
        'Structured outputs retain all dates, spans, scales, offsets and contrasts.', '']
    return '\n'.join(lines)


def write_result(output, result, metadata=None):
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('Use a new or empty output directory; prior outputs are preserved')
    output.mkdir(parents=True, exist_ok=True)
    for name, table in result['tables'].items():
        table.to_json(output/f'{name}.json', orient='records', indent=2, double_precision=12)
    (output/'validation_offsets.json').write_text(json.dumps(result['validation_offsets'],indent=2)+'\n', encoding='utf-8')
    (output/'comparison.md').write_text(render_report(result, metadata), encoding='utf-8')
    receipt = dict(result['receipt'], completed=True, run_information=metadata or {})
    (output/'complete.json').write_text(json.dumps(receipt,indent=2)+'\n', encoding='utf-8')
    return receipt
