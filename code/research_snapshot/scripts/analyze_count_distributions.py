"""Paired temporal intervals and coverage/width view of saved count forecasts."""
import json
from pathlib import Path
import time
import run_local
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from evaluate_count_distributions import OUT as BASE, METRICS, MARGINALS, JOINTS, save_json, save_frame

OUT = BASE / 'analysis'
PERIODS = [('BPA', 'Field study', 'spring', 'BPA spring'),
           ('BPA', 'Field study', 'summer', 'BPA summer'),
           ('ACN', 'Caltech', 'autumn', 'Caltech autumn'),
           ('ACN', 'Caltech', 'winter', 'Caltech winter'),
           ('ACN', 'Office', 'autumn', 'Office autumn'),
           ('ACN', 'Office', 'winter', 'Office winter')]


def calendar_weights(dates, draws=1500):
    stamps = pd.DatetimeIndex(pd.to_datetime(dates))
    weeks = (stamps - pd.to_timedelta(stamps.weekday, unit='D')).strftime('%Y-%m-%d')
    unique = sorted(set(weeks))
    if len(unique) < 2:
        raise ValueError('At least two calendar-week clusters required')
    index = np.array([unique.index(w) for w in weeks])
    counts = np.random.default_rng(2027).multinomial(len(unique), np.ones(len(unique)) / len(unique), size=draws)
    weights = counts[:, index].astype(float)
    weights /= weights.sum(axis=1, keepdims=True)
    return weights, len(unique)


def paired_statistics(a, b, weights):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or weights.shape[1] != len(a) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Aligned finite paired dates required')
    boot_a, boot_b = weights @ a, weights @ b
    low, high = np.quantile(boot_b - boot_a, [.025, .975])
    record = dict(reference=float(a.mean()), candidate=float(b.mean()),
                  difference=float(b.mean() - a.mean()), difference_low=float(low), difference_high=float(high))
    if (boot_a > 0).all() and a.mean() > 0:
        low, high = np.quantile(100 * (boot_b / boot_a - 1), [.025, .975])
        record.update(change_pct=float(100 * (b.mean() / a.mean() - 1)),
                      change_low=float(low), change_high=float(high))
    return record


def main():
    start = time.perf_counter()
    receipt = json.loads((BASE / 'receipt.json').read_text())
    if not receipt['completed'] or len(receipt['periods']) != 6:
        raise ValueError('Complete the prescribed six-period evaluation first')
    OUT.mkdir(parents=True, exist_ok=True)
    daily = pd.read_json(BASE / 'daily_scores.json')
    daily['date'] = pd.to_datetime(daily.date).dt.strftime('%Y-%m-%d')
    native = pd.read_json(BASE / 'native_span_scores.json')
    average = pd.read_json(BASE / 'span_average.json')
    full = daily.groupby(['source', 'site', 'period', 'marginal', 'joint', 'date'], as_index=False)[METRICS].mean()
    full['span_minutes'] = 0
    values = pd.concat([daily, full], ignore_index=True)
    intervals = []; cache = {}
    keys = ['source', 'site', 'period', 'marginal', 'span_minutes']
    for key, frame in values.groupby(keys):
        for metric in METRICS:
            table = frame.pivot(index='date', columns='joint', values=metric).sort_index()
            if table.isna().any().any() or set(table.columns) != set(JOINTS):
                raise ValueError('Joint forecasts use different temporal support')
            dates = tuple(table.index)
            if dates not in cache: cache[dates] = calendar_weights(dates)
            weights, weeks = cache[dates]
            for candidate in JOINTS[1:]:
                intervals.append(dict(zip(keys, key), metric=metric, joint=candidate,
                    dates=len(dates), weeks=weeks,
                    **paired_statistics(table.Independent, table[candidate], weights)))
    intervals = pd.DataFrame(intervals)
    save_frame(OUT / 'paired_intervals.json', intervals)

    # A fixed cross-reference comparison avoids interpreting a weak-marginal
    # calibration benefit as superiority over a stronger simple predictor.
    comparison = []
    for source, site, period, _ in PERIODS:
        p = full[(full.source == source) & (full.site == site) & (full.period == period)]
        table = p.pivot(index='date', columns=['marginal', 'joint'], values='normalized_rps').sort_index()
        weights, weeks = cache[tuple(table.index)]
        comparison.append(dict(source=source, site=site, period=period,
            reference_method='State frequency / Independent', candidate_method='Two-part / Calibrated mixture',
            metric='normalized_rps', weeks=weeks, dates=len(table),
            **paired_statistics(table[('State frequency', 'Independent')], table[('Two-part', 'Calibrated mixture')], weights)))
    save_json(OUT / 'cross_marginal_reference.json', comparison)

    # All joint means must agree, while distribution scores need not agree.
    for _, frame in native.groupby(['source', 'site', 'period', 'marginal', 'span_minutes']):
        if len(frame) != 3 or not np.allclose(frame.count_mse, frame.count_mse.iloc[0], rtol=1e-10, atol=1e-10):
            raise ValueError('Expected-count MSE changed across joint distributions')
    selected = intervals[(intervals.metric == 'normalized_rps') & (intervals.span_minutes == 0)
                          & (intervals.joint == 'Calibrated mixture')]
    lines = ['# Count-distribution forecasts on all original later periods', '',
        'Models retain their original device marginals. Six scalar mixtures use complete original validation blocks.',
        'nRPS divides ranked probability score by group size. Lower is better. Intervals are paired calendar-week',
        'resamples conditional on fitted marginal models and calibration coefficients; they are not training variability.', '',
        '| Period | Marginal predictor | Independent nRPS | Mixture nRPS | Change % [95% CI] | Same expected-count MSE | Dates / weeks |',
        '|---|---|---:|---:|---:|---:|---:|']
    for source, site, period, label in PERIODS:
        for marginal in MARGINALS:
            row = selected[(selected.source == source) & (selected.site == site) &
                (selected.period == period) & (selected.marginal == marginal)].iloc[0]
            values = average[(average.source == source) & (average.site == site) &
                (average.period == period) & (average.marginal == marginal)].set_index('joint')
            change = f'{row.change_pct:+.2f} [{row.change_low:+.2f}, {row.change_high:+.2f}]'
            lines.append(f'| {label} | {marginal} | {row.reference:.3f} | {row.candidate:.3f} | {change} | '
                         f'{values.loc["Independent", "count_mse"]:.3f} | {int(row.dates)} / {int(row.weeks)} |')
    lines += ['', '## All native spans and endpoints', '',
        '| Period | Marginal predictor | Span (min) | Independent nRPS / RPS | Mixture nRPS / RPS | Shared-uniform nRPS / RPS |',
        '|---|---|---:|---:|---:|---:|']
    for source, site, period, label in PERIODS:
        for marginal in MARGINALS:
            for span in (30, 90, 150):
                p = native[(native.source == source) & (native.site == site) & (native.period == period)
                    & (native.marginal == marginal) & (native.span_minutes == span)].set_index('joint')
                cells = [f'{p.loc[j, "normalized_rps"]:.3f} / {p.loc[j, "rps"]:.3f}'
                         for j in ('Independent', 'Calibrated mixture', 'Shared uniform')]
                lines.append('| ' + ' | '.join([label, marginal, str(span)] + cells) + ' |')
    lines += ['', 'Coverage and width are shown separately in coverage_width.png; full80%/90% results are in JSON.',
        'Expected counts are unchanged by construction. This is per-bin count forecasting, not a joint temporal path.',
        'The six periods are from two source families/three site contexts, not six independent datasets.']
    (OUT / 'comparison.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    colors = {'State frequency': '#287e89', 'Two-part': '#bc7137'}
    markers = {'Independent': 'o', 'Shared uniform': 's', 'Calibrated mixture': '^'}
    fig, axes = plt.subplots(2, 3, figsize=(11.8, 6.5), sharex=True, sharey=True)
    for ax, (source, site, period, label) in zip(axes.flat, PERIODS):
        p = average[(average.source == source) & (average.site == site) & (average.period == period)]
        for marginal in MARGINALS:
            part = p[p.marginal == marginal].set_index('joint')
            for joint in JOINTS:
                row = part.loc[joint]
                ax.scatter(row.fraction_width_90, 100 * row.coverage_90, marker=markers[joint],
                    color=colors[marginal], s=48 if joint != 'Calibrated mixture' else 66,
                    facecolors='none' if joint == 'Shared uniform' else colors[marginal],
                    edgecolors='white' if joint == 'Calibrated mixture' else colors[marginal], linewidths=.9,
                    zorder=4 if joint == 'Calibrated mixture' else 3)
            a, b = part.loc['Independent'], part.loc['Calibrated mixture']
            ax.plot([a.fraction_width_90, b.fraction_width_90], [100*a.coverage_90, 100*b.coverage_90],
                    color=colors[marginal], lw=.9, alpha=.8)
        ax.axhline(90, color='#777777', linestyle='--', linewidth=.8)
        ax.set_title(label, fontsize=11)
        ax.set_xlim(0, .82); ax.set_ylim(60, 102)
        ax.grid(alpha=.15); ax.spines[['top', 'right']].set_visible(False)
    for ax in axes[:, 0]: ax.set_ylabel('Observed coverage (%)', fontsize=10)
    for ax in axes[-1]: ax.set_xlabel('Interval width / group size', fontsize=10)
    handles = [Line2D([], [], color=colors[m], lw=3, label=m) for m in MARGINALS]
    handles += [Line2D([], [], color='#555555', linestyle='none', marker=markers[j],
                      markerfacecolor='none' if j == 'Shared uniform' else '#555555', label=j) for j in JOINTS]
    fig.legend(handles=handles, loc='upper center', ncol=5, frameon=False, fontsize=9)
    fig.suptitle('90% count intervals: empirical coverage and width', y=.946, fontsize=12)
    fig.text(.5, .016, 'Date- and span-averaged scores. Dashed line: nominal coverage. Coincident points overlap.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .04, 1, .92])
    fig.savefig(OUT / 'coverage_width.png', dpi=160)
    plt.close(fig)
    save_json(OUT / 'receipt.json', dict(completed=True, seconds=time.perf_counter() - start,
        interval_rows=len(intervals), cross_marginal_contrasts=len(comparison),
        bootstrap_draws=1500, native_rows=len(native), count_mean_identity=True,
        new_fits=0, figure='coverage_width.png', figure_visually_checked=False))
    print('\n'.join(lines[:20]), flush=True)
    print(json.dumps({'analysis_complete': True, 'interval_rows': len(intervals), 'seconds': time.perf_counter() - start}), flush=True)


if __name__ == '__main__': main()
