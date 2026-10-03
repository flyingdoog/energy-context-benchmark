"""Read one preselected public floor; describe support without fitting models."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from field_control.cu_bems_preparation import parse_columns, complete_windows
BASE = ROOT / 'results/benchmark_source_feasibility_v1/cu_bems'
SOURCE = BASE / '2019Floor2.csv'
os.environ.setdefault('MPLCONFIGDIR', str(BASE / 'mpl_cache'))


def longest_missing(observed, clock):
    missing = ~np.asarray(observed, dtype=bool)
    edges = np.diff(np.r_[False, missing, False].astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    if not len(starts):
        return dict(minutes=0, first=None, last=None)
    which = np.argmax(ends - starts)
    a, b = int(starts[which]), int(ends[which])
    return dict(minutes=b-a, first=str(clock[a]), last=str(clock[b-1]))


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf8')


def main():
    started = time.monotonic()
    receipt = BASE / 'census.json'
    if receipt.exists():
        old = json.loads(receipt.read_text(encoding='utf8'))
        if old.get('completed'):
            print(json.dumps({'reused_completed': True, 'rows': old['rows']}))
            return
    frame = pd.read_csv(SOURCE)
    clock = pd.DatetimeIndex(pd.to_datetime(frame['Date'], format='%Y-%m-%d %H:%M:%S'))
    if clock.has_duplicates or not clock.is_monotonic_increasing:
        raise ValueError('Resolve source clock duplicates/order before defining windows')
    minute_ns = 60 * 1_000_000_000
    # Do not assume the DatetimeIndex's internal storage resolution.
    if not np.all(np.diff(clock.as_unit('ns').asi8) == minute_ns):
        raise ValueError('Resolve missing timestamp rows before defining windows')
    if str(clock[0]) != '2019-01-01 00:00:00' or str(clock[-1]) != '2019-12-31 23:59:00':
        raise ValueError('Unexpected period in the preselected 2019 file')
    layout = parse_columns(frame.columns)
    ac = [item for item in layout if item['kind'] == 'AC']
    zone_ids = sorted({item['zone'] for item in ac})
    power = {item['column']: np.isfinite(frame[item['column']].to_numpy(dtype=float))
             for item in ac}
    thermal = {}
    for zone in zone_ids:
        names = [item['column'] for item in layout if item['zone'] == zone
                 and item['kind'] == 'sensor' and item['unit'] in ('degC', 'RH%')]
        if len(names) != 2:
            raise ValueError(f'Expected one paired temperature/humidity sensor for zone {zone}')
        thermal[zone] = np.isfinite(frame[names].to_numpy(dtype=float)).all(axis=1)
    fields = []
    for item in layout:
        values = frame[item['column']].to_numpy(dtype=float)
        finite = np.isfinite(values)
        valid_values = values[finite]
        fields.append(dict(**item, observed=int(finite.sum()), missing=int((~finite).sum()),
            observed_fraction=float(finite.mean()), negative=int((valid_values < 0).sum()),
            zero=int((valid_values == 0).sum()),
            minimum=float(valid_values.min()) if len(valid_values) else None,
            maximum=float(valid_values.max()) if len(valid_values) else None,
            q50=float(np.quantile(valid_values, .50)) if len(valid_values) else None,
            q99=float(np.quantile(valid_values, .99)) if len(valid_values) else None,
            longest_missing=longest_missing(finite, clock)))
    # Descriptive support only. Matching all native spans uses the longest target.
    origins = np.arange(60, len(frame) - 150 + 1, 30)
    origin_clock = clock[origins]
    month = clock.strftime('%Y-%m').to_numpy()
    origin_month = origin_clock.strftime('%Y-%m').to_numpy()
    power_windows = {name: complete_windows(mask, origins, 60, 150)
                     for name, mask in power.items()}
    thermal_windows = {zone: complete_windows(mask, origins, 60, 0)
                       for zone, mask in thermal.items()}
    daily = pd.DataFrame({
        **{f"z{item['zone']} AC{item['instrument'][2:]}": power[item['column']] for item in ac},
        **{f'z{zone} temperature + RH': thermal[zone] for zone in zone_ids}
    }, index=clock).resample('D').mean()
    monthly = []
    support = []
    cohorts = [('Floor 2', ac)]
    cohorts += [(f'Zone {zone}', [x for x in ac if x['zone'] == zone])
                for zone in zone_ids if sum(x['zone'] == zone for x in ac) >= 2]
    for period in sorted(set(month)):
        minute_sel = month == period
        origin_sel = origin_month == period
        row = dict(month=period, minutes=int(minute_sel.sum()), origins=int(origin_sel.sum()))
        for zone in zone_ids:
            zone_ac = [item for item in ac if item['zone'] == zone]
            row[f'z{zone}_ac_count'] = len(zone_ac)
            row[f'z{zone}_power_fraction'] = float(np.mean(
                [power[item['column']][minute_sel].mean() for item in zone_ac]))
            row[f'z{zone}_thermal_fraction'] = float(thermal[zone][minute_sel].mean())
        monthly.append(row)
        for item in ac:
            pw = power_windows[item['column']] & origin_sel
            coupled = pw & thermal_windows[item['zone']]
            support.append(dict(month=period, level='device', entity=item['column'],
                members=1, eligible_origins=int(origin_sel.sum()),
                power_windows=int(pw.sum()), linked_windows=int(coupled.sum()),
                linked_dates=int(len(set(origin_clock[coupled].date)))))
        for name, members in cohorts:
            pw = np.logical_and.reduce([power_windows[x['column']] for x in members]) & origin_sel
            coupled = pw & np.logical_and.reduce([thermal_windows[z] for z in {x['zone'] for x in members}])
            support.append(dict(month=period, level='group', entity=name, members=len(members),
                eligible_origins=int(origin_sel.sum()), power_windows=int(pw.sum()),
                linked_windows=int(coupled.sum()), linked_dates=int(len(set(origin_clock[coupled].date)))))
    dump(BASE / 'field_summary.json', fields)
    dump(BASE / 'monthly_coverage.json', monthly)
    dump(BASE / 'window_support.json', support)
    dump(BASE / 'daily_coverage.json', [dict(date=str(date.date()), **row.to_dict()) for date, row in daily.iterrows()])
    try:
        import matplotlib
    except ModuleNotFoundError as exc:
        if exc.name != 'matplotlib':
            raise
        # Bundled Python has the analysis libraries but no plot library.
        # Reuse the project's existing plotting dependency for this missing
        # capability only; no installation or global environment change.
        sys.path.insert(0, str(ROOT / '.deps'))
        import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(11.4, 6.3), sharex=True,
        gridspec_kw={'height_ratios': [16, 4]}, constrained_layout=True)
    panels = [daily.iloc[:, :len(ac)], daily.iloc[:, len(ac):]]
    for axis, panel, title in zip(axes, panels, ['Individual AC power', 'Paired zone temperature and humidity']):
        im = axis.imshow(panel.to_numpy().T * 100, aspect='auto', interpolation='nearest',
                         cmap='Blues', vmin=0, vmax=100)
        axis.set_yticks(range(len(panel.columns)), panel.columns, fontsize=8)
        axis.set_title(title, fontsize=11, loc='left')
        for spine in axis.spines.values():
            spine.set_visible(False)
    ticks = np.flatnonzero(daily.index.day == 1)
    axes[-1].set_xticks(ticks, daily.index[ticks].strftime('%b'))
    axes[-1].set_xlabel('2019 source calendar (timezone not encoded)')
    fig.colorbar(im, ax=axes, shrink=.7, label='Observed minutes per day (%)')
    fig.suptitle('CU-BEMS Floor 2: recorded power and thermal-context availability', fontsize=13)
    fig.savefig(BASE / 'availability.png', dpi=160)
    plt.close(fig)
    lines = ['# CU-BEMS Floor 2: actual 2019 support', '',
        'Original measurements are unchanged. No model is fitted. Coverage is finite observations, not a claim of sensor accuracy.', '',
        '| Month | AC power coverage | Zone 1 thermal | Zone 2 thermal | Zone 4 thermal | Device windows: power / linked | Floor windows: power / linked | Linked floor dates |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in monthly:
        rows = [x for x in support if x['month'] == row['month']]
        devices = [x for x in rows if x['level'] == 'device']
        floor = next(x for x in rows if x['entity'] == 'Floor 2')
        avg_power = sum(row[f'z{z}_power_fraction'] * row[f'z{z}_ac_count'] for z in zone_ids) / len(ac)
        lines.append(f"| {row['month']} | {100*avg_power:.1f}% | " +
            ' | '.join(f"{100*row[f'z{z}_thermal_fraction']:.1f}%" for z in zone_ids) +
            f" | {sum(x['power_windows'] for x in devices):,} / {sum(x['linked_windows'] for x in devices):,} | " +
            f"{floor['power_windows']:,} / {floor['linked_windows']:,} | {floor['linked_dates']} |")
    lines += ['', 'Support counts use 30-minute-spaced queries, a complete 60-minute power history and 150-minute future power record.',
        'Linked additionally requires temperature and humidity throughout the 60-minute history, never future sensors.',
        'All sixteen ACs remain in the floor group; missing future members are not dropped. Zone 2 has a separate fourteen-AC group.',
        'Monthly counts describe source coverage, not independent trials or approved train/test splits.', '',
        '[Daily availability](availability.png) shows missingness rather than repeating the window table.',
        'Source: [CU-BEMS publisher release, version 6](https://doi.org/10.6084/m9.figshare.11726517), file 2019Floor2.csv.',
        'The publisher metadata names CC BY 4.0. Naive source timestamps are preserved; UTC/arrival-time semantics are not inferred.', '']
    (BASE / 'support.md').write_text('\n'.join(lines), encoding='utf8')
    metadata = json.loads((BASE / 'publisher_metadata.json').read_text(encoding='utf8'))
    result = dict(completed=True, seconds=time.monotonic()-started, source=SOURCE.name,
        bytes=SOURCE.stat().st_size, publisher_version=metadata['version'], license=metadata['license'],
        rows=len(frame), measurements=len(layout), ac_devices=len(ac), zones_with_ac=zone_ids,
        first=str(clock[0]), last=str(clock[-1]), duplicate_timestamps=0, missing_timestamp_rows=0,
        clock='Source naive timestamps; minute grid verified; timezone and telemetry arrival not encoded',
        window_definition={'history_minutes': 60, 'future_minutes': 150, 'query_stride_minutes': 30,
            'range': 'history [q-60,q), future [q,q+150); support census, not a trained benchmark'},
        total_device_power_windows=sum(x['power_windows'] for x in support if x['level']=='device'),
        total_device_linked_windows=sum(x['linked_windows'] for x in support if x['level']=='device'),
        negative_power_observations=sum(x['negative'] for x in fields if x['unit']=='kW'),
        models_fitted=0, gpu_used=False, raw_modified=False, figure_visually_checked=False)
    dump(receipt, result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
