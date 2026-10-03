"""Prepare the fixed CU-BEMS 2019 task from a publisher floor CSV.

Only numpy/pandas and the model-neutral scorer are needed. No experiment
receipts, saved predictions or fitted model libraries are read.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd

from .cu_bems_scoring import make_groups

SPLITS = [('train', '2019-04-01', '2019-07-01'),
          ('validation', '2019-07-01', '2019-09-01'),
          ('later_1', '2019-09-01', '2019-11-01'),
          ('later_2', '2019-11-01', '2020-01-01')]


def parse_columns(columns):
    result = []
    for name in columns:
        if name == 'Date':
            continue
        match = re.fullmatch(r'z(\d+)_(AC\d+|Light|Plug|S\d+)\(([^)]+)\)', name)
        if match is None:
            raise ValueError(f'Unknown measurement header: {name}')
        zone, instrument, unit = match.groups()
        kind = 'AC' if instrument.startswith('AC') else (
            'sensor' if instrument.startswith('S') else instrument)
        result.append(dict(column=name, zone=int(zone), instrument=instrument,
                           kind=kind, unit=unit))
    return result


def complete_windows(observed, origins, before, after):
    """All observations on [q-before, q+after); q is the future start."""
    observed = np.asarray(observed, dtype=bool)
    origins = np.asarray(origins, dtype=int)
    if (observed.ndim != 1 or np.any(origins < before)
            or np.any(origins + after > len(observed))):
        raise ValueError('Window outside the observed time grid')
    missing = np.r_[0, np.cumsum(~observed, dtype=np.int64)]
    return missing[origins + after] == missing[origins - before]


def split_queries(clock, start, end):
    q = np.arange(60, len(clock) - 150 + 1, 30)
    stamps = clock[q]
    return q[(stamps >= pd.Timestamp(start)) &
             (stamps + pd.Timedelta(minutes=150) <= pd.Timestamp(end))]


def sensor_columns(layout, zone):
    selected = [x for x in layout if x['zone'] == zone and x['kind'] == 'sensor'
                and x['unit'] in ('degC', 'RH%')]
    if not selected:
        return None
    if (len(selected) != 2 or {x['unit'] for x in selected} != {'degC', 'RH%'}
            or len({x['instrument'] for x in selected}) != 1):
        raise ValueError(f'Inspect ambiguous/incomplete thermal sensor fields in zone {zone}')
    return [next(x['column'] for x in selected if x['unit'] == unit)
            for unit in ('degC', 'RH%')]


def source_grid(frame):
    """Recorded rows on the year clock; unrecorded minutes remain NaN."""
    if frame.empty or 'Date' not in frame.columns or frame.columns.has_duplicates:
        raise ValueError('A nonempty publisher CSV with unique headers and Date is required')
    parsed = pd.to_datetime(frame['Date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    bad = np.flatnonzero(parsed.isna())
    malformed = []
    if len(bad):
        # The publisher's Floor 6 file has this one known truncated final row.
        if (len(bad) != 1 or bad[0] != len(frame) - 1
                or str(frame.iloc[bad[0]]['Date']) != '2'
                or not frame.iloc[bad[0]].drop('Date').isna().all()):
            raise ValueError('Unresolved malformed timestamp records')
        malformed = [dict(csv_line=int(bad[0]) + 2, date_text='2',
                          measurement_values_present=False)]
    valid = frame.loc[parsed.notna()].drop(columns='Date')
    valid.index = pd.DatetimeIndex(parsed[parsed.notna()])
    if valid.empty or valid.index.has_duplicates or not valid.index.is_monotonic_increasing:
        raise ValueError('Unresolved source clock order/duplicates or empty records')
    clock = pd.date_range('2019-01-01', '2019-12-31 23:59', freq='min')
    if not valid.index.isin(clock).all():
        raise ValueError('Source timestamps outside the 2019 minute calendar')
    quality = dict(raw_rows=len(frame), valid_timestamp_rows=len(valid),
        malformed_rows=malformed, duplicate_timestamps=0,
        missing_timestamp_rows=int(len(clock) - len(valid)),
        first_recorded=str(valid.index[0]), last_recorded=str(valid.index[-1]),
        analysis_grid_rows=len(clock), missing_minutes_filled_with='NaN in memory only')
    return valid.reindex(clock), clock, quality


def cohorts(devices):
    groups = [('floor', list(range(len(devices))))]
    for zone in sorted({x['zone'] for x in devices}):
        members = [i for i, x in enumerate(devices) if x['zone'] == zone]
        if len(members) >= 2 and members != groups[0][1]:
            groups.append((f'zone{zone}', members))
    return groups


def prepare_cu_bems(source, output, *, floor):
    """Write a new packet without overwriting original data or existing outputs.

    ``floor`` is explicit because the original column names encode zones but
    not floor identity. A publisher-style filename must agree when present.
    All fixed splits are constructed; missing sensor observations stay masked.
    """
    if isinstance(floor, bool) or not isinstance(floor, (int, np.integer)) or floor not in range(1, 8):
        raise ValueError('CU-BEMS floor must be an integer from 1 to 7')
    floor = int(floor)
    source, output = Path(source), Path(output)
    if not source.is_file():
        raise FileNotFoundError(source)
    named_floor = re.fullmatch(r'2019Floor([1-7])\.csv', source.name, flags=re.IGNORECASE)
    if named_floor and int(named_floor[1]) != floor:
        raise ValueError('The source filename and requested floor differ')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError('Use a new or empty output directory; existing packets are preserved')
    started = time.monotonic()
    frame = pd.read_csv(source)
    layout = parse_columns(frame.columns)
    devices = [x for x in layout if x['kind'] == 'AC']
    if not devices or any(x['unit'] != 'kW' for x in devices):
        raise ValueError('The task requires original AC power columns in kW')
    mapping = {z: sensor_columns(layout, z) for z in sorted({x['zone'] for x in devices})}
    present = [mapping[x['zone']] is not None for x in devices]
    if any(present) and not all(present):
        raise ValueError('Inspect partially absent per-floor thermal fields before preparing this source')
    has_thermal = all(present)
    raw, clock, quality = source_grid(frame)
    if quality['malformed_rows'] and floor != 6:
        raise ValueError('The known truncated final record is documented only for Floor 6')
    names = ['power', 'calendar', 'target'] + (['thermal'] if has_thermal else [])
    arrays = {name: [] for name in names}
    rows, training = [], []
    queries = [(split, split_queries(clock, begin, end)) for split, begin, end in SPLITS]
    for device_id, item in enumerate(devices):
        power = raw[item['column']].to_numpy(dtype=np.float32)
        sensor = (raw[mapping[item['zone']]].to_numpy(dtype=np.float32)
                  if has_thermal else None)
        for split, candidates in queries:
            q = candidates[complete_windows(np.isfinite(power), candidates, 60, 150)]
            if not len(q):
                continue
            history = q[:, None] + np.arange(-60, 0)
            future = q[:, None] + np.arange(150)
            p = power[history]
            y = power[future].reshape(-1, 10, 15).mean(axis=2, dtype=np.float64).astype(np.float32)
            stamps = clock[q]
            hours = stamps.hour.to_numpy() + stamps.minute.to_numpy() / 60
            week = stamps.dayofweek.to_numpy()
            calendar = np.column_stack([np.sin(2*np.pi*hours/24), np.cos(2*np.pi*hours/24),
                np.sin(2*np.pi*week/7), np.cos(2*np.pi*week/7)]).astype(np.float32)
            for name, value in [('power', p), ('target', y), ('calendar', calendar)]:
                arrays[name].append(value)
            if has_thermal:
                arrays['thermal'].append(sensor[history])
            rows.append(pd.DataFrame(dict(case_id=[f'cu_f{floor}_{device_id}_{i}' for i in q],
                floor=floor, device_index=device_id, device=item['column'], zone=item['zone'],
                split=split, query=stamps.strftime('%Y-%m-%d %H:%M:%S'),
                date=stamps.strftime('%Y-%m-%d'))))
            if split == 'train':
                training.append(dict(device=item['column'], cases=len(q), mean_power=float(p.mean()),
                    history_zero_fraction=float((p == 0).mean()), history_standard_deviation=float(p.std()),
                    thermal_observed_fraction=float(np.isfinite(sensor[history]).mean()) if has_thermal else None))
    if len(training) != len(devices):
        raise ValueError('Every source AC requires at least one eligible training case')
    meta = pd.concat(rows, ignore_index=True)
    if meta.case_id.duplicated().any() or meta.duplicated(['query', 'device_index']).any():
        raise ValueError('Repeated device cases')
    definitions = cohorts(devices)
    groups = make_groups(meta, definitions)
    support = meta.groupby('split').agg(cases=('case_id', 'size'), dates=('date', 'nunique'),
                                         devices=('device_index', 'nunique')).reset_index()
    group_support = [dict(split=s, group=g, windows=len(ids), members=ids.shape[1], dates=gmeta.date.nunique())
                     for s, g, gmeta, ids in groups]
    output.mkdir(parents=True, exist_ok=True)
    meta.to_json(output / 'cases.jsonl.gz', orient='records', lines=True, compression='gzip')
    for name, values in arrays.items():
        np.save(output / f'{name}.npy', np.concatenate(values), allow_pickle=False)
    result = dict(completed=True, floor=floor, seconds=time.monotonic() - started,
        cases=len(meta), devices=devices, device_count=len(devices), has_thermal=has_thermal,
        sensor_mapping=mapping, groups=definitions, support=support.to_dict('records'),
        group_support=group_support, splits=SPLITS, source_quality=quality,
        source=source.name, source_doi='10.6084/m9.figshare.11726517',
        source_release_version=6, source_license='CC BY 4.0',
        source_credit='Manisa Pipattanasomporn, Gopal Chitalia, Jitkomut Songsiri, Chaodit Aswakul, Wanchalerm Pora, Surapong Suwankawin, Kulyos Audomvongseree, Naebboon Hoonchareon; CU-BEMS',
        transformation='AC records aligned to the 2019 minute calendar; observed history windows and 15-minute target means; missing thermal observations retained',
        history_minutes=60, target_bins=10, bin_minutes=15, query_stride_minutes=30,
        clock='Naive source calendar; no assumed UTC or receipt time',
        feature_recipe='Unchanged power/calendar, optional real thermal history with masks; original device/zone identity',
        new_ml_fits=0, raw_modified=False, gpu_used=False)
    for name, value in [('training_channel_summary.json', training), ('preparation.json', result)]:
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return result
