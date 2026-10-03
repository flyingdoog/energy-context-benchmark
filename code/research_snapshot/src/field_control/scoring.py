"""Case-aligned scores in reported kW, preserving dates and device groups."""
from pathlib import Path
import numpy as np
import pandas as pd

HORIZONS = (15, 30, 60, 120, 180)


def read_submission(path, case_ids):
    with np.load(Path(path), allow_pickle=False) as packet:
        ids = pd.Index(packet['case_id'].astype(str))
        prediction = packet['prediction_kw']
    expected = pd.Index(np.asarray(case_ids).astype(str))
    if not ids.is_unique or not expected.is_unique or len(ids) != len(expected):
        raise ValueError('Submission must contain every case exactly once')
    order = ids.get_indexer(expected)
    if (order < 0).any() or prediction.shape != (len(ids), 12):
        raise ValueError('Submission case set or lead shape differs from the benchmark')
    if not np.isfinite(prediction).all():
        raise ValueError('Predictions contain missing or infinite values')
    return prediction[order]


def score_predictions(meta, target, prediction, method, season, min_cohort=10):
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if target.shape != (len(meta), 12) or target.shape != prediction.shape:
        raise ValueError('Expected case-aligned 12-bin predictions')
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError('Missing predictions/targets cannot be silently dropped')
    if meta.duplicated(['cohort_id', 'alias']).any():
        raise ValueError('A device cannot appear twice in a synchronous group')
    frame = meta[['local_date', 'cohort_id', 'alias', 'command_type']].copy()
    cols = [f'r{j}' for j in range(12)]
    frame[cols] = prediction-target
    by = frame.groupby(['local_date', 'cohort_id', 'command_type'], sort=False)
    group = by[cols].mean().reset_index()
    group['devices'] = by.size().to_numpy()
    group = group[group.devices >= min_cohort].copy()
    total = group.copy()
    total[cols] = total[cols].to_numpy()*total.devices.to_numpy()[:, None]
    records, dates = [], []
    for scale, table in [('device', frame), ('cohort_mean', group), ('cohort_total', total)]:
        for minutes in HORIZONS:
            residual = table[cols[:minutes//15]].to_numpy()
            errors = table[['local_date', 'command_type']].copy()
            errors['mse'] = np.mean(residual**2, axis=1)
            errors['mae'] = np.mean(np.abs(residual), axis=1)
            # This integrates binned reported power, not independently metered energy.
            errors['reported_energy_ae_kwh'] = np.abs(residual.sum(axis=1)*.25)
            for code in ['all', 1, 5, 8]:
                part = errors if code == 'all' else errors[errors.command_type == code]
                if part.empty:
                    continue
                daily = part.groupby('local_date')[['mse', 'mae', 'reported_energy_ae_kwh']].mean()
                shared = dict(method=method, season=season, scale=scale, horizon_minutes=minutes,
                              command_type=str(code), minimum_group_devices=min_cohort)
                records.append(dict(**shared, units_scored=len(part), dates=len(daily), **daily.mean().to_dict()))
                dates.append(daily.reset_index().assign(**shared))
    return pd.DataFrame(records), pd.concat(dates, ignore_index=True)
