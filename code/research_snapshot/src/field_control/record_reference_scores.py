"""Four-arm daily-error aggregation and paired calendar-week uncertainty."""
import numpy as np
import pandas as pd

RAW_ARMS = ['Power only', 'Paired'] + [f'{arm} {seed}' for arm in ['Fixed', 'Retrained'] for seed in (2027, 2028, 2029)]
ARMS = ['Power only', 'Paired', 'Fixed', 'Retrained']
METRICS = ['mse', 'mae', 'energy_ae_kwh']
KEYS = ['source', 'site', 'period', 'scale', 'span_minutes', 'date']


def aggregate_errors(raw):
    if raw.duplicated(KEYS+['method']).any():
        raise ValueError('Duplicate daily errors')
    if not np.isfinite(raw[METRICS]).all().all():
        raise ValueError('Nonfinite daily error')
    if any(set(part.method) != set(RAW_ARMS) for _, part in raw.groupby(KEYS)):
        raise ValueError('Every date/scale/span needs all eight mapping-specific arms')
    result = raw[KEYS+['method']+METRICS].copy()
    result['method'] = result.method.str.replace(r' (2027|2028|2029)$', '', regex=True)
    return result.groupby(KEYS+['method'], as_index=False)[METRICS].mean()


def paired_effects(daily, draws=1500):
    if daily.duplicated(KEYS+['method']).any():
        raise ValueError('Duplicate averaged daily errors')
    avg = daily.groupby(['source', 'site', 'period', 'scale', 'date', 'method'], as_index=False)[METRICS].mean()
    avg['span_minutes'] = 0
    data = pd.concat([daily, avg], ignore_index=True)
    rows = []
    cache = {}
    for key, part in data.groupby(['source', 'site', 'period', 'scale', 'span_minutes']):
        for metric in ['mse', 'mae']:
            p = part.pivot(index='date', columns='method', values=metric).sort_index()
            if set(p.columns) != set(ARMS) or p.isna().any().any():
                raise ValueError('Four-arm comparison has different date support')
            p = p[ARMS]
            dates = tuple(p.index)
            if dates not in cache:
                stamps = pd.DatetimeIndex(pd.to_datetime(dates))
                weeks = (stamps-pd.to_timedelta(stamps.weekday, unit='D')).strftime('%Y-%m-%d')
                unique = sorted(set(weeks))
                if len(unique) >= 2:
                    ids = np.asarray([unique.index(w) for w in weeks])
                    count = np.random.default_rng(2027).multinomial(len(unique), np.ones(len(unique))/len(unique), size=draws)
                    weights = count[:, ids].astype(float)
                    weights /= weights.sum(axis=1, keepdims=True)
                else:
                    weights = None
                cache[dates] = weights, len(unique)
            weights, nweeks = cache[dates]
            mean = p.mean().to_numpy()
            boot = weights @ p.to_numpy() if weights is not None else None
            row = dict(zip(['source', 'site', 'period', 'scale', 'span_minutes'], key), metric=metric,
                dates=len(p), weeks=nweeks, power_error=float(mean[0]), paired_error=float(mean[1]),
                frozen_error=float(mean[2]), retrained_error=float(mean[3]))
            for name, i, j in [('frozen_penalty', 2, 1), ('retrained_penalty', 3, 1), ('relearning_recovery', 2, 3)]:
                row[name+'_difference'] = float(mean[i]-mean[j])
                row[name+'_pct'] = float(100*(mean[i]-mean[j])/mean[1]) if mean[1] > 0 else None
                lo = hi = None
                if boot is not None and (boot[:, 1] > 0).all():
                    lo, hi = map(float, np.quantile(100*(boot[:, i]-boot[:, j])/boot[:, 1], [.025, .975]))
                row.update({name+'_low': lo, name+'_high': hi})
            rows.append(row)
    return pd.DataFrame(rows)
