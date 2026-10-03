"""Native group residuals and date-balanced level/variation decomposition."""
import numpy as np
import pandas as pd
from .acn_scoring import complete_cohort_mask

LEADS = [f'e{j}' for j in range(10)]


def group_residuals(source, meta, residual, population=None):
    residual = np.asarray(residual, dtype=float)
    if residual.shape != (len(meta), 10) or not np.isfinite(residual).all():
        raise ValueError('Expected complete finite ten-lead residuals')
    if source == 'BPA':
        if meta.duplicated(['cohort_id', 'alias']).any():
            raise ValueError('Repeated BPA member')
        keys = ['local_date', 'cohort_id', 'command_type']
        frame = meta[keys].copy()
        frame[LEADS] = residual
        minimum = 10
    elif source == 'ACN':
        if population is None:
            raise ValueError('ACN requires its full query-known population')
        keep = complete_cohort_mask(meta, population)
        duplicates = meta.duplicated(['query_utc', 'station_id'], keep=False)
        bad = set(meta.loc[duplicates, 'query_utc'])
        keep &= ~meta.query_utc.isin(bad).to_numpy()
        keys = ['local_date', 'query_utc']
        frame = meta[keys].copy()
        frame[LEADS] = residual
        frame = frame.loc[keep]
        minimum = 2
    else:
        raise ValueError('Group level analysis supports BPA and ACN')
    grouped = frame.groupby(keys, sort=True)
    mean, counts = grouped[LEADS].mean(), grouped.size()
    mean = mean.loc[counts >= minimum]
    counts = counts.loc[counts >= minimum]
    if mean.empty:
        raise ValueError('No complete supported groups')
    total = mean.mul(counts, axis=0)
    return {scale: values.reset_index().rename(columns={'local_date': 'date'})
            for scale, values in [('cohort_mean', mean), ('cohort_total', total)]}


def equal_date_mean(frame):
    """One mean per source-calendar date, then equal date weights."""
    return frame.groupby('date')[LEADS].mean().mean().to_numpy()


def lead_decomposition(frame, validation_offset=None):
    e = frame[LEADS].to_numpy(dtype=float)
    if not np.isfinite(e).all():
        raise ValueError('Nonfinite group residuals')
    offset = np.zeros(10) if validation_offset is None else np.asarray(validation_offset, dtype=float)
    if offset.shape != (10,) or not np.isfinite(offset).all():
        raise ValueError('A fixed finite validation intercept per lead is required')
    e = e-offset
    work = pd.DataFrame(e, columns=LEADS).assign(date=frame.date.to_numpy())
    mean = equal_date_mean(work)
    squared = pd.DataFrame(e**2, columns=LEADS).assign(date=frame.date.to_numpy())
    absolute = pd.DataFrame(abs(e), columns=LEADS).assign(date=frame.date.to_numpy())
    mse, mae = equal_date_mean(squared), equal_date_mean(absolute)
    variation = mse-mean**2
    if (variation < -1e-8).any():
        raise ValueError('Date-weighted variance is negative beyond numerical precision')
    variation = np.maximum(variation, 0)
    return pd.DataFrame(dict(lead=np.arange(1, 11), mse=mse, mae=mae,
                             signed_mean=mean, bias_mse=mean**2, centered_mse=variation))


def daily_errors(frame, validation_offset=None):
    offset = np.zeros(10) if validation_offset is None else np.asarray(validation_offset)
    e = frame[LEADS].to_numpy(dtype=float)-offset
    rows = []
    for bins in [2, 6, 10]:
        values = e[:, :bins]
        day = pd.DataFrame(dict(date=frame.date.to_numpy(), mse=np.mean(values**2, axis=1),
            mae=np.mean(abs(values), axis=1), energy_ae_kwh=abs(values.sum(axis=1)*.25)))
        rows.append(day.groupby('date', as_index=False).mean().assign(span_minutes=bins*15))
    return pd.concat(rows, ignore_index=True)


def span_decomposition(leads):
    columns = ['mse', 'mae', 'bias_mse', 'centered_mse']
    rows = [dict(span_minutes=n*15, **leads.iloc[:n][columns].mean().to_dict()) for n in [2, 6, 10]]
    rows.append(dict(span_minutes=0, **pd.DataFrame(rows)[columns].mean().to_dict()))
    return pd.DataFrame(rows)


def paired_week_difference(dates, first, second, draws=1500):
    """Paired difference and pointwise interval in original loss units."""
    dates = pd.DatetimeIndex(pd.to_datetime(dates))
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if len(dates) != len(first) or first.shape != second.shape or len(set(dates)) != len(dates):
        raise ValueError('Paired daily arrays need identical unique dates')
    value = first-second
    if not np.isfinite(value).all():
        raise ValueError('Nonfinite paired errors')
    weeks = (dates-pd.to_timedelta(dates.weekday, unit='D')).strftime('%Y-%m-%d')
    unique = sorted(set(weeks))
    if len(unique) < 2:
        return dict(difference=float(value.mean()), low=None, high=None, weeks=len(unique))
    ids = np.asarray([unique.index(w) for w in weeks])
    counts = np.random.default_rng(2027).multinomial(len(unique), np.ones(len(unique))/len(unique), size=draws)
    weights = counts[:, ids].astype(float)
    weights /= weights.sum(axis=1, keepdims=True)
    low, high = np.quantile(weights@value, [.025, .975])
    return dict(difference=float(value.mean()), low=float(low), high=float(high), weeks=len(unique))
