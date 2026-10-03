"""Date-balanced group squared-error accounting; no model fitting."""
import numpy as np
import pandas as pd
from .acn_scoring import complete_cohort_mask

COMPONENTS = ['group_mse', 'individual_sum', 'cross_error']


def decompose_group_errors(meta, residual, source, population=None):
    """Return daily terms in (sum e_i)^2 = sum e_i^2 + sum_{i!=j} e_i e_j.

    The cross-error term is not a covariance: residual means are not removed.
    Group membership and minimum size match the original source scorers.
    """
    error = np.asarray(residual, dtype=np.float64)
    if error.shape != (len(meta), 10) or not np.isfinite(error).all():
        raise ValueError('Expected ten finite residuals per case')
    if source == 'ACN':
        if population is None:
            raise ValueError('ACN requires the full query-known population')
        keys = ['site', 'split', 'local_date', 'query_utc']
        duplicate = meta.duplicated(['site', 'split', 'query_utc', 'station_id'], keep=False)
        bad = pd.MultiIndex.from_frame(meta.loc[duplicate, ['site', 'split', 'query_utc']])
        keep = complete_cohort_mask(meta, population) & ~pd.MultiIndex.from_frame(
            meta[['site', 'split', 'query_utc']]).isin(bad)
        minimum = 2
    elif source == 'BPA':
        if meta.duplicated(['cohort_id', 'alias']).any():
            raise ValueError('Repeated device within group')
        keys = ['local_date', 'cohort_id', 'command_type']
        keep = np.ones(len(meta), dtype=bool)
        minimum = 10
    else:
        raise ValueError('Only existing BPA and ACN group tasks are supported')
    frame = meta.loc[keep, keys].reset_index(drop=True).copy()
    ecols = [f'e{i}' for i in range(10)]
    qcols = [f'q{i}' for i in range(10)]
    frame[ecols] = error[keep]
    frame[qcols] = error[keep] ** 2
    groups = frame.groupby(keys, sort=False)
    summed = groups[ecols + qcols].sum()
    sizes = groups.size()
    summed = summed.loc[sizes >= minimum]
    sizes = sizes.loc[sizes >= minimum]
    if summed.empty:
        raise ValueError('No eligible groups')
    totals = summed[ecols].to_numpy() ** 2
    individual = summed[qcols].to_numpy()
    dates = summed.index.get_level_values('local_date')
    rows = []
    for bins in [2, 6, 10]:
        total = totals[:, :bins].mean(axis=1)
        own = individual[:, :bins].mean(axis=1)
        for scale, divisor in [('cohort_total', np.ones(len(sizes))),
                               ('cohort_mean', sizes.to_numpy(dtype=float) ** 2)]:
            per_group = pd.DataFrame(dict(local_date=dates,
                group_mse=total / divisor, individual_sum=own / divisor,
                cross_error=(total - own) / divisor, members=sizes.to_numpy()))
            daily = per_group.groupby('local_date')[COMPONENTS + ['members']].mean()
            daily['groups'] = per_group.groupby('local_date').size()
            rows.append(daily.reset_index().assign(scale=scale, span_minutes=15 * bins))
    return pd.concat(rows, ignore_index=True)
