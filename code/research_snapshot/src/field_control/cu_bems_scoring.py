"""Model-independent scoring for fixed CU-BEMS device populations."""
import numpy as np
import pandas as pd


def make_groups(meta, definitions=None):
    if meta.duplicated(['query', 'device_index']).any():
        raise ValueError('Repeated device in fixed population')
    if definitions is None:
        definitions = [('floor', sorted(meta.device_index.unique().tolist())),
                       ('zone2', sorted(meta.loc[meta.zone == 2, 'device_index'].unique().tolist()))]
        if len(definitions[0][1]) != 16 or len(definitions[1][1]) != 14:
            raise ValueError('Unexpected fixed source population')
    output = []
    for split, block in meta.groupby('split', sort=False):
        block = block.assign(case_index=block.index)
        pivot = block.pivot(index='query', columns='device_index', values='case_index')
        for name, members in definitions:
            complete = pivot.reindex(columns=members).dropna(how='any')
            gmeta = pd.DataFrame(dict(query=complete.index, date=[q[:10] for q in complete.index]))
            output.append((split, name, gmeta, complete.to_numpy(dtype=np.int64)))
    return output


def forecast_rows(meta, residual, groups, method, *, include_train=False):
    """Date-balanced device and fixed-group errors, with unchanged native spans."""
    if residual.shape != (len(meta), 10) or not np.isfinite(residual).all():
        raise ValueError('Finite aligned ten-bin residuals required')
    rows, days = [], []
    for split, block in meta.groupby('split', sort=False):
        if split == 'train' and not include_train:
            continue
        views = [('device', block.date.to_numpy(), residual[block.index])]
        for gs, group, gmeta, ids in groups:
            if gs != split:
                continue
            total = residual[ids].sum(axis=1)
            views.extend([(f'{group}_total', gmeta.date.to_numpy(), total),
                          (f'{group}_mean', gmeta.date.to_numpy(), total / ids.shape[1])])
        for scale, dates, errors in views:
            if not len(errors):
                continue
            for bins in (2, 6, 10):
                e = errors[:, :bins]
                daily = pd.DataFrame(dict(date=dates, mse=np.mean(e**2, axis=1),
                    mae=np.mean(abs(e), axis=1), energy_ae_kwh=abs(e.sum(axis=1)*.25)))
                daily = daily.groupby('date').mean().reset_index()
                info = dict(method=method, split=split, scale=scale, span_minutes=15*bins)
                rows.append(dict(**info, units=len(e), dates=len(daily),
                    **daily[['mse', 'mae', 'energy_ae_kwh']].mean().to_dict()))
                days.append(daily.assign(**info))
    return rows, days
