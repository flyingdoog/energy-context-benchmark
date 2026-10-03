"""Validation-only objective choices on matched device/group populations."""
import numpy as np
import pandas as pd

METRICS = ['mse', 'mae']


def matched_scores(meta, residual, group_keys, bins=(2, 6, 10), minutes=15):
    """Call with exactly the eligible group members; retain original date weights."""
    e = np.asarray(residual, dtype=float)
    if e.shape != (len(meta), 10) or not np.isfinite(e).all() or len(meta) == 0:
        raise ValueError('Expected finite residuals for matched group members')
    codes, unique = pd.factorize(pd.MultiIndex.from_frame(meta[group_keys]), sort=False)
    sizes = np.bincount(codes)
    totals = np.zeros((len(unique), e.shape[1]))
    np.add.at(totals, codes, e)
    dates = meta.groupby(group_keys, sort=False, observed=True).date.first().to_numpy()
    if meta.groupby(group_keys, sort=False, observed=True).date.nunique().max() != 1:
        raise ValueError('A group cannot span multiple evaluation dates')
    rows = []
    for scale, values, labels in [('device', e, meta.date.to_numpy()),
                                  ('group_total', totals, dates),
                                  ('group_mean', totals / sizes[:, None], dates)]:
        for n in bins:
            error = values[:, :n]
            day = pd.DataFrame(dict(date=labels, mse=np.mean(error**2, axis=1),
                                    mae=np.mean(abs(error), axis=1)))
            rows.append(day.groupby('date', as_index=False).mean().assign(
                scale=scale, span_minutes=n*minutes))
    return pd.concat(rows, ignore_index=True)


def validate_scores(frame, candidates):
    keys = ['period', 'date', 'scale', 'span_minutes', 'candidate']
    if frame.duplicated(keys).any() or not np.isfinite(frame[METRICS]).all().all():
        raise ValueError('Duplicate/nonfinite candidate score')
    expected = set(candidates)
    for _, block in frame.groupby(['period', 'date', 'span_minutes']):
        for scale in ['device', 'group_total', 'group_mean']:
            if set(block.loc[block.scale == scale, 'candidate']) != expected:
                raise ValueError('Candidate/date/scale support differs')
    if 'validation' not in set(frame.period):
        raise ValueError('Validation scores required before any model choice')


def select_model(frame, candidates, scale, metric):
    """No later errors enter this function; alphabetical ties at numerical tolerance."""
    if metric not in METRICS or scale not in ('device', 'group_total', 'group_mean'):
        raise ValueError('Unknown objective')
    val = frame[(frame.period == 'validation') & frame.candidate.isin(candidates)]
    validate_scores(val, candidates)
    scores = val[val.scale == scale].groupby('candidate')[metric].mean()
    best = float(scores.min())
    tied = sorted(scores.index[np.isclose(scores, best, rtol=1e-10, atol=1e-12)])
    return tied[0], scores.to_dict()


def week_weights(dates, draws=1500):
    stamps = pd.DatetimeIndex(pd.to_datetime(dates))
    week = (stamps-pd.to_timedelta(stamps.weekday, unit='D')).strftime('%Y-%m-%d')
    unique = sorted(set(week))
    if len(unique) < 2:
        raise ValueError('At least two calendar-week clusters required')
    ids = np.array([unique.index(w) for w in week])
    counts = np.random.default_rng(2027).multinomial(
        len(unique), np.ones(len(unique))/len(unique), size=draws)
    weights = counts[:, ids].astype(float)
    return weights/weights.sum(axis=1, keepdims=True), len(unique)


def paired_effect(a, b, weights):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or weights.shape[1] != len(a) or not np.isfinite(a+b).all():
        raise ValueError('Aligned finite paired scores required')
    ba, bb = weights @ a, weights @ b
    lo, hi = np.quantile(bb-ba, [.025, .975])
    result = dict(device_selected_error=float(a.mean()), group_selected_error=float(b.mean()),
                  difference=float((b-a).mean()), low=float(lo), high=float(hi))
    if a.mean() > 0 and (ba > 0).all():
        lo, hi = np.quantile(100*(bb/ba-1), [.025, .975])
        result.update(change_pct=float(100*(b.mean()/a.mean()-1)), change_low=float(lo), change_high=float(hi))
    return result


def choice_frequencies(frame, candidates, scale, metric):
    """Selection sensitivity on the original validation dates, not new fits."""
    val=frame[(frame.period=='validation')&frame.candidate.isin(candidates)]
    validate_scores(val,candidates)
    table=val[val.scale==scale].groupby(['date','candidate'])[metric].mean().unstack().sort_index()
    ordered=sorted(candidates);table=table[ordered]
    weights,weeks=week_weights(table.index)
    samples=weights@table.to_numpy();best=samples.min(axis=1,keepdims=True)
    ties=np.isclose(samples,best,rtol=1e-10,atol=1e-12)
    selected=ties.argmax(axis=1)
    return dict(weeks=weeks,dates=len(table),draws=len(weights),
        frequency={candidate:float(np.mean(selected==i)) for i,candidate in enumerate(ordered)})
