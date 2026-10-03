"""Matched three-arm error contrasts; these are model-conditional quantities."""
import numpy as np


def three_arm_effects(power, paired, reassigned, weights):
    """Use a common no-record denominator so the two contrasts add exactly."""
    x = np.asarray([power, paired, reassigned], dtype=float).T
    w = np.asarray(weights, dtype=float)
    if x.ndim != 2 or x.shape[1] != 3 or not len(x) or not np.isfinite(x).all() or (x < 0).any():
        raise ValueError('Expected three finite nonnegative paired loss vectors')
    if w.ndim != 2 or w.shape[1] != len(x) or not np.isfinite(w).all() or (w < 0).any() or not np.allclose(w.sum(1), 1):
        raise ValueError('Expected date-aligned normalized bootstrap weights')
    p, c, r = x.mean(axis=0)
    boot = w @ x
    if p <= 0 or (boot[:, 0] <= 0).any():
        raise ValueError('The common reference loss must be positive')
    result = dict(base_error=float(p), paired_error=float(c), reassigned_error=float(r))
    for name, value, samples in [
        ('total_gain', p-c, boot[:, 0]-boot[:, 1]),
        ('unlinked_gain', p-r, boot[:, 0]-boot[:, 2]),
        ('association_gain', r-c, boot[:, 2]-boot[:, 1]),
    ]:
        lo, hi = np.quantile(100*samples/boot[:, 0], [.025, .975])
        result.update({name:float(value), name+'_pct':float(100*value/p),
                       name+'_low':float(lo), name+'_high':float(hi)})
    return result


def chronological_choice(frame, metric):
    """Ordinary validation argmin; no later labels participate in selection."""
    val = frame[frame.period.eq('validation')]
    if val.empty or val.duplicated(['date','span_minutes','candidate']).any():
        raise ValueError('Need unique validation date/span/candidate scores')
    pivot = val.pivot(index=['date','span_minutes'], columns='candidate', values=metric)
    if pivot.isna().any().any() or not np.isfinite(pivot.to_numpy()).all():
        raise ValueError('Candidate validation support differs')
    scores = pivot.mean().sort_index()
    minimum = scores.min()
    tied = scores.index[np.isclose(scores, minimum, rtol=1e-10, atol=1e-12)]
    return str(tied[0]), scores.to_dict()
