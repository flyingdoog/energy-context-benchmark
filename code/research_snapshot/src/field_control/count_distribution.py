"""Exact Bernoulli count distributions with unchanged device marginals."""
import numpy as np


def validate_probability(probability):
    p = np.asarray(probability, dtype=np.float64)
    if p.ndim != 2 or not len(p) or not p.shape[1]:
        raise ValueError('Expected nonempty member-by-lead probabilities')
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Probabilities must be finite and within [0,1]')
    return p


def count_endpoints(probability):
    p = validate_probability(probability)
    n, leads = p.shape
    independent = np.zeros((leads, n + 1), dtype=np.float64)
    independent[:, 0] = 1
    for member in range(n):
        old = independent[:, :member + 1].copy()
        independent[:, :member + 1] = old * (1 - p[member, :, None])
        independent[:, 1:member + 2] += old * p[member, :, None]
    ordered = np.sort(p, axis=0)[::-1].T
    common = np.concatenate((1 - ordered[:, :1],
                             ordered[:, :-1] - ordered[:, 1:],
                             ordered[:, -1:]), axis=1)
    return independent, common


def count_scores(pmf, observed_count):
    pmf = np.asarray(pmf, dtype=np.float64)
    y = np.asarray(observed_count)
    if pmf.ndim != 2 or pmf.shape[1] < 2 or y.shape != (len(pmf),):
        raise ValueError('Expected lead-by-count PMF and aligned integer counts')
    n = pmf.shape[1] - 1
    if (not np.isfinite(pmf).all() or (pmf < -1e-12).any()
            or not np.allclose(pmf.sum(axis=1), 1, atol=1e-10, rtol=1e-10)):
        raise ValueError('Invalid count PMF')
    if not np.isfinite(y).all() or ((y < 0) | (y > n) | (y != np.floor(y))).any():
        raise ValueError('Invalid observed counts')
    cdf = np.cumsum(pmf, axis=1)
    target_cdf = (np.arange(n)[None, :] >= y[:, None]).astype(float)
    rps = np.sum((cdf[:, :n] - target_cdf) ** 2, axis=1)
    mean = pmf @ np.arange(n + 1)
    out = {'rps': rps, 'normalized_rps': rps / n, 'count_mse': (mean - y) ** 2}
    for coverage in (80, 90):
        alpha = (1 - coverage / 100) / 2
        low = np.sum(cdf < alpha, axis=1)
        high = np.minimum(np.sum(cdf < 1 - alpha, axis=1), n)
        out[f'coverage_{coverage}'] = ((y >= low) & (y <= high)).astype(float)
        out[f'width_{coverage}'] = (high - low).astype(float)
        out[f'fraction_width_{coverage}'] = (high - low) / n
    return out


def mixture_terms(independent, common, observed_count):
    n = independent.shape[1] - 1
    a = np.cumsum(independent, axis=1)[:, :n]
    d = np.cumsum(common - independent, axis=1)[:, :n]
    truth = (np.arange(n)[None, :] >= np.asarray(observed_count)[:, None]).astype(float)
    # Normalized-RPS quadratic in lambda, separately per lead.
    return np.sum(d * (truth - a), axis=1) / n, np.sum(d * d, axis=1) / n


def fit_mixture(numerator, denominator, weights):
    a, b, w = map(lambda x: np.asarray(x, dtype=float), (numerator, denominator, weights))
    if a.shape != b.shape or a.shape != w.shape or not len(a):
        raise ValueError('Aligned nonempty mixture statistics required')
    if not all(np.isfinite(v).all() for v in (a, b, w)) or (w < 0).any() or (b < 0).any() or w.sum() <= 0:
        raise ValueError('Invalid mixture statistics/weights')
    total = float(np.sum(w * b))
    return float(np.clip(np.sum(w * a) / total, 0, 1)) if total > 1e-14 else 0.0
