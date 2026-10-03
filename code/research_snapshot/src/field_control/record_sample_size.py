"""Whole-session training-quantity intervention for the ACN record comparison."""
import numpy as np
import pandas as pd
from .record_reference_scores import KEYS, METRICS


def session_table(metadata, recipients):
    rows = metadata.iloc[np.asarray(recipients, dtype=int)].copy()
    if set(rows.period) != {'train'}:
        raise ValueError('Session sampling is restricted to training inputs')
    if not rows.case_id.str.contains('@', regex=False).all():
        raise ValueError('ACN case IDs must include a session and query')
    rows['session'] = rows.case_id.str.rsplit('@', n=1).str[0]
    rows['month'] = rows.date.astype(str).str[:7]
    return rows.groupby('session', as_index=False).agg(
        month=('month', 'min'), windows=('case_id', 'size'))


def select_training_sessions(metadata, recipients, budget, seed):
    sessions = session_table(metadata, recipients)
    if not isinstance(budget, (int, np.integer)) or not 1 <= budget <= len(sessions):
        raise ValueError('Session budget outside training support')
    counts = sessions.groupby('month').size().sort_index()
    ideal = counts * (budget / len(sessions))
    quotas = np.floor(ideal).astype(int)
    left = budget - int(quotas.sum())
    for month in sorted(counts.index, key=lambda m: (-(ideal[m]-quotas[m]), m))[:left]:
        quotas[month] += 1
    selected = []
    # Separate month streams preserve prefixes at the two fixed budgets.
    for i, (month, n) in enumerate(quotas.items()):
        names = np.array(sorted(sessions.loc[sessions.month == month, 'session']))
        order = np.random.default_rng(np.random.SeedSequence([int(seed), i])).permutation(names)
        selected.extend(order[:int(n)].tolist())
    case_sessions = metadata.iloc[recipients].case_id.str.rsplit('@', n=1).str[0]
    positions = np.flatnonzero(case_sessions.isin(selected).to_numpy())
    details = dict(sampling_seed=int(seed), training_sessions=len(selected),
        training_cases=len(positions), original_sessions=len(sessions),
        original_cases=len(recipients), monthly_quotas={k:int(v) for k,v in quotas.items()},
        selected_sessions=sorted(selected))
    return positions, details


def average_sampling_runs(frames):
    if set(frames) != {2031, 2032, 2033}:
        raise ValueError('All three planned session samples are required')
    keys = KEYS + ['method']
    aligned, index = [], None
    for seed, frame in sorted(frames.items()):
        if frame.duplicated(keys).any():
            raise ValueError('Duplicate date/span/method score')
        table = frame.set_index(keys)[METRICS].sort_index()
        if not np.isfinite(table.to_numpy()).all():
            raise ValueError('Nonfinite daily score')
        if index is not None and not table.index.equals(index):
            raise ValueError('Sampling runs must retain identical evaluation support')
        index = table.index
        aligned.append(table)
    result = aligned[0].copy()
    result.loc[:, METRICS] = np.mean([v.to_numpy() for v in aligned], axis=0)
    return result.reset_index()
