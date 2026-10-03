"""Whole-query-group inputs from the completed record-linkage cache.

The input class never retrieves targets. Explicit helpers below separately
load outcomes for training or scoring. The two input views retain the same
power/history and auxiliary-record multisets, with different associations.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .record_linkage import GROUP_KEYS, reassign_blocks


class RecordSetInputs:
    def __init__(self, cases, features, donor, moved_columns, groups):
        self.cases = cases.reset_index(drop=True)
        self.features = features
        self.donor = np.asarray(donor, dtype=np.int64)
        self.moved_columns = np.asarray(moved_columns, dtype=np.int64)
        if features.ndim != 2 or len(features) != len(cases):
            raise ValueError('Feature rows do not match query-known cases')
        if self.donor.shape != (len(cases),):
            raise ValueError('Wrong donor index shape')
        if not self.cases.case_id.is_unique:
            raise ValueError('Repeated query case')
        keep = groups.reassigned & groups.complete_targets
        self.meta = groups.loc[keep].reset_index(drop=True).copy()
        if self.meta.duplicated(GROUP_KEYS).any() or self.meta.empty:
            raise ValueError('Expected distinct supported query groups')
        by_group = self.cases.groupby(GROUP_KEYS, sort=False).indices
        self.rows = []
        for group in self.meta.to_dict('records'):
            key = tuple(group[name] for name in GROUP_KEYS)
            rows = np.asarray(by_group[key], dtype=np.int64)
            part = self.cases.iloc[rows]
            if len(rows) != group['members'] or part.member.nunique() != len(rows):
                raise ValueError('Incomplete or duplicate-member group')
            if not part.target_complete.all():
                raise ValueError('Saved group support contains an unlabelled member')
            # All original records must remain accessible to the set model.
            if not np.array_equal(np.sort(self.donor[rows]), np.sort(rows)):
                raise ValueError('Reassignment is not a bijection of this complete group')
            self.rows.append(rows)
        self.protected_columns = np.setdiff1d(np.arange(features.shape[1]), self.moved_columns)

    def __len__(self):
        return len(self.rows)

    def group(self, index, *, reassigned=False):
        rows = self.rows[index]
        if reassigned:
            return reassign_blocks(self.features, self.donor, self.moved_columns, rows)
        return np.array(self.features[rows], dtype=np.float32, copy=True)

    def batch(self, indices, *, reassigned=False):
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1 or not len(indices) or ((indices < 0) | (indices >= len(self))).any():
            raise ValueError('Invalid group batch')
        rows = np.concatenate([self.rows[index] for index in indices])
        counts = np.array([len(self.rows[index]) for index in indices], dtype=np.int64)
        values = (reassign_blocks(self.features, self.donor, self.moved_columns, rows)
                  if reassigned else np.array(self.features[rows], dtype=np.float32, copy=True))
        segments = np.repeat(np.arange(len(indices), dtype=np.int64), counts)
        return values, segments, counts


def load_cached_sets(folder, *, site=None, period=None, seed=2027):
    """Reuse saved complete-group support; never reconstruct source records."""
    folder = Path(folder)
    receipt = json.loads((folder / 'receipt.json').read_text())
    if not receipt.get('completed'):
        raise ValueError('Record-linkage input cache is incomplete')
    cases = pd.read_csv(folder / 'cases.csv.gz', dtype={'member': str})
    features = np.load(folder / 'features.npy', mmap_mode='r', allow_pickle=False)
    with np.load(folder / f'mapping_{seed}.npz', allow_pickle=False) as mapping:
        if not np.array_equal(mapping['case_id'].astype(str), cases.case_id.to_numpy(dtype=str)):
            raise ValueError('Mapping case order differs from the cached input rows')
        donor, moved = mapping['donor_index'], mapping['moved_columns']
    groups = pd.read_csv(folder / f'groups_{seed}.csv.gz')
    if site is not None:
        groups = groups[groups.site == site]
    if period is not None:
        groups = groups[groups.period == period]
    return RecordSetInputs(cases, features, donor, moved, groups)


class SetFeatureNormalizer:
    """Training-member moments, with mean fill and explicit finite masks."""
    def __init__(self, mean, scale):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)

    @classmethod
    def fit(cls, inputs, chunk_rows=512):
        if set(inputs.meta.period) != {'train'}:
            raise ValueError('Normalization must use training groups only')
        width = inputs.features.shape[1]
        count = np.zeros(width, dtype=np.int64)
        total = np.zeros(width, dtype=np.float64)
        squares = np.zeros(width, dtype=np.float64)
        members = np.concatenate(inputs.rows)
        for begin in range(0, len(members), chunk_rows):
            values = np.asarray(inputs.features[members[begin:begin + chunk_rows]], dtype=np.float64)
            observed = np.isfinite(values)
            values = np.where(observed, values, 0)
            count += observed.sum(axis=0)
            total += values.sum(axis=0)
            squares += (values * values).sum(axis=0)
        mean = total / np.maximum(count, 1)
        variance = np.maximum(squares / np.maximum(count, 1) - mean * mean, 0)
        scale = np.sqrt(variance)
        scale[scale < 1e-6] = 1
        return cls(mean, scale)

    def transform(self, values):
        values = np.asarray(values, dtype=np.float32)
        observed = np.isfinite(values)
        normalized = np.where(observed, (values - self.mean) / self.scale, 0)
        return np.concatenate((normalized, observed.astype(np.float32)), axis=1)


def training_group_targets(inputs, project_root):
    """Read only the original training rows into the learning objective."""
    if set(inputs.meta.period) != {'train'}:
        raise ValueError('This feasibility loader is restricted to training groups')
    return group_targets(inputs, project_root)


def group_targets(inputs, project_root):
    """Load the original outcomes separately from the group input view."""
    root = Path(project_root)
    sources = set(inputs.meta.source)
    periods = set(inputs.meta.period)
    if len(periods) != 1 or inputs.meta.site.nunique() != 1:
        raise ValueError('Load one source/site/period target set at a time')
    period = next(iter(periods))
    indices = inputs.cases.original_index.to_numpy(dtype=np.int64)
    members = np.concatenate(inputs.rows)
    if (indices[members] < 0).any():
        raise ValueError('Group includes an unavailable target')
    if sources == {'BPA'}:
        season = 'winter' if period in ('train', 'validation') else period
        if season not in ('winter', 'spring', 'summer'):
            raise ValueError('Unknown BPA target period')
        path = root / f'results/benchmark_bpa_sequences_v1/{season}/target_kw.npy'
        target = np.load(path, mmap_mode='r', allow_pickle=False)
        values = np.asarray(target[indices[members], 2:], dtype=np.float32)
    elif sources == {'ACN'}:
        with np.load(root / 'results/acn_fleet_months_v1/packet.npz', allow_pickle=False) as packet:
            values = np.asarray(packet['y'][indices[members]], dtype=np.float32)
    else:
        raise ValueError('Use one supported source/site separately')
    counts = np.array([len(rows) for rows in inputs.rows])
    starts = np.r_[0, counts.cumsum()[:-1]]
    result = np.add.reduceat(values, starts, axis=0) / counts[:, None]
    if result.shape != (len(inputs), 10) or not np.isfinite(result).all():
        raise ValueError('Expected ten finite group-mean targets')
    return result.astype(np.float32)
