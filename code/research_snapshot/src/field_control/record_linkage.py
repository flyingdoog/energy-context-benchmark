"""Query-local record reassignment and support diagnostics; no target access."""
import json
import numpy as np
import pandas as pd

GROUP_KEYS = ['source', 'site', 'period', 'query_group']
IDENTITY_COLUMNS = GROUP_KEYS + ['member', 'case_id', 'date']


def donor_mapping(metadata, min_members, seed=2027):
    """Random cyclic reassignment within eligible, distinct-member query groups.

    Only identity/calendar metadata are read. Target-completeness flags cannot
    determine donors. Small or duplicate-member groups remain unchanged.
    """
    if min_members < 2:
        raise ValueError('Reassignment needs at least two distinct members')
    meta = metadata[IDENTITY_COLUMNS].reset_index(drop=True).copy()
    if meta.isna().any().any() or not meta.case_id.is_unique:
        raise ValueError('Expected complete group identities and unique cases')
    donor = np.arange(len(meta), dtype=np.int64)
    rows = []
    for key, indices in meta.groupby(GROUP_KEYS, sort=True).indices.items():
        indices = np.asarray(indices, dtype=np.int64)
        part = meta.iloc[indices]
        distinct = part.member.nunique() == len(part)
        allowed = len(part) >= min_members and distinct
        if part.date.nunique() != 1:
            raise ValueError('One query group cannot span multiple local dates')
        if allowed:
            order = indices[np.argsort(part.case_id.to_numpy(dtype=str), kind='stable')]
            payload = json.dumps(list(key), ensure_ascii=True, separators=(',', ':')).encode('utf-8')
            rng = np.random.default_rng(np.random.SeedSequence([int(seed), *payload]))
            order = rng.permutation(order)
            donor[order] = np.roll(order, 1)
        rows.append(dict(zip(GROUP_KEYS, key), date=part.date.iloc[0],
                         members=len(part), distinct_members=bool(distinct),
                         reassigned=bool(allowed)))
    return donor, pd.DataFrame(rows)


def reassign_blocks(values, donor, moved_columns, recipients=None):
    """Return recipient features with the complete donor block, masks included."""
    values = np.asarray(values)
    donor = np.asarray(donor)
    moved = np.asarray(moved_columns, dtype=int)
    ix = np.arange(len(values)) if recipients is None else np.asarray(recipients, dtype=int)
    if values.ndim != 2 or donor.shape != (len(values),) or donor.dtype.kind not in 'iu':
        raise ValueError('Invalid feature/donor dimensions')
    if ((donor < 0) | (donor >= len(values))).any():
        raise ValueError('Donor outside the query-input matrix')
    if moved.ndim != 1 or len(np.unique(moved)) != len(moved) or ((moved < 0) | (moved >= values.shape[1])).any():
        raise ValueError('Invalid moved columns')
    if ix.ndim != 1 or ((ix < 0) | (ix >= len(values))).any():
        raise ValueError('Recipient outside the input matrix')
    out = values[ix].copy()
    out[:, moved] = values[np.ix_(donor[ix], moved)]
    return out


def support_statistics(metadata, values, donor, groups, column_groups):
    """Report reassignment support and actual feature change, not accuracy."""
    meta = metadata.reset_index(drop=True)
    values = np.asarray(values)
    if len(meta) != len(values) or len(donor) != len(meta):
        raise ValueError('Metadata and input lengths differ')
    group_index = pd.MultiIndex.from_frame(meta[GROUP_KEYS])
    eligible_index = pd.MultiIndex.from_frame(groups.loc[groups.reassigned, GROUP_KEYS])
    movable = group_index.isin(eligible_index)
    changed = {}
    for name, columns in column_groups.items():
        columns = np.asarray(columns, dtype=int)
        flags = np.zeros(len(meta), dtype=bool)
        for start in range(0, len(meta), 1024):
            rows = np.arange(start, min(start+1024, len(meta)))
            same = np.isclose(values[np.ix_(rows, columns)],
                              values[np.ix_(donor[rows], columns)],
                              rtol=1e-6, atol=1e-7, equal_nan=True)
            flags[rows] = (~same).any(axis=1)
        changed[name] = flags
    # Completeness enters the support readout only, never donor_mapping.
    complete = meta.target_complete.to_numpy(dtype=bool) if 'target_complete' in meta else np.ones(len(meta),dtype=bool)
    status = meta[GROUP_KEYS].copy()
    status['complete'] = complete
    all_complete = status.groupby(GROUP_KEYS).complete.all()
    groups = groups.copy()
    groups['complete_targets'] = all_complete.reindex(pd.MultiIndex.from_frame(groups[GROUP_KEYS])).to_numpy()
    rows = []
    for key, ix in meta.groupby(['source','site','period'], sort=True).indices.items():
        ix = np.asarray(ix)
        part = groups[(groups.source == key[0]) & (groups.site == key[1]) & (groups.period == key[2])]
        active = part[part.reassigned]
        scored = active[active.complete_targets]
        item = dict(zip(['source','site','period'], key),
            query_known_cases=len(ix), labeled_cases=int(complete[ix].sum()),
            query_groups=len(part), reassignable_groups=len(active),
            scored_groups=len(scored), scored_dates=int(scored.date.nunique()),
            moved_cases=int((donor[ix] != ix).sum()),
            moved_labeled_cases=int(((donor[ix] != ix) & complete[ix]).sum()),
            median_members=float(active.members.median()) if len(active) else None,
            min_members=int(active.members.min()) if len(active) else None,
            max_members=int(active.members.max()) if len(active) else None)
        for name, flags in changed.items():
            item[name+'_changed_cases'] = int(flags[ix].sum())
            denominator = int(movable[ix].sum())
            item[name+'_changed_fraction_among_moved'] = float(flags[ix].sum()/denominator) if denominator else None
        rows.append(item)
    return pd.DataFrame(rows), groups
