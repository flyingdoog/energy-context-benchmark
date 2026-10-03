"""Observed ACN session trajectories. No model library or network access."""
from pathlib import Path
import numpy as np
import pandas as pd

NUMERIC = ('Power (kW)', 'Charging Current (A)', 'Actual Pilot (A)',
           'Voltage (V)', 'Energy Delivered (kWh)')
STATES = ('ADAPTIVE', 'NOT CHARGING', 'READY', 'UNPLUGGED', 'IDLE', 'CHARGING', 'OTHER')


def read_static(path):
    raw = pd.read_csv(Path(path))
    stamps = pd.to_datetime(raw.iloc[:, 0], utc=True)
    frame = raw.iloc[:, 1:].copy()
    frame.index = stamps
    # Source duplicate updates use the final row at that timestamp, not a
    # numeric average of different states. Do not infer a nominal voltage.
    frame = frame[~frame.index.duplicated(keep='last')].sort_index()
    for name in NUMERIC:
        if name not in frame:
            frame[name] = np.nan
        frame[name] = pd.to_numeric(frame[name], errors='coerce')
    if 'Charging State' not in frame:
        frame['Charging State'] = None
    return frame


def minute_grid(frame, hold_seconds=10, terminal_extension_minutes=150):
    """Past-only one-second hold, bounded age, duration-averaged minute values.

    A minute label is missing below 80% observed duration. Explicit terminal
    UNPLUGGED supports zero power/current for this departed session afterward,
    not zero station demand. An unreturned tail or an interior gap stays missing.
    No state or cumulative-energy field is extended after the record ends.
    """
    if frame.empty:
        raise ValueError('Empty source session')
    first = frame.index[0].floor('min')
    last = frame.index[-1]
    state = frame['Charging State'].fillna('').astype(str)
    terminal = state.iloc[-1] == 'UNPLUGGED'
    end = last.ceil('min') + pd.Timedelta(minutes=terminal_extension_minutes if terminal else 0)
    if (end-first).total_seconds() > 14 * 86400:
        raise ValueError('More than fourteen days in one session; inspect before expanding')
    seconds = pd.date_range(first, end, freq='s', inclusive='left')
    indices = frame.index.searchsorted(seconds, side='right') - 1
    safe = np.maximum(indices, 0)
    age = np.asarray((seconds-frame.index[safe]).total_seconds())
    fresh = (indices >= 0) & (age >= 0) & (age <= hold_seconds) & (seconds <= last)
    values = frame[list(NUMERIC)].to_numpy(dtype=float)[safe].copy()
    values[~fresh] = np.nan
    if terminal:
        after = seconds > last
        values[after, 0] = 0.0
        values[after, 1] = 0.0
    numeric = pd.DataFrame(values, index=seconds, columns=NUMERIC)
    coverage = numeric.notna().resample('min').mean()
    means = numeric.resample('min').mean().where(coverage >= .8)
    labels = state.to_numpy(dtype=object)[safe].copy()
    labels[~fresh] = ''
    observed = (labels != '')
    for name in STATES:
        match = (labels == name) if name != 'OTHER' else observed & ~np.isin(labels, STATES[:-1])
        onehot = np.where(observed, match.astype(float), np.nan)
        means['state_' + name] = pd.Series(onehot, index=seconds).resample('min').mean()
    return means, coverage, dict(first=str(first), last=str(last), terminal_unplugged=terminal,
                                 hold_seconds=hold_seconds, minimum_duration_coverage=.8)


def history_features(minutes, query, connected_at):
    """Latest sixty closed minute intervals, never any value after the query."""
    query = pd.Timestamp(query)
    grid = pd.date_range(query-pd.Timedelta(minutes=60), periods=60, freq='min')
    block = minutes.reindex(grid)
    result = {}
    column_groups = {'power': ['Power (kW)'],
                     'control': ['Actual Pilot (A)'],
                     'state': ['Charging Current (A)', 'Voltage (V)', 'Energy Delivered (kWh)']}
    for group, columns in column_groups.items():
        raw = block[columns].to_numpy(dtype=float).reshape(-1)
        result[group] = np.r_[np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0), np.isfinite(raw).astype(float)]
    categories = block[['state_' + name for name in STATES]].to_numpy(dtype=float)
    result['state'] = np.r_[result['state'], np.nan_to_num(categories, nan=0.0).reshape(-1),
                             np.isfinite(categories).astype(float).reshape(-1)]
    local = query.tz_convert('America/Los_Angeles')
    hour = local.hour + local.minute / 60
    result['calendar'] = np.array([np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24),
                                  np.sin(2*np.pi*local.dayofweek/7), np.cos(2*np.pi*local.dayofweek/7),
                                  (query-pd.Timestamp(connected_at)).total_seconds()/3600])
    return result


def future_labels(minutes, query, bins=10):
    grid = pd.date_range(pd.Timestamp(query), periods=15*bins, freq='min')
    values = minutes.reindex(grid)['Power (kW)'].to_numpy(dtype=float).reshape(bins, 15)
    valid = np.isfinite(values).sum(axis=1) >= 12
    total = np.nansum(values, axis=1)
    return np.divide(total, np.isfinite(values).sum(axis=1),
                     out=np.full(bins, np.nan), where=valid)
