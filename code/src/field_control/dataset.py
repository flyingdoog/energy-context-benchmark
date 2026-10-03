"""Lazy, batch-oriented access to a local BPA sequence release.

Inputs never include realized future commands or future device state. Current
means the first source report carrying a changed command, not dispatch time.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .schema import input_schema

STATE_NAMES = ('power_kw', 'present', 'capacity', 'oper_state', 'running', 'opt_out')
HISTORY_STATE_NAMES = STATE_NAMES[1:]
CALENDAR_NAMES = ('hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos')


def calendar_features(first_timestamp, minutes):
    shape = np.shape(minutes)
    values = np.asarray(minutes).ravel()
    unique, inverse = np.unique(values, return_inverse=True)
    stamps = (pd.Timestamp(first_timestamp) + pd.to_timedelta(unique, unit='min'))
    local = stamps.tz_convert('America/Los_Angeles')
    hour = local.hour.to_numpy() + local.minute.to_numpy()/60.0 + local.second.to_numpy()/3600.0
    day = local.dayofweek.to_numpy()
    result = np.c_[np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24),
                   np.sin(2*np.pi*day/7), np.cos(2*np.pi*day/7)]
    return result[inverse].reshape(*shape, 4).astype(np.float32)


class BpaDataset:
    """Read-only season interface. Labels and case identities are separate.

    inputs(indices) returns a batch of 24-hour, 15-minute histories by default.
    Up to 672 bins (seven days) are available without rebuilding the release.
    Power bins retain observed fractions. State/command values at each history
    bin's last minute carry their own masks; unknown numeric fields become NaN.
    """
    def __init__(self, root, season):
        self.folder = Path(root) / season
        self.receipt = json.loads((self.folder/'receipt.json').read_text())
        self.meta = pd.read_csv(self.folder/'cases.csv.gz', dtype={'alias': str, 'case_id': str})
        self.target = np.load(self.folder/'target_kw.npy', mmap_mode='r', allow_pickle=False)
        self.target_observed_fraction = np.load(self.folder/'target_observed_fraction.npy', mmap_mode='r', allow_pickle=False)
        self.grid = {name: np.load(self.folder/'grid'/f'{name}.npy', mmap_mode='r', allow_pickle=False)
                     for name in self.receipt['grid_fields']}
        self._prefix = None
        if self.target.shape != (len(self.meta), 12):
            raise ValueError('Expected one 12-bin forecast target per case')
        if not self.meta.case_id.is_unique:
            raise ValueError('Duplicate case identifiers')

    def __len__(self):
        return len(self.meta)

    @property
    def schema(self):
        """Field types and units; raw categorical codes are not magnitudes."""
        return input_schema()

    def _power_prefix(self):
        if self._prefix is None:
            power = self.grid['power_kw']
            good = np.isfinite(power)
            sums = np.zeros((len(power), power.shape[1]+1), dtype=np.float64)
            counts = np.zeros(sums.shape, dtype=np.int32)
            np.cumsum(np.where(good, power, 0), axis=1, dtype=np.float64, out=sums[:, 1:])
            np.cumsum(good, axis=1, dtype=np.int32, out=counts[:, 1:])
            self._prefix = sums, counts
        return self._prefix

    def power_windows(self, indices, relative_starts, width=15):
        """Historical means only; width-minute windows must end before origin."""
        relative_starts = np.asarray(relative_starts, dtype=np.int64)
        if width < 1 or np.any(relative_starts + width > 0):
            raise ValueError('A historical window cannot include the current/future report')
        rows = self.meta.iloc[np.atleast_1d(indices)]
        dev = rows.device_index.to_numpy()[:, None]
        starts = rows.origin_minute.to_numpy()[:, None] + relative_starts[None, :]
        if np.any(starts < 0):
            raise ValueError('History extends outside the source archive')
        sums, counts = self._power_prefix()
        count = counts[dev, starts+width] - counts[dev, starts]
        total = sums[dev, starts+width] - sums[dev, starts]
        mean = np.divide(total, count, out=np.full(total.shape, np.nan), where=count > 0)
        return mean.astype(np.float32), (count/width).astype(np.float32)

    def inputs(self, indices, lookback_bins=96, bin_minutes=15):
        if lookback_bins < 1 or bin_minutes < 1 or lookback_bins*bin_minutes > 10080:
            raise ValueError('History must fit in the seven-day source context')
        indices = np.atleast_1d(indices).astype(np.int64)
        rows = self.meta.iloc[indices]
        dev = rows.device_index.to_numpy()[:, None]
        origin = rows.origin_minute.to_numpy()[:, None]
        starts = origin + np.arange(-lookback_bins, 0)[None, :]*bin_minutes
        ends = starts + bin_minutes-1
        power, fraction = self.power_windows(indices, np.arange(-lookback_bins, 0)*bin_minutes, bin_minutes)
        online = self.grid['online'][dev, ends]
        state = np.stack([self.grid[n][dev, ends] for n in HISTORY_STATE_NAMES], axis=-1).astype(np.float32)
        state_mask = online[..., None] & np.isfinite(state)
        state[~state_mask] = np.nan
        code = self.grid['command_type'][dev, ends].astype(np.int16)
        code[~online] = -1
        parameter = self.grid['command_numeric'][dev, ends].astype(np.float32)
        parameter[~online] = np.nan
        current = np.stack([self.grid[n][dev[:, 0], origin[:, 0]] for n in STATE_NAMES], axis=-1).astype(np.float32)
        command = np.c_[self.grid['command_type'][dev[:, 0], origin[:, 0]],
                        self.grid['command_numeric'][dev[:, 0], origin[:, 0]]].astype(np.float32)
        return {
            'past_power_kw': power, 'past_power_observed_fraction': fraction,
            'past_state': state, 'past_state_observed': state_mask,
            'past_command_type': code, 'past_command_parameter': parameter,
            'past_command_observed': online & np.isfinite(parameter),
            'current_state': current, 'current_state_observed': np.isfinite(current),
            'current_command': command,
            'past_calendar': calendar_features(self.receipt['first_timestamp'], ends),
            'future_calendar': calendar_features(self.receipt['first_timestamp'], origin + 1 + 15*np.arange(12)[None, :])}

    def submission(self, path, prediction_kw):
        prediction_kw = np.asarray(prediction_kw)
        if prediction_kw.shape != self.target.shape or not np.isfinite(prediction_kw).all():
            raise ValueError('Submit finite predictions for every case and lead')
        np.savez_compressed(path, case_id=self.meta.case_id.to_numpy(dtype=str),
                            prediction_kw=prediction_kw.astype(np.float32))
