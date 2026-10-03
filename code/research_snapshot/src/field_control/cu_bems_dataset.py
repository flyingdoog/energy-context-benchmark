"""Typed query-time views of the existing CU-BEMS floor packets; no model imports."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .cu_bems_scoring import make_groups, forecast_rows


class CuBemsDataset:
    def __init__(self, root, *, floor):
        if isinstance(floor, bool) or not isinstance(floor, (int, np.integer)) or floor not in range(1, 8):
            raise ValueError('CU-BEMS floor must be an integer from 1 to 7')
        self.root = Path(root)
        self.floor = floor
        self.receipt = json.loads((self.root / 'preparation.json').read_text())
        if not self.receipt.get('completed') or self.receipt.get('floor', 2) != floor:
            raise ValueError('Prepared floor differs from requested task')
        self.meta = pd.read_json(self.root / 'cases.jsonl.gz', orient='records',
                                 lines=True, convert_dates=False).reset_index(drop=True)
        self.meta['local_date'] = self.meta.date
        self.meta['floor'] = floor
        if (not len(self.meta) or not self.meta.case_id.is_unique or
                not self.meta.case_id.str.startswith(f'cu_f{floor}_').all() or
                self.meta.duplicated(['query', 'device_index']).any()):
            raise ValueError('Unique aligned floor cases required')
        self.device_count = len(self.receipt['devices'])
        self.power = np.load(self.root / 'power.npy', mmap_mode='r', allow_pickle=False)
        self.calendar = np.load(self.root / 'calendar.npy', mmap_mode='r', allow_pickle=False)
        self.target = np.load(self.root / 'target.npy', mmap_mode='r', allow_pickle=False)
        self.has_thermal = self.receipt.get('has_thermal', True)
        self.thermal = (np.load(self.root / 'thermal.npy', mmap_mode='r', allow_pickle=False)
                        if self.has_thermal else None)
        n = len(self.meta)
        if self.power.shape != (n, 60) or self.calendar.shape != (n, 4) or self.target.shape != (n, 10):
            raise ValueError('Prepared CU-BEMS packet dimensions differ')
        if self.has_thermal and self.thermal.shape != (n, 60, 2):
            raise ValueError('Prepared thermal dimensions differ')
        expected = [('floor', list(range(self.device_count)))]
        devices = self.receipt['devices']
        for zone in sorted({d['zone'] for d in devices}):
            members = [i for i, d in enumerate(devices) if d['zone'] == zone]
            if len(members) >= 2 and members != expected[0][1]:
                expected.append((f'zone{zone}', members))
        if 'groups' in self.receipt:
            recorded = [(name, list(members)) for name, members in self.receipt['groups']]
            if recorded != expected:
                raise ValueError('Prepared fixed membership differs from source devices')
        self.groups = expected
        self.available_conditions = ('power', 'power_state', 'full') if self.has_thermal else ('power',)

    def __len__(self):
        return len(self.meta)

    def close(self):
        """Release read-only packet mappings, including Windows file handles."""
        for name in ('power', 'calendar', 'target', 'thermal'):
            array = getattr(self, name, None)
            mapping = getattr(array, '_mmap', None)
            if mapping is not None and not mapping.closed:
                mapping.close()

    def _indices(self, indices):
        if indices is None:
            return np.arange(len(self))
        ix = np.atleast_1d(indices)
        if ix.dtype.kind == 'b':
            if ix.shape != (len(self),):
                raise ValueError('Boolean case mask has wrong length')
            ix = np.flatnonzero(ix)
        if ix.ndim != 1 or ix.dtype.kind not in 'iu' or ((ix < 0) | (ix >= len(self))).any():
            raise ValueError('Use in-range integer case indices')
        return ix.astype(int)

    def inputs(self, indices=None, condition='full'):
        if condition not in self.available_conditions:
            raise ValueError(f'Floor {self.floor} supports input conditions {self.available_conditions}')
        ix = self._indices(indices)
        power = np.array(self.power[ix, :, None], copy=True)
        out = dict(power_kw=power, power_observed=np.isfinite(power),
            history_offsets_minutes=np.arange(-60, 0),
            target_intervals_minutes=np.c_[15*np.arange(10), 15*np.arange(1, 11)],
            calendar=np.array(self.calendar[ix], copy=True),
            calendar_names=('hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos'),
            device_index=self.meta.device_index.to_numpy()[ix].copy(),
            device_names=tuple(d['column'] for d in self.receipt['devices']),
            zone=self.meta.zone.to_numpy()[ix].copy(), floor=self.floor)
        if condition != 'power':
            state = np.array(self.thermal[ix], copy=True)
            out.update(numeric_state=state, numeric_state_observed=np.isfinite(state),
                numeric_state_names=('zone_temperature', 'zone_relative_humidity'),
                numeric_state_units=('degC', '%'), state_scope='Shared sensor within each source zone')
        return out

    def score(self, prediction, method='submission', indices=None, *, activity=False):
        ix = self._indices(indices)
        if not len(ix) or len(np.unique(ix)) != len(ix):
            raise ValueError('Score unique nonempty cases')
        values = np.asarray(prediction, dtype=float)
        if values.shape != (len(ix), 10) or not np.isfinite(values).all():
            raise ValueError('Expected one finite ten-bin forecast per case')
        if activity and ((values < 0) | (values > 1)).any():
            raise ValueError('Probability must be within [0,1]')
        meta = self.meta.iloc[ix].reset_index(drop=True)
        target = self.target[ix]
        residual = values - ((target > 0) if activity else target)
        rows, days = forecast_rows(meta, residual, make_groups(meta, self.groups), method, include_train=True)
        summary, daily = pd.DataFrame(rows), pd.concat(days, ignore_index=True)
        if activity:
            summary = summary[summary.scale != 'device'].copy()
            daily = daily[daily.scale != 'device'].copy()
            for frame in (summary, daily):
                frame['scale'] = frame.scale.str.replace('_total', '_positive_count').str.replace('_mean', '_positive_fraction')
                count = frame.scale.str.endswith('_count')
                frame['mse_unit'] = np.where(count, 'devices squared', 'fraction squared')
                frame['mae_unit'] = np.where(count, 'devices', 'fraction')
            common = ['scale', 'span_minutes', 'method', 'mse', 'mae', 'mse_unit', 'mae_unit']
            return summary[common + ['units', 'dates']], daily[common + ['date']]
        return summary, daily
