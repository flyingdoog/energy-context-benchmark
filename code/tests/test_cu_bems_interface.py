import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_local
import numpy as np
import pandas as pd
from field_control import OperationalForecastTask, CuBemsDataset


def packet(root, floor=3, thermal=True, missing_member=False):
    rows = []
    for q, split in enumerate(('train', 'validation', 'later_1', 'later_2')):
        stamp = ('2019-04-01', '2019-07-01', '2019-09-01', '2019-11-01')[q]
        for d in range(3):
            if missing_member and split == 'later_1' and d == 2:
                continue
            rows.append(dict(case_id=f'cu_f{floor}_{d}_{q}', device_index=d,
                device=f'AC{d}', zone=2 if d < 2 else 1, query=stamp+' 01:00:00', date=stamp, split=split))
    meta = pd.DataFrame(rows)
    meta.to_json(root/'cases.jsonl.gz', orient='records', lines=True, compression='gzip')
    n = len(meta)
    for name, value in [('power', np.full((n, 60), 2, dtype=np.float32)),
                        ('calendar', np.zeros((n, 4), dtype=np.float32)),
                        ('target', np.ones((n, 10), dtype=np.float32))]:
        np.save(root/f'{name}.npy', value)
    if thermal:
        value = np.ones((n, 60, 2), dtype=np.float32)
        value[0, 0, 0] = np.nan
        np.save(root/'thermal.npy', value)
    info = dict(completed=True, floor=floor, has_thermal=thermal,
        devices=[dict(column=f'AC{i}', zone=2 if i<2 else 1) for i in range(3)],
        groups=[['floor', [0, 1, 2]], ['zone2', [0, 1]]])
    (root/'preparation.json').write_text(json.dumps(info))


class CuBemsInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        packet(self.root)
        self.task = OperationalForecastTask(self.root, 'CU-BEMS', site='floor3')
        self.addCleanup(self.task.base.close)

    def test_typed_inputs_preserve_missingness_and_no_labels(self):
        full = self.task.inputs([0])['native_inputs']
        self.assertEqual(full['power_kw'].shape, (1, 60, 1))
        self.assertFalse(full['numeric_state_observed'][0, 0, 0])
        self.assertTrue(np.isnan(full['numeric_state'][0, 0, 0]))
        self.assertNotIn('target', full)
        self.assertNotIn('y', full)
        self.assertNotIn('numeric_state', self.task.inputs([0], condition='power')['native_inputs'])
        full['power_kw'][:] = 999
        self.assertEqual(self.task.current_power([0])[0][0], 2)
        np.testing.assert_equal(self.task.report_offsets[[0, -1]], [[0, 14], [135, 149]])

    def test_complete_groups_and_energy_units(self):
        prediction = np.full((3, 10), 2.)
        scores = self.task.score(split='later_1', prediction_kw=prediction, require_groups=True)['power'][0]
        device = scores[scores.scale=='device'].set_index('span_minutes')
        np.testing.assert_allclose(device.mse, 1)
        np.testing.assert_allclose(device.energy_ae_kwh, [.5, 1.5, 2.5])
        np.testing.assert_allclose(scores[scores.scale=='floor_total'].mse, 9)
        np.testing.assert_allclose(scores[scores.scale=='zone2_total'].mse, 4)

    def test_incomplete_floor_is_not_resized(self):
        folder = self.root/'partial'; folder.mkdir()
        packet(folder, missing_member=True)
        task = OperationalForecastTask(folder, 'CU-BEMS', site='floor3')
        self.addCleanup(task.base.close)
        scores = task.score(split='later_1', prediction_kw=np.full((2, 10), 2.))['power'][0]
        self.assertNotIn('floor_total', set(scores.scale))
        self.assertIn('zone2_total', set(scores.scale))
        self.assertTrue((scores[scores.scale=='zone2_total'].units == 1).all())

    def test_absent_thermal_is_not_an_imputed_arm(self):
        folder = self.root/'absent'; folder.mkdir()
        packet(folder, floor=1, thermal=False)
        task = OperationalForecastTask(folder, 'CU-BEMS', site='floor1')
        self.addCleanup(task.base.close)
        self.assertEqual(task.contract['available_conditions'], ('power',))
        with self.assertRaises(ValueError): task.inputs(condition='full')
        self.assertNotIn('numeric_state', task.inputs(condition='power')['native_inputs'])

    def test_blackout_hides_only_recent_state(self):
        original = self.task.inputs([1])['native_inputs']
        hidden = self.task.inputs([1], auxiliary_blackout_minutes=5)['native_inputs']
        np.testing.assert_equal(hidden['power_kw'], original['power_kw'])
        self.assertTrue(hidden['numeric_state_observed'][:, :-5].all())
        self.assertFalse(hidden['numeric_state_observed'][:, -5:].any())
        self.assertTrue(np.isnan(hidden['numeric_state'][:, -5:]).all())

    def test_probability_group_counts(self):
        scores = self.task.score(split='later_1', positive_probability=np.full((3, 10), .5))
        np.testing.assert_allclose(scores['probability'][0].brier, .25)
        group = scores['group_activity'][0]
        np.testing.assert_allclose(group[group.scale=='floor_positive_count'].mse, 2.25)
        np.testing.assert_allclose(group[group.scale=='floor_positive_fraction'].mse, .25)

    def test_submission_reordering_and_floor_identity(self):
        path = self.root/'submission.npz'
        prediction = np.repeat(np.arange(3)[:, None], 10, axis=1)
        self.task.submission(path, split='later_1', prediction_kw=prediction)
        with np.load(path, allow_pickle=False) as archive:
            saved = {k: archive[k] for k in archive.files}
        saved['case_id'] = saved['case_id'][::-1]
        saved['prediction_kw'] = saved['prediction_kw'][::-1]
        np.savez_compressed(path, **saved)
        np.testing.assert_equal(self.task.read_submission(path, split='later_1')['prediction_kw'], prediction)
        saved['site'] = 'floor4'
        np.savez_compressed(path, **saved)
        with self.assertRaises(ValueError): self.task.read_submission(path, split='later_1')
        with self.assertRaises(ValueError): CuBemsDataset(self.root, floor=2)

    def test_invalid_indices_and_missing_split(self):
        for ix in ([.5], [-1], [len(self.task)], [True]):
            with self.assertRaises(ValueError): self.task.inputs(ix)
        with self.assertRaises(ValueError): self.task.indices('absent')
        with self.assertRaises(ValueError): CuBemsDataset(self.root, floor=True)
        with self.assertRaises(ValueError): CuBemsDataset(self.root, floor=3.)


if __name__ == '__main__': unittest.main()
