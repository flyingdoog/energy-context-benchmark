import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import pandas as pd
from field_control import OperationalForecastTask
from field_control.cu_bems_preparation import prepare_cu_bems, source_grid, parse_columns, sensor_columns


def fixture(thermal=True, later2=True):
    pieces = []
    for date in ['2019-04-01', '2019-07-01', '2019-09-01'] + (['2019-11-01'] if later2 else []):
        clock = pd.date_range(pd.Timestamp(date) - pd.Timedelta(hours=1), periods=360, freq='min')
        n = len(clock)
        part = pd.DataFrame({'Date': clock.strftime('%Y-%m-%d %H:%M:%S'),
            'z1_AC1(kW)': np.arange(n) / 100, 'z1_AC2(kW)': np.zeros(n),
            'z2_AC1(kW)': np.ones(n)})
        if thermal:
            # Deliberately reversed source column order must not swap units.
            part['z1_S1(RH%)'] = 60.
            part['z1_S1(degC)'] = 23.
            part.loc[0, 'z1_S1(degC)'] = np.nan
            part['z2_S1(degC)'] = 24.
            part['z2_S1(RH%)'] = 65.
        pieces.append(part)
    return pd.concat(pieces, ignore_index=True)


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def prepare(self, frame=None, floor=3, folder='packet'):
        source = self.root / f'2019Floor{floor}.csv'
        (fixture() if frame is None else frame).to_csv(source, index=False)
        output = self.root / folder
        receipt = prepare_cu_bems(source, output, floor=floor)
        task = OperationalForecastTask(output, 'CU-BEMS', site=f'floor{floor}')
        self.addCleanup(task.base.close)
        return receipt, task

    def test_original_csv_roundtrip_and_observation_masks(self):
        receipt, task = self.prepare()
        self.assertEqual(receipt['groups'], [('floor', [0, 1, 2]), ('zone1', [0, 1])])
        self.assertEqual(receipt['sensor_mapping'][1], ['z1_S1(degC)', 'z1_S1(RH%)'])
        self.assertEqual([r['cases'] for r in receipt['support']], [18] * 4)
        ix = task.indices('train')
        values = task.inputs(ix)['native_inputs']
        self.assertFalse(values['numeric_state_observed'][0, 0, 0])
        self.assertEqual(values['numeric_state'][0, 1, 0], 23.)
        self.assertEqual(values['numeric_state'][0, 1, 1], 60.)
        self.assertEqual(values['power_kw'][0, -1, 0], np.float32(.59))
        self.assertAlmostEqual(float(task.targets(ix)[0, 0]), .67, places=6)
        self.assertNotIn('target', values)
        self.assertNotIn('y', values)
        self.assertTrue(np.isfinite(task.targets(ix)).all())

    def test_future_sensor_values_do_not_enter_query_inputs(self):
        raw = fixture()
        raw.loc[raw.Date >= '2019-04-01 00:00:00', 'z1_S1(degC)'] = 999.
        _, task = self.prepare(raw)
        at_origin = task.base.meta.index[(task.base.meta['query'] == '2019-04-01 00:00:00') &
                                         (task.base.meta.device_index == 0)].to_numpy()
        inputs = task.inputs(at_origin)['native_inputs']
        self.assertLess(np.nanmax(inputs['numeric_state'][:, :, 0]), 999.)

    def test_absent_floor_sensor_fields_do_not_create_zero_sensors(self):
        receipt, task = self.prepare(fixture(thermal=False), floor=1)
        self.assertFalse(receipt['has_thermal'])
        self.assertFalse((self.root/'packet/thermal.npy').exists())
        self.assertEqual(task.contract['available_conditions'], ('power',))
        self.assertNotIn('numeric_state', task.inputs([0], condition='power')['native_inputs'])

    def test_missing_power_does_not_resize_fixed_floor(self):
        raw = fixture()
        raw.loc[raw.Date == '2019-09-01 00:00:00', 'z2_AC1(kW)'] = np.nan
        receipt, task = self.prepare(raw)
        count = len(task.indices('later_1'))
        self.assertLess(count, 18)
        score = task.score(split='later_1', prediction_kw=np.zeros((count, 10)))['power'][0]
        floor = score[(score.scale == 'floor_total') & (score.span_minutes == 30)].iloc[0]
        # Missing minute 00:00 affects queries 00:00, 00:30 and 01:00.
        self.assertEqual(floor.units, 3)
        self.assertEqual(receipt['groups'][0][1], [0, 1, 2])

    def test_truncated_floor_retains_empty_later_period(self):
        raw = fixture(later2=False)
        raw.loc[len(raw)] = {name: ('2' if name == 'Date' else np.nan) for name in raw.columns}
        receipt, task = self.prepare(raw, floor=6)
        self.assertEqual(receipt['source_quality']['malformed_rows'][0]['date_text'], '2')
        self.assertNotIn('later_2', {x['split'] for x in receipt['support']})
        with self.assertRaises(ValueError):
            task.indices('later_2')

    def test_partial_sensor_schema_is_not_silently_filled(self):
        raw = fixture().drop(columns=['z2_S1(degC)', 'z2_S1(RH%)'])
        with self.assertRaisesRegex(ValueError, 'partially absent'):
            self.prepare(raw)
        with self.assertRaisesRegex(ValueError, 'ambiguous/incomplete'):
            sensor_columns(parse_columns(['z1_S1(degC)', 'z1_S2(RH%)']), 1)

    def test_wrong_units_and_floor_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'kW'):
            self.prepare(fixture().rename(columns={'z1_AC1(kW)': 'z1_AC1(W)'}))
        source = self.root/'2019Floor3.csv'
        fixture().to_csv(source, index=False)
        with self.assertRaisesRegex(ValueError, 'filename'):
            prepare_cu_bems(source, self.root/'wrong', floor=2)
        with self.assertRaises(ValueError):
            prepare_cu_bems(source, self.root/'wrong', floor=True)

    def test_existing_output_is_preserved(self):
        source = self.root/'2019Floor3.csv'
        fixture().to_csv(source, index=False)
        output = self.root/'keep'; output.mkdir()
        sentinel = output/'keep.txt'; sentinel.write_text('existing user result')
        with self.assertRaises(FileExistsError):
            prepare_cu_bems(source, output, floor=3)
        self.assertEqual(sentinel.read_text(), 'existing user result')

    def test_clock_duplicates_nonminute_and_unknown_tail_rejected(self):
        base = fixture(thermal=False)
        with self.assertRaises(ValueError):
            source_grid(pd.concat([base.iloc[:1], base], ignore_index=True))
        base.loc[0, 'Date'] = '2019-03-31 23:00:01'
        with self.assertRaises(ValueError):
            source_grid(base)
        base = fixture()
        base.loc[len(base)] = {name: ('2' if name == 'Date' else np.nan) for name in base.columns}
        with self.assertRaisesRegex(ValueError, 'only for Floor 6'):
            self.prepare(base, floor=3)

    def test_cli_works_from_another_directory_without_model_libraries(self):
        source = self.root/'2019Floor3.csv'; fixture().to_csv(source, index=False)
        script = Path(__file__).resolve().parents[1]/'scripts/prepare_cu_bems.py'
        result = subprocess.run([sys.executable, str(script), '--csv', str(source),
            '--floor', '3', '--output', str(self.root/'cli')], cwd=self.root,
            check=False, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        message = json.loads(result.stdout)
        self.assertTrue(message['completed'])
        self.assertEqual(message['model_libraries_loaded'], [])
        self.assertEqual(message['cases'], 72)
        example = script.with_name('example_operational_submission.py')
        result = subprocess.run([sys.executable, str(example), '--source', 'CU-BEMS',
            '--site', 'floor3', '--data', str(self.root/'cli'), '--split', 'later_1',
            '--output', str(self.root/'example')], cwd=self.root,
            check=False, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['model_libraries_loaded'], [])
        self.assertTrue((self.root/'example/group_activity_scores.json').is_file())


if __name__ == '__main__':
    unittest.main()
