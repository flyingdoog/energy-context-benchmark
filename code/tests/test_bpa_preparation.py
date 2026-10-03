from pathlib import Path
import gc
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_local
from field_control import bpa_preparation as prep
from field_control import OperationalForecastTask


class BpaPreparationTests(unittest.TestCase):
    def archive(self, root, power=(1000., 1000., 2000., 2000.)):
        frame = pd.DataFrame(dict(local_time=['2018-01-01T00:00:30Z',
            '2018-01-01T00:00:30Z', '2018-01-01T00:01:30Z', '2018-01-01T00:02:30Z'],
            alias=['BPAa1'] * 4, curr_watts=power, curr_curtail_id=[1]*4,
            curr_curtail_type=[5]*4, curr_curtail_numeric=[0.]*4,
            wifi_online=[1]*4, opt_out=[0]*4, present_whr=[10.]*4,
            total_whr=[20.]*4, oper_state=[2.]*4, running=[1.]*4))
        with zipfile.ZipFile(root/'spring2018.zip', 'w', zipfile.ZIP_DEFLATED) as z:
            z.writestr('spring2018.csv', frame.to_csv(index=False))
        return frame

    def test_archive_dimensions_without_saved_census(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            self.archive(root)
            info = prep.scan_archive(root, 'spring')
            self.assertEqual((info['rows'], info['devices']), (4, 1))
            self.assertEqual(info['last_timestamp'], '2018-01-01T00:02:30Z')
            out = root/'metadata_output'; out.mkdir()
            prep.archive_info(root, out, 'spring')
            self.assertEqual(prep.archive_info(root, out, 'spring'), info)

    def test_identical_and_conflicting_device_reports(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            self.archive(root)
            arrays, seen, aliases, first, info = prep.read_season(root, 'spring')
            np.testing.assert_array_equal(seen, [[True, True, True]])
            self.assertEqual(info['identical_duplicate_rows'], 1)
            self.archive(root, power=(1000., 3000., 2000., 2000.))
            arrays, seen, aliases, first, info = prep.read_season(root, 'spring')
            np.testing.assert_array_equal(seen, [[False, True, True]])
            self.assertEqual(info['conflicting_device_minutes'], 1)

    def test_original_target_bins_and_refreshed_query(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            source = root/'source'; source.mkdir()
            out = root/'output'; out.mkdir()
            length = 11520; onset = 10800
            arrays = {key: np.full((2, length), 1, dtype=dtype)
                for key, dtype, fill in prep.EXTRA_FIELDS.values()}
            arrays['power'][:] = 1000.
            arrays['power'][:, onset] = 9000.
            arrays['power'][:, onset+1:] = 2000.
            arrays['command_id'][:, onset:] = 2
            arrays['command_type'][:] = 5
            seen = np.ones((2, length), dtype=bool)
            first = pd.Timestamp('2018-01-01T08:00:30Z')
            result = (arrays, seen, ['BPAa1', 'BPAa2'], first,
                      dict(raw_rows=2*length, devices=2, minutes=length))
            with patch.object(prep, 'archive_info', return_value={}), patch.object(prep, 'read_season', return_value=result):
                receipt = prep.build_season(source, out, 'spring')
            self.assertEqual(receipt['cases'], 2)
            task = OperationalForecastTask(out, 'BPA', season='spring')
            np.testing.assert_allclose(task.targets(), 2.)
            np.testing.assert_allclose(task.current_power()[0], 2.)
            self.assertEqual(task.targets().shape, (2, 10))
            self.assertEqual(task.contract['query_shift_from_original_minutes'], 30)
            np.testing.assert_array_equal(task.contract['report_offsets_minutes'][0], [1, 15])
            saved = prep.build_season(source, out, 'spring')
            self.assertEqual(saved['cases'], 2)
            del task
            gc.collect()

    def test_original_winter_gap(self):
        dates = pd.date_range('2018-01-01', periods=18, tz='America/Los_Angeles')
        stamps = list(dates + pd.Timedelta(hours=12)) + [dates[4]-pd.Timedelta(minutes=180)]
        meta = pd.DataFrame(dict(origin_utc=[x.tz_convert('UTC').isoformat() for x in stamps],
            local_date=[str(x.date()) for x in stamps]))
        train, validation, cutoff = prep.split_winter(meta)
        self.assertEqual(len(validation), 14)
        self.assertNotIn(18, train)
        self.assertNotIn(18, validation)
        origin = pd.to_datetime(meta.origin_utc, utc=True)
        self.assertTrue((origin.iloc[train] + pd.Timedelta(minutes=180) < pd.Timestamp(cutoff)).all())

    def test_source_directory_is_not_an_output(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            with self.assertRaisesRegex(ValueError, 'separate directories'):
                prep.prepare(root, root/'output', seasons=['spring'])


if __name__ == '__main__':
    unittest.main()
