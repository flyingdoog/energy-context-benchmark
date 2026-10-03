import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_local
import numpy as np
import pandas as pd
from field_control.epfl_source import read_measurements, link_sessions, extract_source, download_source, REVISION
from field_control.epfl_preparation import build_packet
import quickstart_epfl


def records(start='2022-11-01', n=32):
    time = pd.date_range(start, periods=n, freq='min')
    data = pd.DataFrame({'Date and time': time, 'CCS': 'CCS2', 'Session': 1,
        'P (W)': 1000., 'Preq (W)': 1500., 'Pset (W)': 2000.,
        'SOC (%)': 50., 'TotalCapacity': 10000., 'RemainingCapacity': 5000.,
        'BulkCapacity': 8000., 'Energy (Wh)': 100.,
        'Ctrl_t (0 = False, 1 = True)': 0., 'Ctrl_s (0 = False, 1 = True)': 1.})
    sessions = pd.DataFrame({'CCS': ['CCS2'], 'Session': [1130],
        'Arrival': [time[0]], 'Departure': [time[-1]]})
    return data, sessions


class SourceTests(unittest.TestCase):
    def test_stream_filter_keeps_true_zero_and_excludes_empty_calendar(self):
        class Sheet:
            def iter_rows(self, **kwargs):
                return iter([('Session', 'P (W)'), (None, None), (1, 0), (1, 1000)])
        class Book(dict):
            closed = False
            def close(self):
                self.closed = True
        book = Book(CCS1=Sheet(), CCS2=Sheet())
        with patch('openpyxl.load_workbook', return_value=book):
            data, support = read_measurements('unused.xlsx')
        self.assertEqual(len(data), 4)
        self.assertEqual((data['P (W)'] == 0).sum(), 2)
        self.assertEqual(sum(s['calendar_rows'] for s in support), 6)
        self.assertTrue(book.closed)

    def test_numbering_link_uses_arrival_not_numeric_offset(self):
        data, sessions = records()
        original = data.copy()
        linked, support, cross = link_sessions(data, sessions)
        self.assertEqual(support.key.tolist(), ['CCS2:1'])
        self.assertEqual(cross.key_session.tolist(), ['CCS2:1130'])
        self.assertTrue((support.end_offset_seconds == 0).all())
        pd.testing.assert_frame_equal(data, original)

    def test_unconfirmed_departure_and_duplicate_reports_rejected(self):
        data, sessions = records()
        sessions.loc[0, 'Departure'] += pd.Timedelta(minutes=1)
        with self.assertRaisesRegex(ValueError, 'endpoints'):
            link_sessions(data, sessions)
        data, sessions = records()
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            link_sessions(pd.concat([data, data.iloc[:1]]), sessions)

    def build(self, data, sessions, dest):
        raw = dest / 'raw'
        raw.mkdir()
        for name in ['Measurement_data.xlsx', 'Session_data.xlsx']:
            (raw / name).touch()
        with patch('field_control.epfl_source.read_measurements', return_value=(data, [{'sheet': 'CCS2', 'calendar_rows': len(data)+2, 'active_rows': len(data)}])), \
             patch('field_control.epfl_source.pd.read_excel', return_value=sessions):
            receipt = extract_source(raw, dest / 'cache')
        return raw, receipt, build_packet(dest / 'cache')

    def test_raw_to_packet_preserves_post_departure_rule_and_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, sessions = records()
            raw, receipt, (meta, x, y) = self.build(data, sessions, Path(tmp))
            self.assertEqual(receipt['empty_calendar_rows'], 2)
            self.assertEqual(x.shape, (1, 94))
            np.testing.assert_allclose(y[0, :, 0], [1, 1, 1, 1, .4, 0])
            with patch('field_control.epfl_source.read_measurements', side_effect=AssertionError('must reuse')):
                reused = extract_source(raw, Path(tmp) / 'cache')
            self.assertEqual(reused, receipt)

    def test_internal_gap_excludes_case_not_filled_with_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, sessions = records()
            _, _, (meta, _, _) = self.build(data.drop(17), sessions, Path(tmp))
            self.assertTrue(meta.empty)

    def test_target_crossing_split_is_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, sessions = records('2022-12-31 23:40')
            _, _, (meta, _, _) = self.build(data, sessions, Path(tmp))
            self.assertTrue(meta.empty)

    def test_future_changes_do_not_enter_historical_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, sessions = records()
            one = Path(tmp) / 'one'
            two = Path(tmp) / 'two'
            one.mkdir(); two.mkdir()
            _, _, (_, x, y) = self.build(data, sessions, one)
            data.loc[10:, 'P (W)'] = 9000
            data.loc[10:, 'SOC (%)'] = 99
            data['Ctrl_s (0 = False, 1 = True)'] = 0
            _, _, (_, changed_x, changed_y) = self.build(data, sessions, two)
            np.testing.assert_array_equal(x, changed_x)
            self.assertFalse(np.array_equal(y, changed_y))

    def test_existing_files_are_not_overwritten_or_downloaded_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp)
            (raw / 'README.md').touch()
            with patch('field_control.epfl_source.urlopen', side_effect=AssertionError('no network')):
                with self.assertRaisesRegex(ValueError, 'inspection'):
                    download_source(raw)

    def test_quickstart_rejects_unrelated_nonempty_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'user-file.txt').touch()
            with self.assertRaisesRegex(ValueError, 'empty directory'):
                quickstart_epfl.main(['--download', '--workdir', str(root)])
            self.assertTrue((root / 'user-file.txt').exists())


if __name__ == '__main__':
    unittest.main()
