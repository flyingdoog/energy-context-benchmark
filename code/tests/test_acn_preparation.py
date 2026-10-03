import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_local
from field_control.acn import NUMERIC
from field_control.acn_preparation import read_manifest, build_acn_packet
from field_control import AcnDataset, OperationalForecastTask
from prepare_acn_dataset import prepare
import quickstart_acn


def source_fixture(root, *, incomplete=False, both_sites=False, splits=False):
    source = root / 'source'
    (source / 'sessions').mkdir(parents=True)
    rows = []
    periods = [('train','2019-04-01'), ('validation','2019-07-01'),
               ('autumn','2019-10-01'), ('winter','2020-01-01')] if splits else [('autumn','2019-10-01')]
    for site in ['caltech','office001'] if both_sites else ['caltech']:
        for split, day in periods:
            for station in range(3 if incomplete else 2):
                start = pd.Timestamp(day+'T08:30:00Z')
                index = pd.date_range(start, periods=541, freq='5s')
                frame = pd.DataFrame(float(station+2), index=index, columns=NUMERIC)
                frame['Charging State'] = 'CHARGING'
                frame.iloc[-1, frame.columns.get_loc('Charging State')] = 'UNPLUGGED'
                if incomplete and station == 2:
                    query = pd.Timestamp(day+'T09:00:00Z')
                    frame = frame[~((frame.index >= query) & (frame.index < query+pd.Timedelta(minutes=12)))]
                filename = f'{site}-{station}-{day}.csv.gz'
                frame.to_csv(source / 'sessions' / filename, index_label='Timestamp')
                rows.append(dict(site=site, split=split, station_id=f'{site}-{station}',
                    connection_utc=start.isoformat(), filename=filename,
                    query_start_local=day,
                    query_end_exclusive_local=str((pd.Timestamp(day)+pd.Timedelta(days=1)).date())))
    pd.DataFrame(rows).to_csv(source / 'manifest.csv', index=False)
    return source


class AcnPreparationTests(unittest.TestCase):
    def test_raw_source_needs_no_research_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = source_fixture(Path(tmp))
            self.assertEqual(len(read_manifest(source)), 2)
            receipt = build_acn_packet(source, Path(tmp)/'packet')
            self.assertEqual(receipt['cases'], 2)
            data = AcnDataset(Path(tmp)/'packet')
            self.assertEqual(data.target.shape, (2, 10))
            np.testing.assert_equal(data.target[:, 2:], 0)
            self.assertEqual(data.inputs([0])['power_kw'].shape, (1,60,1))

    def test_absent_source_is_not_silently_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = source_fixture(Path(tmp))
            manifest = pd.read_csv(source/'manifest.csv')
            manifest.loc[0,'filename'] = 'missing.csv.gz'
            manifest.to_csv(source/'manifest.csv',index=False)
            with self.assertRaisesRegex(ValueError, 'Missing source'):
                read_manifest(source)

    def test_manifest_rejects_traversal_duplicate_and_missing_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = source_fixture(Path(tmp))
            original = pd.read_csv(source/'manifest.csv')
            for changed in [original.assign(filename='../outside.csv.gz'),
                            pd.concat([original, original.iloc[:1]]),
                            original.drop(columns=['query_start_local'])]:
                changed.to_csv(source/'manifest.csv', index=False)
                with self.assertRaises(ValueError):
                    read_manifest(source)

    def test_query_and_targets_stay_inside_calendar_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = source_fixture(Path(tmp))
            manifest = pd.read_csv(source/'manifest.csv')
            # 09:00 UTC + 150 min ends at 04:30 local; an earlier block end excludes it.
            manifest['query_end_exclusive_local'] = '2019-10-01 04:00:00'
            manifest.to_csv(source/'manifest.csv', index=False)
            with self.assertRaisesRegex(ValueError, 'No supported'):
                build_acn_packet(source, Path(tmp)/'packet')

    def test_missing_future_member_does_not_shrink_cohort(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = source_fixture(Path(tmp), incomplete=True)
            receipt = build_acn_packet(source, Path(tmp)/'packet')
            self.assertEqual(receipt['cases'], 2)
            self.assertEqual(receipt['query_known_cases'], 3)
            task = OperationalForecastTask(Path(tmp)/'packet', 'ACN', site='caltech')
            scores = task.score(split='autumn', prediction_kw=np.zeros((2,10)))
            self.assertEqual(set(scores['power'][0].scale), {'device'})

    def test_future_values_do_not_enter_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = source_fixture(Path(tmp))
            build_acn_packet(source, Path(tmp)/'before')
            before = AcnDataset(Path(tmp)/'before')
            for path in (source/'sessions').glob('*.csv.gz'):
                frame = pd.read_csv(path, index_col=0)
                future = pd.to_datetime(frame.index, utc=True) >= pd.Timestamp('2019-10-01T09:00:00Z')
                frame.loc[future, list(NUMERIC)] = 999
                frame.to_csv(path, index_label='Timestamp')
            build_acn_packet(source, Path(tmp)/'after')
            after = AcnDataset(Path(tmp)/'after')
            np.testing.assert_equal(before.features(), after.features())
            self.assertFalse(np.array_equal(before.target, after.target))

    def test_nonempty_outputs_and_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = source_fixture(Path(tmp))
            output = Path(tmp)/'output'
            output.mkdir()
            (output/'user.txt').write_text('keep')
            with self.assertRaises(ValueError):
                build_acn_packet(source, output)
            with self.assertRaises(ValueError):
                build_acn_packet(source, source/'packet')
            with self.assertRaises(ValueError):
                quickstart_acn.main(['--source',str(source),'--workdir',str(output)])
            self.assertEqual((output/'user.txt').read_text(), 'keep')

    def test_parallel_merge_preserves_serial_cases_and_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = source_fixture(Path(tmp), both_sites=True, splits=True)
            build_acn_packet(source, Path(tmp)/'serial')
            prepare(source, Path(tmp)/'parallel', workers=2)
            a, b = AcnDataset(Path(tmp)/'serial'), AcnDataset(Path(tmp)/'parallel')
            order = pd.Index(a.meta.case_id).get_indexer(b.meta.case_id)
            self.assertTrue((order >= 0).all())
            np.testing.assert_equal(a.features(order), b.features())
            np.testing.assert_equal(a.target[order], b.target)
            # A completed run is reused without relaunching workers.
            record = prepare(source, Path(tmp)/'parallel', workers=2)
            self.assertEqual(record['cases'], 16)

    def test_incomplete_acquisition_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = source_fixture(Path(tmp))
            (source/'receipt.json').write_text(json.dumps(dict(completed=False, terminal=True)))
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                read_manifest(source)


if __name__ == '__main__':
    unittest.main()
