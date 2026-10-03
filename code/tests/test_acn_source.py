"""Source planning, bounded acquisition and completion semantics, without network."""
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
import pandas as pd
from field_control import acn_source as source
from field_control.acn_preparation import read_manifest


def entry(date, station='2-39-78-367', size=40):
    return dict(path=f'{station}-{date}T12-00-00-000001.csv.gz', type='blob', size=size)


def trees(entries):
    return {'caltech': {'tree': entries}, 'office001': {'tree': []}}


class SourceTests(unittest.TestCase):
    def test_calendar_boundaries_and_overlapping_contexts(self):
        rows = source.calendar_rows(trees([entry('2019-03-17'), entry('2019-03-18'),
            entry('2019-06-17'), entry('2019-06-30'), entry('2019-07-31'), entry('2019-08-01')]))
        train = [row for row in rows if row['split'] == 'train']
        validation = [row for row in rows if row['split'] == 'validation']
        self.assertEqual([r['local_date'] for r in train], ['2019-03-18', '2019-06-17', '2019-06-30'])
        self.assertEqual([r['local_date'] for r in validation], ['2019-06-17', '2019-06-30', '2019-07-31'])
        self.assertEqual(len(source.unique_files(rows)), 4)
        self.assertTrue(all(r['query_start_local'] == '2019-04-01' for r in train))
        self.assertTrue(all(r['query_end_exclusive_local'] == '2019-08-01' for r in validation))

    def test_connection_day_uses_local_timezone(self):
        item = entry('2019-03-18')
        item['path'] = item['path'].replace('T12-', 'T01-')
        self.assertEqual(source.calendar_rows(trees([item])), [])

    def test_source_identity_and_size_checks(self):
        for item in [dict(path='../bad.csv.gz', type='blob', size=40),
                     entry('2019-04-01', size=source.FILE_LIMIT + 1),
                     dict(entry('2019-04-01'), type='tree')]:
            with self.subTest(item=item), self.assertRaises(ValueError):
                source.calendar_rows(trees([item]))

    def test_filename_collision_rejected(self):
        rows = source.calendar_rows(trees([entry('2019-04-01')]))
        with self.assertRaises(ValueError):
            source.unique_files(rows + [dict(rows[0], source_path='another/path')])

    def test_sample_preserves_site_block_coverage(self):
        entries = [entry(date) for date in ['2019-03-18', '2019-05-01', '2019-06-30',
                   '2019-07-15', '2019-07-31', '2019-09-17', '2019-10-15', '2019-10-31',
                   '2019-12-18', '2020-01-15', '2020-01-31']]
        rows = source.calendar_rows(trees(entries))
        sample = source.sample_files(rows)
        self.assertLessEqual(len(sample), 8)
        self.assertIn('2019-03-18', {r['local_date'] for r in sample})
        self.assertNotIn('2019-05-01', {r['local_date'] for r in sample})
        self.assertLess(len(sample), len(source.unique_files(rows)))

    def test_nonempty_directory_protected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / 'original.txt').write_text('keep')
            with self.assertRaises(ValueError):
                source._workspace(path)
            self.assertEqual((path / 'original.txt').read_text(), 'keep')

    def test_other_revision_protected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = source._workspace(temporary)
            marker = path / 'source_plan.json'
            data = json.loads(marker.read_text())
            data['revision'] = 'other'
            marker.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                source._workspace(path)

    def test_truncated_index_not_saved(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = source._workspace(temporary)
            with patch.object(source, '_get', return_value=json.dumps(dict(sha=source.REVISION,
                       truncated=True, tree=[])).encode()):
                with self.assertRaises(ValueError):
                    source._tree(path, source.REVISION, 'root')
            self.assertFalse((path / 'metadata/root.json').exists())

    def test_access_failure_not_retried(self):
        for code in (401, 403, 429):
            with self.subTest(code=code), patch.object(source, 'urlopen',
                    side_effect=HTTPError('https://example.test', code, 'blocked', {}, None)) as request:
                with self.assertRaises(HTTPError):
                    source._get('https://example.test', 100)
                self.assertEqual(request.call_count, 1)

    def test_optional_provider_index_must_match_reference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = source._workspace(root / 'run')
            cached = root / 'index'
            cached.mkdir()
            index = dict(sha=source.REVISION, truncated=False, tree=[])
            (cached / 'root.json').write_text(json.dumps(index))
            (cached / 'other.json').write_text(json.dumps(index))
            with patch.object(source, '_get', side_effect=AssertionError('No network expected')):
                self.assertEqual(source._tree(output, source.REVISION, 'root', cached), index)
                with self.assertRaisesRegex(ValueError, 'unexpected revision'):
                    source._tree(output, 'other-revision', 'other', cached)

    def test_transient_failure_has_two_attempt_bound(self):
        with (patch.object(source, 'urlopen', side_effect=HTTPError(
                'https://example.test', 503, 'unavailable', {}, None)) as request,
                patch.object(source.time, 'sleep')):
            with self.assertRaises(HTTPError):
                source._get('https://example.test', 100)
            self.assertEqual(request.call_count, 2)

    def test_partial_sample_resume_then_full(self):
        content = gzip.compress(b'time,power\n0,1\n')
        rows = source.calendar_rows(trees([entry(date, size=len(content))
            for date in ['2019-04-01', '2019-04-02', '2019-04-03']]))
        with tempfile.TemporaryDirectory() as temporary:
            path = source._workspace(temporary)
            pd.DataFrame(rows).to_csv(path / 'manifest.csv', index=False)
            with patch.object(source, 'build_plan'), patch.object(source, '_get', return_value=content) as get:
                record = source.acquire(path, sample=True, workers=2)
                self.assertTrue(record['selected_completed'])
                self.assertFalse(record['completed'])
                self.assertEqual(record['available_files'], 2)
                self.assertEqual(get.call_count, 2)
                with self.assertRaisesRegex(ValueError, 'incomplete source'):
                    read_manifest(path)
                record = source.acquire(path, sample=True, workers=2)
                self.assertEqual(record['downloaded'], 0)
                self.assertEqual(get.call_count, 2)
                record = source.acquire(path, workers=2)
                self.assertTrue(record['completed'])
                self.assertEqual(record['downloaded'], 1)
                self.assertEqual(len(read_manifest(path)), 3)

    def test_bad_source_response_leaves_no_final_file(self):
        rows = source.calendar_rows(trees([entry('2019-04-01', size=3)]))
        with tempfile.TemporaryDirectory() as temporary:
            path = source._workspace(temporary)
            pd.DataFrame(rows).to_csv(path / 'manifest.csv', index=False)
            with patch.object(source, 'build_plan'), patch.object(source, '_get', return_value=b'bad'):
                record = source.acquire(path, workers=1)
            self.assertFalse(record['completed'])
            self.assertFalse(record['selected_completed'])
            self.assertEqual(record['errors'], 1)
            self.assertFalse((path / 'sessions' / rows[0]['filename']).exists())


if __name__ == '__main__':
    unittest.main()
