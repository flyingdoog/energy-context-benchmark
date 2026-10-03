import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_local
import pandas as pd
import quickstart_bpa


class BpaQuickstartTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source, self.work = self.root / 'source', self.root / 'run'
        self.source.mkdir()
        for season in quickstart_bpa.SEASONS:
            (self.source / f'{season}2018.zip').write_bytes(b'original fixture')
        self.prepared = {'completed': True, 'seasons': [{'season': s} for s in quickstart_bpa.SEASONS],
                         'split': {'training_cases': 20, 'validation_cases': 20}, 'seconds': 0.1}
        self.prepare = self.patch('prepare_dataset', side_effect=self.fake_prepare)
        self.example = self.patch('example_operational_submission', Mock())
        self.example.main.side_effect = self.fake_example
        self.tasks = {}
        self.task_factory = self.patch('OperationalForecastTask', side_effect=self.fake_task)
        version_patch = patch.object(quickstart_bpa.importlib.metadata, 'version', return_value='test')
        version_patch.start()
        self.addCleanup(version_patch.stop)

    def patch(self, name, *args, **kwargs):
        replacement = patch.object(quickstart_bpa, name, *args, **kwargs)
        result = replacement.start()
        self.addCleanup(replacement.stop)
        return result

    def fake_prepare(self, source, output, workers):
        for season in quickstart_bpa.SEASONS:
            folder = output / season
            (folder / 'grid').mkdir(parents=True, exist_ok=True)
            (folder / 'receipt.json').write_text(json.dumps({'grid_fields': ['power_kw']}))
            for name in ('cases.csv.gz', 'target_kw.npy', 'target_observed_fraction.npy', 'grid/power_kw.npy'):
                (folder / name).write_bytes(b'prepared fixture')
        (output / 'update_validation').mkdir(exist_ok=True)
        for name in ('split.npz', 'split.json'):
            (output / 'update_validation' / name).write_bytes(b'split fixture')
        return self.prepared

    def fake_task(self, packet, source, *, season):
        task = Mock()
        task.indices.return_value = list(range(20))
        task.read_submission.return_value = {'prediction_kw': 'power', 'positive_probability': 'activity'}
        task.score.return_value = {track: (pd.DataFrame([{'method': 'quickstart', 'value': 1}]), None)
                                   for track in quickstart_bpa.TRACKS}
        self.tasks[season] = task
        return task

    def fake_example(self, argv):
        season = argv[argv.index('--season') + 1]
        self.assertEqual(argv[argv.index('--source') + 1], 'BPA')
        self.assertEqual(argv[argv.index('--split') + 1], season)
        folder = Path(argv[argv.index('--output') + 1])
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'receipt.json').write_text(json.dumps({'completed': True, 'source': 'BPA', 'split': season}))
        (folder / 'example_submission.npz').write_bytes(b'submission fixture')
        for track in quickstart_bpa.TRACKS:
            (folder / f'{track}_scores.json').write_text('[]')

    def test_path_objects_prepare_once_score_both_seasons_and_reuse(self):
        record = quickstart_bpa.run(self.source, self.work)
        self.prepare.assert_called_once_with(self.source, self.work / 'dataset', 2)
        self.assertEqual(self.example.main.call_count, 2)
        self.assertEqual(set(record['examples']), {'spring', 'summer'})
        self.assertEqual(record['preparation'], self.prepared)
        self.assertEqual(record['target_report_offsets_minutes'][0], [1, 15])
        self.assertEqual(record['target_report_offsets_minutes'][-1], [136, 150])
        self.assertEqual(record['query_shift_from_original_minutes'], 30)
        self.assertEqual(record['new_ml_fits'], 0)
        self.assertFalse(record['gpu_used'])
        self.assertIn('inflate64', record['versions'])
        for season in quickstart_bpa.EVALUATIONS:
            self.tasks[season].score.assert_called_once_with(split=season, prediction_kw='power',
                positive_probability='activity', method='quickstart', require_groups=True)
        self.assertEqual(quickstart_bpa.run(self.source, self.work), record)
        self.assertEqual(self.prepare.call_count, 1)
        self.assertEqual(self.example.main.call_count, 2)

    def test_interrupted_work_reuses_finished_example(self):
        first = quickstart_bpa.run(self.source, self.work, workers=1)
        (self.work / 'quickstart_receipt.json').unlink()
        second = quickstart_bpa.run(self.source, self.work, workers=1)
        self.assertEqual(first['examples'], second['examples'])
        self.assertEqual(self.prepare.call_count, 2)
        self.assertEqual(self.example.main.call_count, 2)

    def test_completed_run_with_missing_output_is_rejected(self):
        quickstart_bpa.run(self.source, self.work)
        (self.work / 'example/spring/example_submission.npz').unlink()
        with self.assertRaisesRegex(ValueError, 'missing example outputs'):
            quickstart_bpa.run(self.source, self.work)
        self.assertEqual(self.prepare.call_count, 1)

    def test_overlap_nonempty_and_invalid_workers_fail_before_preparing(self):
        for output in (self.source, self.source / 'nested', self.root):
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, 'outside'):
                quickstart_bpa.run(self.source, output)
        self.work.mkdir()
        (self.work / 'user.txt').write_text('keep')
        with self.assertRaisesRegex(ValueError, 'empty'):
            quickstart_bpa.run(self.source, self.work)
        with self.assertRaisesRegex(ValueError, 'one or two'):
            quickstart_bpa.run(self.source, self.root / 'unused', workers=3)
        self.assertEqual((self.work / 'user.txt').read_text(), 'keep')
        self.prepare.assert_not_called()

    def test_missing_archive_and_changed_workflow_are_rejected(self):
        (self.source / 'summer2018.zip').unlink()
        with self.assertRaisesRegex(ValueError, 'summer2018.zip'):
            quickstart_bpa.run(self.source, self.work)
        self.assertFalse(self.work.exists())
        (self.source / 'summer2018.zip').write_bytes(b'original fixture')
        quickstart_bpa.run(self.source, self.work)
        with self.assertRaisesRegex(ValueError, 'another source or worker layout'):
            quickstart_bpa.run(self.source, self.work, workers=1)

    def test_cli_passes_paths_and_single_worker(self):
        quickstart_bpa.main(['--source', str(self.source), '--workdir', str(self.work), '--workers', '1'])
        self.prepare.assert_called_once_with(self.source, self.work / 'dataset', 1)


if __name__ == '__main__':
    unittest.main()
