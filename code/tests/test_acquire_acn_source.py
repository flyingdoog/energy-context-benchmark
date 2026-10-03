import contextlib
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import acquire_acn_source


class AcnSourceCliTests(unittest.TestCase):
    def invoke(self, *arguments):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = acquire_acn_source.main(list(arguments))
        return result, json.loads(output.getvalue())

    def test_default_only_builds_plan_and_returns_original_receipt(self):
        receipt = {'plan_completed': True, 'revision': 'fixed-revision', 'planned_files': 7761,
                   'manifest_contexts': 8246, 'session_downloads': 0,
                   'sessions': [{'filename': 'large-detail'}]}
        with patch.object(acquire_acn_source, 'build_plan', return_value=receipt) as plan, \
             patch.object(acquire_acn_source, 'acquire') as download:
            result, printed = self.invoke('--output', 'source with spaces')
        plan.assert_called_once_with(Path('source with spaces').resolve())
        download.assert_not_called()
        self.assertIs(result, receipt)
        self.assertEqual(printed['mode'], 'plan')
        self.assertEqual(printed['manifest_contexts'], 8246)
        self.assertEqual(printed['session_downloads'], 0)
        self.assertNotIn('sessions', printed)

    def test_sample_is_explicit_and_passes_worker_limit(self):
        receipt = {'completed': False, 'selected_completed': True, 'phase': 'sample'}
        with patch.object(acquire_acn_source, 'build_plan') as plan, \
             patch.object(acquire_acn_source, 'acquire', return_value=receipt) as download:
            result, printed = self.invoke('--output', 'sample-source', '--sample', '--workers', '1')
        plan.assert_not_called()
        download.assert_called_once_with(Path('sample-source').resolve(), sample=True, workers=1)
        self.assertIs(result, receipt)
        self.assertEqual(printed['mode'], 'sample')
        self.assertFalse(printed['completed'])
        self.assertTrue(printed['selected_completed'])

    def test_plan_passes_explicit_index_source_as_a_path(self):
        with patch.object(acquire_acn_source, 'build_plan', return_value={'plan_completed': True}) as plan, \
             patch.object(acquire_acn_source, 'acquire') as download:
            self.invoke('--output', 'planned-source', '--index-source', 'provider indexes')
        plan.assert_called_once_with(Path('planned-source').resolve(),
                                     index_source=Path('provider indexes').resolve())
        download.assert_not_called()

    def test_download_modes_pass_explicit_index_source(self):
        for flag, sample in (('--sample', True), ('--download', False)):
            with self.subTest(flag=flag), \
                 patch.object(acquire_acn_source, 'acquire',
                              return_value={'selected_completed': True}) as download:
                self.invoke('--output', 'resumed-source', flag, '--index-source', 'provider indexes')
            download.assert_called_once_with(Path('resumed-source').resolve(), sample=sample,
                workers=4, index_source=Path('provider indexes').resolve())

    def test_full_download_is_explicit_and_defaults_to_four_workers(self):
        with patch.object(acquire_acn_source, 'build_plan') as plan, \
             patch.object(acquire_acn_source, 'acquire',
                          return_value={'completed': True, 'selected_completed': True}) as download:
            _, printed = self.invoke('--output', 'full-source', '--download')
        plan.assert_not_called()
        download.assert_called_once_with(Path('full-source').resolve(), sample=False, workers=4)
        self.assertEqual(printed['mode'], 'download')

    def test_conflicting_modes_or_invalid_workers_do_not_call_source_module(self):
        invalid = [('--sample', '--download'), ('--workers', '0'), ('--workers', '5')]
        with patch.object(acquire_acn_source, 'build_plan') as plan, \
             patch.object(acquire_acn_source, 'acquire') as download:
            for extra in invalid:
                with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), \
                     self.assertRaises(SystemExit):
                    acquire_acn_source.main(['--output', 'unused', *extra])
        plan.assert_not_called()
        download.assert_not_called()

    def test_output_is_required(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            acquire_acn_source.main([])

    def test_source_failure_propagates_without_success_json(self):
        output = io.StringIO()
        with patch.object(acquire_acn_source, 'build_plan', side_effect=ValueError('unrelated output')), \
             contextlib.redirect_stdout(output), self.assertRaisesRegex(ValueError, 'unrelated output'):
            acquire_acn_source.main(['--output', 'unused'])
        self.assertEqual(output.getvalue(), '')

    def test_incomplete_selected_download_prints_receipt_and_exits_nonzero(self):
        receipt = {'completed': False, 'selected_completed': False, 'errors': 1, 'stopped': True}
        for flag in ('--sample', '--download'):
            output = io.StringIO()
            with self.subTest(flag=flag), \
                 patch.object(acquire_acn_source, 'acquire', return_value=receipt), \
                 contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as failure:
                acquire_acn_source.main(['--output', 'unused', flag])
            self.assertEqual(failure.exception.code, 1)
            printed = json.loads(output.getvalue())
            self.assertFalse(printed['selected_completed'])
            self.assertEqual(printed['errors'], 1)


if __name__ == '__main__':
    unittest.main()
