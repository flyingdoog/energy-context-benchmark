import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import run_local
import run_record_comparison as runner
from field_control.acn_preparation import build_acn_packet
from field_control.record_reference_inputs import prepare_inputs, bpa_names_and_moved, typed, describe_inputs
from field_control.record_reference_scores import aggregate_errors, paired_effects, RAW_ARMS
from test_acn_preparation import source_fixture


def daily_fixture():
    return pd.DataFrame([dict(source='ACN', site='test', period=period, scale='cohort_total',
        span_minutes=span, date=date, method=method, mse=error, mae=error/2, energy_ae_kwh=error/3)
        for period in ['validation', 'autumn', 'winter']
        for span in [30, 90, 150] for date in ['2019-01-01', '2019-01-09']
        for method, error in [('Power only', 10), ('Paired', 8),
            *[(f'Fixed {s}', 14+i) for i, s in enumerate([2027, 2028, 2029])],
            *[(f'Retrained {s}', 9+i) for i, s in enumerate([2027, 2028, 2029])]]])


class RecordReferenceTests(unittest.TestCase):
    def test_complete_case_and_input_only_donor_support(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            source = source_fixture(root, both_sites=True, splits=True, incomplete=True)
            build_acn_packet(source, root/'data')
            for site in ['caltech', 'office001']:
                cache = root/site
                desc = prepare_inputs(root/'data', 'ACN', site, cache, source)
                self.assertEqual(len(desc['support']), 4)
                self.assertTrue(desc['required_original_source'])
                for period in ['train', 'validation', 'autumn', 'winter']:
                    meta, x, rows, y, moved, donors = runner.load_period(cache, period)
                    self.assertEqual((len(meta), len(rows)), (3, 2))
                    self.assertEqual(y.shape, (2, 10))
                    for donor in donors.values():
                        self.assertTrue((donor != np.arange(3)).all())
                        self.assertTrue(np.any(meta.native_index.iloc[donor[rows]].to_numpy() < 0))
                    z = runner.input_view(x, rows, moved, donors[2027])
                    keep = np.setdiff1d(np.arange(x.shape[1]), moved)
                    np.testing.assert_equal(z[:, keep], x[np.ix_(rows, keep)])
                    np.testing.assert_equal(z[:, moved], x[np.ix_(donors[2027][rows], moved)])
                    p = runner.input_view(x, rows, moved, power_only=True)
                    np.testing.assert_equal(p[:, moved], 0)
                    np.testing.assert_equal(p[:, keep], x[np.ix_(rows, keep)])
                    del x
                prepare_inputs(root/'data', 'ACN', site, cache, source)
                with self.assertRaisesRegex(ValueError, 'different source'):
                    prepare_inputs(root/'data', 'ACN', 'office001' if site == 'caltech' else 'caltech', cache, source)

    def test_missing_original_source_cannot_shrink_donors(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            source = source_fixture(root, splits=True, incomplete=True)
            build_acn_packet(source, root/'data')
            with self.assertRaisesRegex(ValueError, 'input-only donors'):
                prepare_inputs(root/'data', 'ACN', 'caltech', root/'cache')

    def test_model_free_preview_has_no_side_effects(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            source = source_fixture(root, splits=True)
            build_acn_packet(source, root/'data')
            with patch.object(runner, 'prepare_inputs', side_effect=AssertionError('preview must not prepare')):
                desc = runner.main(['--data', str(root/'data'), '--source', 'ACN', '--site', 'caltech', '--output', str(root/'out')])
            self.assertFalse((root/'out').exists())
            self.assertFalse(desc['required_original_source'])
            self.assertEqual(desc['fitted_heads'], 50)

    def test_typed_bpa_features_keep_power_duplicates_and_profiles(self):
        names, moved = bpa_names_and_moved()
        self.assertEqual(len(names), 4064)
        self.assertEqual(len(moved), 3782)
        for name in ['snapshot:power_kw', 'snapshot:power_kw_observed',
                     'history0:power_observed_fraction', 'snapshot:previous_week_lead_11']:
            self.assertNotIn(names.index(name), moved)
        values = typed(np.asarray([[2., 1., 0.], [255., np.nan, 2.]]), ['oper_state', 'running', 'opt_out'])
        self.assertEqual(values.shape, (2, 21))
        self.assertEqual(values[0, 2], 1)
        self.assertEqual(values[1, 13], 1)
        self.assertEqual(values[1, -1], 1)
        self.assertEqual(values[1, 15:18].sum(), 0)

    def test_errors_are_averaged_and_both_metrics_retained(self):
        daily = aggregate_errors(daily_fixture())
        self.assertEqual(set(daily.method), {'Power only', 'Paired', 'Fixed', 'Retrained'})
        self.assertTrue((daily.loc[daily.method == 'Fixed', 'mse'] == 15).all())
        self.assertTrue((daily.loc[daily.method == 'Retrained', 'mse'] == 10).all())
        effects = paired_effects(daily, draws=100)
        self.assertEqual(len(effects), 24)
        self.assertEqual(set(effects.span_minutes), {0, 30, 90, 150})
        self.assertTrue((effects.frozen_penalty_pct == 87.5).all())
        self.assertTrue((effects.retrained_penalty_pct == 25).all())
        self.assertTrue((effects.relearning_recovery_pct == 62.5).all())
        self.assertTrue((effects.weeks == 2).all())

    def test_missing_mapping_or_date_rejected(self):
        raw = daily_fixture()
        for changed in [raw.iloc[:-1], pd.concat([raw, raw.iloc[:1]])]:
            with self.assertRaises(ValueError):
                aggregate_errors(changed)

    def test_one_week_has_no_invented_interval(self):
        raw = daily_fixture()
        daily = aggregate_errors(raw[raw.date == '2019-01-01'])
        effects = paired_effects(daily, draws=100)
        self.assertTrue(effects.frozen_penalty_low.isna().all())
        self.assertTrue((effects.weeks == 1).all())

    def test_run_smoke_exports_complete_named_splits(self):
        try:
            import lightgbm
        except ImportError:
            self.skipTest('Optional tree dependency not installed')
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            source = source_fixture(root, splits=True)
            build_acn_packet(source, root/'data')
            args = ['--data', str(root/'data'), '--source', 'ACN', '--site', 'caltech',
                    '--output', str(root/'run'), '--run', '--smoke-train-cases', '2']
            record = runner.main(args)
            self.assertTrue(record['completed'])
            self.assertFalse(record['paper_result_eligible'])
            self.assertEqual(record['fitted_heads_this_run'], 50)
            self.assertEqual(len(list((root/'run'/'predictions').glob('*/*.npz'))), 24)
            again = runner.main(args)
            self.assertEqual(again, record)
            with self.assertRaisesRegex(ValueError, 'another run'):
                runner.main(args[:-2])


if __name__ == '__main__':
    unittest.main()
