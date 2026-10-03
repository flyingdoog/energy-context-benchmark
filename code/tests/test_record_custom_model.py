import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import run_local
import example_record_model as example
import run_custom_record_comparison as runner
from field_control.record_comparison import RecordInputView, fit_record_models, predict_record_models, load_record_cache
from field_control.acn_preparation import build_acn_packet
from field_control.record_reference_inputs import prepare_inputs
from test_acn_preparation import source_fixture


class SpyModel(example.LinearExample):
    def __init__(self, seed=2027):
        super().__init__(seed)
        self.fits = self.predicts = 0

    def fit(self, x, y):
        self.fits += 1
        return super().fit(x, y)

    def predict(self, x):
        self.predicts += 1
        return super().predict(x)


class CustomRecordTests(unittest.TestCase):
    def test_five_fits_same_initialization_and_fixed_model_reuse(self):
        train, _ = example.synthetic_view('train', 1)
        view, _ = example.synthetic_view('later_1', 2)
        seeds = []
        def factory(seed):
            seeds.append(seed)
            return SpyModel(seed)
        models = fit_record_models(train, factory)
        packets = list(predict_record_models(models, view, batch_size=13))
        self.assertEqual(seeds, [2027]*5)
        self.assertEqual(sum(m.fits for m in models.values()), 5)
        self.assertEqual(len(packets), 8)
        self.assertEqual(models['paired'].predicts, 12)
        self.assertEqual(models['power'].predicts, 3)
        self.assertTrue(all(p.shape == (36, 10) for _, p in packets))

    def test_donors_masks_and_protected_columns(self):
        train, _ = example.synthetic_view('train', 1)
        x = train.features()
        reassigned = train.features('reassigned', 2027)
        np.testing.assert_equal(x[:, [0, 3]], reassigned[:, [0, 3]])
        np.testing.assert_equal(reassigned[:, [1, 2]],
            train._features[np.ix_(train.donors[2027][train.recipients], [1, 2])])
        self.assertTrue(np.isin(train.donors[2027][train.recipients], np.arange(3, 48, 4)).any())
        np.testing.assert_equal(train.features('power')[:, [1, 2]], 0)
        x[:] = 0
        self.assertGreater(train.features()[:, 0].sum(), 0)

    def test_evaluation_labels_unavailable(self):
        view, truth = example.synthetic_view('later_1', 2)
        with self.assertRaises(ValueError):
            view.training_targets()
        with self.assertRaises(ValueError):
            RecordInputView(view.metadata, view._features, view.recipients, view.moved_columns,
                            view.donors, 'later_1', targets=truth)
        with self.assertRaises(ValueError):
            fit_record_models(view, example.make_model)

    def test_factory_cannot_share_a_mutable_model(self):
        train, _ = example.synthetic_view('train', 1)
        shared = SpyModel()
        with self.assertRaisesRegex(ValueError, 'fresh model'):
            fit_record_models(train, lambda seed: shared)

    def test_predictions_have_complete_shape_and_finite_values(self):
        train, _ = example.synthetic_view('train', 1)
        view, _ = example.synthetic_view('later_1', 2)
        models = fit_record_models(train, example.make_model)
        for bad in [np.zeros((36, 1)), np.full((36, 10), np.nan)]:
            with patch.object(models['power'], 'predict', return_value=bad):
                with self.assertRaises(ValueError):
                    list(predict_record_models(models, view))

    def test_synthetic_demonstration_is_explicit(self):
        report = example.run_example()
        self.assertEqual(report['fitted_models'], 5)
        self.assertEqual(report['prediction_streams'], 24)
        self.assertIn('not benchmark evidence', report['purpose'])

    def test_source_cache_run_uses_native_scoring_and_preview_does_not_import_model(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            source = source_fixture(root, splits=True, incomplete=True)
            build_acn_packet(source, root/'data')
            prepare_inputs(root/'data', 'ACN', 'caltech', root/'cache', source)
            args = ['--data', str(root/'data'), '--inputs-cache', str(root/'cache'),
                    '--source', 'ACN', '--site', 'caltech', '--model-script',
                    str(Path(example.__file__)), '--model-name', 'Linear usage example',
                    '--output', str(root/'run')]
            with patch.object(runner.runpy, 'run_path', side_effect=AssertionError('preview cannot load model')):
                self.assertTrue(runner.main(args)['preview'])
            self.assertFalse((root/'run').exists())
            result = runner.main(args+['--run'])
            self.assertEqual(result['fitted_models'], 5)
            self.assertEqual(result['prediction_packets'], 24)
            self.assertEqual(len(list((root/'run/predictions').glob('*/*.npz'))), 24)
            text = (root/'run/comparison.md').read_text()
            self.assertIn('Linear usage example', text)
            self.assertNotIn('Full default training run.', text)
            self.assertTrue((root/'run/effects.json').is_file())
            with self.assertRaisesRegex(ValueError, 'new or empty'):
                runner.main(args+['--run'])
            with self.assertRaisesRegex(ValueError, 'different data root'):
                load_record_cache(root/'cache', root/'data', 'ACN', 'office001')


if __name__ == '__main__':
    unittest.main()
