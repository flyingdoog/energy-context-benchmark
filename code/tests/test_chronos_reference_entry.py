import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import run_chronos_reference as entry
from field_control.chronos_inputs import ForecastGrid


class Tensor:
    def __init__(self, value): self.value = value
    def cpu(self): return self
    def numpy(self): return self.value


class EntryTests(unittest.TestCase):
    def test_native_medians_binned_without_quantile_aggregation(self):
        class Pipeline:
            def predict_quantiles(self, items, **kw):
                self.options = kw
                return None, [Tensor(np.arange(30).reshape(1, 30)) for _ in items]
        model = Pipeline()
        grid = ForecastGrid("EPFL", 10, 30, 6, 5, 5)
        value = entry.point_forecasts(model, [{}, {}], grid, 256)
        np.testing.assert_allclose(value[0], [2, 7, 12, 17, 22, 27])
        self.assertEqual(value.shape, (2, 6))
        self.assertFalse(model.options["cross_learning"])
        self.assertEqual(model.options["prediction_length"], 30)

    def test_resume_requires_matching_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"batch.npz"
            np.savez(path, case_id=np.array(["a"]), prediction_kw=np.ones((1, 6)))
            self.assertEqual(entry.saved_batch(path, np.array(["a"]), 6).shape, (1, 6))
            with self.assertRaisesRegex(ValueError, "cases"):
                entry.saved_batch(path, np.array(["b"]), 6)

    def test_resume_rejects_wrong_native_width(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"batch.npz"
            np.savez(path, case_id=np.array(["a"]), prediction_kw=np.ones((1, 10)))
            with self.assertRaisesRegex(ValueError, "shape"):
                entry.saved_batch(path, np.array(["a"]), 6)

    def test_preview_has_no_model_or_score(self):
        task = SimpleNamespace(source="EPFL", site="Fast charging", base=SimpleNamespace(),
            contract={"available_conditions": ("power", "full")}, indices=lambda split:np.arange(5))
        grid = ForecastGrid("EPFL", 10, 30, 6, 5, 5)
        items = [{"past_covariates": {"x":np.ones(10)}, "future_covariates":{"calendar":np.ones(30)}}]*2
        output = io.StringIO()
        with patch.object(entry, "OperationalForecastTask", return_value=task), \
             patch.object(entry, "native_items", return_value=(items, grid)), \
             contextlib.redirect_stdout(output):
            entry.main(["--source", "EPFL", "--data", "example", "--split", "train", "--describe-inputs"])
        result = json.loads(output.getvalue())
        self.assertEqual(result["split_cases"], 5)
        self.assertEqual(result["new_predictions"], 0)
        self.assertFalse(result["scores_produced"])

    def test_nonpositive_batch_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            entry.main(["--source", "EPFL", "--data", "example", "--split", "train",
                        "--describe-inputs", "--case-batch", "0"])


if __name__ == "__main__":
    unittest.main()
