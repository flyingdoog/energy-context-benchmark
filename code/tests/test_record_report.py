import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
import json

SPEC = importlib.util.spec_from_file_location('record_report', Path(__file__).resolve().parents[1]/'scripts/summarize_record_comparison.py')
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


def fixture():
    receipt = dict(completed=True, paper_result_eligible=True, source='ACN', site='office001',
                   training_cases=807, original_training_cases=807)
    rows = []
    for span in [0, 30, 90, 150]:
        for metric, mul in [('mse', 1), ('mae', 0.5)]:
            row = dict(source='ACN', site='office001', scale='cohort_total', period='validation',
                       span_minutes=span, metric=metric,
                       power_error=10*mul, paired_error=8*mul, frozen_error=15*mul, retrained_error=10*mul)
            for name in REPORT.CONTRASTS:
                row.update({name+'_pct': 25., name+'_low': None, name+'_high': None})
            rows.append(row)
    return receipt, rows


class RecordReportTests(unittest.TestCase):
    def test_all_arms_and_metrics_explicit(self):
        receipt, rows = fixture()
        text = REPORT.render_report(receipt, rows)
        self.assertIn('MSE / MAE', text)
        self.assertIn('10.000 / 5.000 | 8.000 / 4.000 | 15.000 / 7.500 | 10.000 / 5.000', text)
        self.assertIn('Average', text)
        self.assertIn('25.00 [unavailable]', text)
        self.assertNotRegex(text, r'\bnan\b')

    def test_smoke_and_incomplete(self):
        receipt, rows = fixture()
        receipt['paper_result_eligible'] = False
        self.assertIn('Training-subset smoke', REPORT.render_report(receipt, rows))
        receipt['completed'] = False
        with self.assertRaises(ValueError):
            REPORT.render_report(receipt, rows)

    def test_bpa_default_site(self):
        receipt, rows = fixture()
        receipt.update(source='BPA', site=None)
        for row in rows:
            row.update(source='BPA', site='Field study')
        self.assertIn('BPA / Field study', REPORT.render_report(receipt, rows))
        rows[0]['site'] = 'another study'
        with self.assertRaises(ValueError):
            REPORT.render_report(receipt, rows)

    def test_missing_duplicate_and_nonfinite(self):
        receipt, rows = fixture()
        for broken in [rows[:-1], rows+[rows[0]]]:
            with self.assertRaises(ValueError):
                REPORT.render_report(receipt, broken)
        broken = copy.deepcopy(rows)
        broken[0]['power_error'] = float('nan')
        with self.assertRaises(ValueError):
            REPORT.render_report(receipt, broken)

    def test_cli_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            receipt, rows = fixture()
            (root/'complete.json').write_text(json.dumps(receipt), encoding='utf-8')
            (root/'effects.json').write_text(json.dumps(rows), encoding='utf-8')
            args = ['--run', str(root), '--output', str(root/'comparison.md')]
            REPORT.main(args)
            with self.assertRaises(FileExistsError):
                REPORT.main(args)


if __name__ == '__main__':
    unittest.main()
