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
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_local
import example_record_level as example
import run_record_level_diagnostic as runner
from field_control.group_error_level import LEADS
from field_control.record_level_diagnostic import analyze_groups, write_result, render_report
from field_control.record_reference_inputs import task_for
from field_control.record_reference_scores import RAW_ARMS
from field_control.acn_preparation import build_acn_packet
from test_acn_preparation import source_fixture


class PortableLevelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = analyze_groups(example.synthetic_groups(),source='Synthetic',site='Example',
                                    periods=example.PERIODS,draws=25)

    def test_all_periods_spans_scales_and_arms(self):
        result = self.result
        self.assertEqual(result['receipt']['prediction_packets'],24)
        self.assertEqual(result['receipt']['validation_offsets'],16)
        self.assertEqual(result['receipt']['effect_rows'],96)
        self.assertEqual(result['receipt']['paired_contrasts'],288)
        effects = result['tables']['effects']
        self.assertEqual(set(effects.period),set(example.PERIODS))
        self.assertEqual(set(effects.span_minutes),{0,30,90,150})
        self.assertEqual(set(effects.metric),{'mse','mae'})
        self.assertEqual(set(effects.calibration),{'raw','validation_offset'})

    def test_unequal_dates_have_equal_weight_and_offsets_transfer(self):
        item = next(r for r in self.result['validation_offsets']
                    if r['method']=='Paired' and r['scale']=='cohort_total')
        # Validation paired errors: first date 3; second date average 2.
        np.testing.assert_allclose(item['intercept_kw'],2.5*np.linspace(.8,1.2,10))
        frame = self.result['tables']['lead_decomposition']
        raw = frame[(frame.method=='Paired')&(frame.period=='later_1')&
                    (frame.scale=='cohort_total')&(frame.calibration=='raw')]
        corrected = frame[(frame.method=='Paired')&(frame.period=='later_1')&
                          (frame.scale=='cohort_total')&(frame.calibration=='validation_offset')]
        np.testing.assert_allclose(corrected.bias_mse,.6**2*np.linspace(.8,1.2,10)**2)
        np.testing.assert_allclose(raw.centered_mse,corrected.centered_mse)
        spans = self.result['tables']['decomposition']
        np.testing.assert_allclose(spans.mse,spans.bias_mse+spans.centered_mse,atol=1e-12)

    def test_later_changes_never_refit_validation_even_if_stream_reordered(self):
        entries = list(example.synthetic_groups())
        for period,method,groups in entries:
            if period != 'validation':
                for frame in groups.values():
                    frame[LEADS] += 10
        changed = analyze_groups(reversed(entries),source='Synthetic',site='Example',
                                 periods=example.PERIODS,draws=25)
        self.assertEqual(changed['validation_offsets'],self.result['validation_offsets'])

    def test_missing_validation_or_mapping_and_bad_leads_fail(self):
        with self.assertRaisesRegex(ValueError,'Missing required'):
            analyze_groups(list(example.synthetic_groups())[1:],source='Synthetic',site='Example',
                           periods=example.PERIODS,draws=2)
        for issue in ('missing','nonfinite'):
            records = list(example.synthetic_groups())
            frame = records[0][2]['cohort_total']
            if issue=='missing':
                frame.drop(columns=['e9'],inplace=True)
            else:
                frame.loc[0,'e9']=np.nan
            with self.assertRaises(ValueError):
                analyze_groups(records,source='Synthetic',site='Example',periods=example.PERIODS,draws=2)

    def test_membership_mismatch_and_duplicates_fail(self):
        records = list(example.synthetic_groups())
        for frame in records[1][2].values():
            frame.loc[0,'query_utc']='a_different_group'
        with self.assertRaisesRegex(ValueError,'different group memberships'):
            analyze_groups(records,source='Synthetic',site='Example',periods=example.PERIODS,draws=2)
        records = list(example.synthetic_groups())
        with self.assertRaisesRegex(ValueError,'duplicate group packet'):
            analyze_groups(records+[records[0]],source='Synthetic',site='Example',periods=example.PERIODS,draws=2)

    def test_report_preserves_absolute_values_and_prior_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/'example'
            receipt = write_result(output,self.result,dict(purpose='Synthetic, not benchmark evidence'))
            report = (output/'comparison.md').read_text(encoding='utf-8')
            self.assertIn('MSE / MAE',report)
            self.assertIn('not benchmark evidence',report)
            for period in example.PERIODS:
                for span in (30,90,150):
                    self.assertIn(f'| {period} | total | {span} | Offset |',report)
            self.assertEqual(receipt['model_fits'],0)
            self.assertEqual(receipt['inference_calls'],0)
            with self.assertRaisesRegex(ValueError,'new or empty'):
                write_result(output,self.result)

    def test_native_packets_preview_relocation_replay_and_time_mismatch(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            source = source_fixture(root,splits=True)
            build_acn_packet(source,root/'data')
            run = root/'supplied_run'
            run.mkdir()
            # Historical paths and model code are deliberately unavailable.
            (run/'protocol.json').write_text(json.dumps(dict(source='ACN',site='caltech',
                data_root='not-present',model_script='must-not-be-loaded.py')))
            (run/'complete.json').write_text(json.dumps(dict(completed=True,model_name='Synthetic packet source')))
            for period in ('validation','autumn','winter'):
                task = task_for(root/'data','ACN','caltech',period)
                folder = run/'predictions'/period
                folder.mkdir(parents=True)
                truth = task.targets(task.indices(period))
                for method in RAW_ARMS:
                    prediction = truth + RAW_ARMS.index(method)*.1
                    task.submission(folder/(method.replace(' ','_')+'.npz'),split=period,prediction_kw=prediction)
            args = ['--run-directory',str(run),'--data',str(root/'data'),
                    '--source','ACN','--site','caltech','--output',str(root/'analysis')]
            with patch.object(runner,'packet_groups',side_effect=AssertionError('Preview must not read values')):
                self.assertTrue(runner.main(args)['preview'])
            self.assertFalse((root/'analysis').exists())
            result = runner.main(args+['--run'])
            self.assertEqual(result['prediction_packets'],24)
            self.assertEqual(result['inference_calls'],0)
            self.assertTrue((root/'analysis/effects.json').exists())
            with self.assertRaisesRegex(ValueError,'new or empty'):
                runner.main(args+['--run'])
            path = run/'predictions/validation/Paired.npz'
            with np.load(path,allow_pickle=False) as packet:
                payload = {key:packet[key] for key in packet.files}
            payload['report_offsets_minutes'][0] += 1
            np.savez_compressed(path,**payload)
            with self.assertRaisesRegex(ValueError,'target times differ'):
                runner.main(args[:-1]+[str(root/'bad-times'),'--run'])
            self.assertFalse((root/'bad-times').exists())

    def test_preview_rejects_missing_validation_and_site_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data,run=root/'data',root/'run'
            data.mkdir();run.mkdir()
            (run/'protocol.json').write_text(json.dumps(dict(source='ACN',site='office001')))
            (run/'complete.json').write_text(json.dumps(dict(completed=True)))
            with self.assertRaisesRegex(ValueError,'Missing required prediction packets'):
                runner.inspect_run(run,data,'ACN','office001')
            with self.assertRaisesRegex(ValueError,'differs from the prediction run'):
                runner.inspect_run(run,data,'ACN','caltech')


if __name__=='__main__':
    unittest.main()
