import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import run_local
from field_control.group_error_level import (LEADS, group_residuals, equal_date_mean,
    lead_decomposition, daily_errors, span_decomposition, paired_week_difference)
from field_control.acn_scoring import score_acn_residuals
from field_control.response import score_response_predictions
from field_control.record_level_diagnostic import average_mappings


class GroupLevelTests(unittest.TestCase):
    def test_mapping_bias_squares_are_averaged_not_cancelled(self):
        rows = pd.DataFrame(dict(source='BPA',method=['Fixed 2027','Fixed 2028','Fixed 2029'],
                                 mse=[4.,0.,4.],bias_mse=[4.,0.,4.],centered_mse=0.))
        value = average_mappings(rows,['source'],['mse','bias_mse','centered_mse'])
        self.assertAlmostEqual(value.bias_mse.iloc[0],8/3)
        self.assertAlmostEqual(value.mse.iloc[0],8/3)

    def test_equal_date_weights_and_identity(self):
        frame = pd.DataFrame(np.repeat([[4.], [0.], [0.], [0.]], 10, axis=1), columns=LEADS)
        frame['date'] = ['2020-01-01', '2020-01-09', '2020-01-09', '2020-01-09']
        values = lead_decomposition(frame)
        np.testing.assert_allclose(values.mse, 8)
        np.testing.assert_allclose(values.bias_mse, 4)
        np.testing.assert_allclose(values.centered_mse, 4)
        np.testing.assert_allclose(equal_date_mean(frame), 2)
        self.assertEqual(set(span_decomposition(values).span_minutes), {0, 30, 90, 150})

    def test_validation_intercept_stays_fixed_on_later_errors(self):
        frame = pd.DataFrame(np.repeat([[3.], [5.]], 10, axis=1), columns=LEADS)
        frame['date'] = ['2020-01-01', '2020-01-09']
        offset = np.full(10, 2.)
        raw, corrected = lead_decomposition(frame), lead_decomposition(frame, offset)
        np.testing.assert_allclose(corrected.mse, 5)
        np.testing.assert_allclose(corrected.bias_mse, 4)
        np.testing.assert_allclose(raw.centered_mse, corrected.centered_mse)
        np.testing.assert_equal(offset, 2)
        np.testing.assert_allclose(daily_errors(frame, offset).groupby('span_minutes').mse.mean(), 5)

    def test_acn_incomplete_groups_do_not_shrink(self):
        meta = pd.DataFrame(dict(site='caltech', split='autumn',
            query_utc=['q1']*2+['q2']*2, local_date=['d1']*2+['d2']*2,
            station_id=['a', 'b', 'a', 'b']))
        population = pd.concat([meta, pd.DataFrame([dict(site='caltech', split='autumn',
            query_utc='q1', local_date='d1', station_id='c')])], ignore_index=True)
        residual = np.arange(40).reshape(4, 10)/10
        grouped = group_residuals('ACN', meta, residual, population)
        self.assertEqual(grouped['cohort_total'].date.tolist(), ['d2'])
        expected, _ = score_acn_residuals(meta, residual, 'arm', population)
        for scale, frame in grouped.items():
            actual = daily_errors(frame).groupby('span_minutes').mse.mean()
            target = expected[expected.scale == scale].set_index('span_minutes').mse
            np.testing.assert_allclose(actual, target)

    def test_bpa_group_weights_match_native(self):
        n = 29
        meta = pd.DataFrame(dict(cohort_id=['q1']*10+['q2']*10+['q3']*9,
            alias=[f'd{i}' for i in range(n)], local_date=['d1']*10+['d2']*19, command_type=1))
        residual = np.arange(n*10).reshape(n, 10)/100
        grouped = group_residuals('BPA', meta, residual)
        expected, _ = score_response_predictions(meta, np.zeros_like(residual), residual, 'arm', 'spring')
        for scale, frame in grouped.items():
            self.assertEqual(len(frame), 2)
            actual = daily_errors(frame).groupby('span_minutes').mse.mean()
            target = expected[(expected.scale == scale)&(expected.command_type == 'all')].set_index('target_end_minutes').mse
            target.index = target.index-30
            np.testing.assert_allclose(actual, target)

    def test_paired_intervals_and_unavailable_single_week(self):
        single = paired_week_difference(['2020-01-01'], [3], [2])
        self.assertEqual(single['difference'], 1)
        self.assertIsNone(single['low'])
        paired = paired_week_difference(['2020-01-01', '2020-01-09'], [3, 5], [2, 4], draws=100)
        self.assertEqual(paired['low'], 1)
        self.assertEqual(paired['high'], 1)


if __name__ == '__main__':
    unittest.main()
