import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_local
import numpy as np
import pandas as pd
from field_control import OperationalForecastTask,AcnDataset,EpflDataset


def epfl():
    data=EpflDataset.__new__(EpflDataset);data.target_name='actual_power'
    data._x=np.zeros((4,94));data._x[:,:10]=2
    data._targets=np.zeros((4,6,2));data.target=data._targets[:,:,0]
    data.meta=pd.DataFrame(dict(case_id=list('abcd'),split=['train','validation','later','later'],
        date=['2022-01-01','2023-01-01','2023-04-01','2023-04-02']))
    return data


def acn():
    data=AcnDataset.__new__(AcnDataset);data._x=np.zeros((6,1445));data._x[:,:60]=2;data._x[:,60:120]=1
    data.target=np.ones((6,10));data.receipt={'feature_groups':{'power':[0,120],'control':[120,240],'state':[240,1440],'calendar':[1440,1445]}}
    data.meta=pd.DataFrame(dict(case_id=list('abcdef'),site=['caltech']*6,split=['train']*2+['validation']*2+['winter']*2,
        local_date=['2019-01-01']*2+['2019-07-01']*2+['2020-01-01']*2,
        query_utc=['q1']*2+['q2']*2+['q3']*2,station_id=['a','b']*3))
    data.population=data.meta.copy();return data


class OperationalTaskTests(unittest.TestCase):
    def task(self):
        with patch('field_control.operational.EpflDataset',return_value=epfl()):
            return OperationalForecastTask('.', 'EPFL')

    def test_split_separation_and_native_time_contract(self):
        task=self.task();self.assertEqual(task.indices('train').tolist(),[0])
        self.assertEqual(task.indices('later').tolist(),[2,3])
        np.testing.assert_equal(task.contract['report_offsets_minutes'][[0,-1]],[[0,4],[25,29]])
        with self.assertRaises(ValueError):task.indices('boundary_excluded')
        with self.assertRaises(ValueError):task.indices('winter')

    def test_inputs_are_separate_from_labels_and_masked(self):
        task=self.task();before=task.inputs([2],condition='power')
        task.base.target[:]=999
        after=task.inputs([2],condition='power')
        self.assertNotIn('target',after['native_inputs']);self.assertNotIn('numeric_state',after['native_inputs'])
        np.testing.assert_equal(before['native_inputs']['power_kw'],after['native_inputs']['power_kw'])
        self.assertNotIn('requested_kw',after['native_inputs'])
        with self.assertRaises(ValueError):task.inputs([.5])

    def test_power_only_and_probability_only(self):
        task=self.task();power=np.repeat([[1.],[3.]],6,axis=1)
        scores=task.score(split='later',prediction_kw=power)
        self.assertEqual(set(scores),{'power'});np.testing.assert_allclose(scores['power'][0].mse,5)
        scores=task.score(split='later',positive_probability=np.full((2,6),.5))
        self.assertEqual(set(scores),{'probability'});np.testing.assert_allclose(scores['probability'][0].brier,.25)
        with self.assertRaises(ValueError):task.score(split='later',prediction_kw=power,require_groups=True)
        with self.assertRaises(ValueError):task.score(split='later')

    def test_submission_roundtrip_reordering(self):
        task=self.task()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'p.npz';power=np.repeat([[1.],[3.]],6,axis=1)
            task.submission(path,split='later',prediction_kw=power,positive_probability=np.full((2,6),.5))
            with np.load(path,allow_pickle=False) as archive:stored={k:archive[k] for k in archive.files}
            for k in ['case_id','prediction_kw','positive_probability']:stored[k]=stored[k][::-1]
            np.savez_compressed(path,**stored)
            loaded=task.read_submission(path,split='later');np.testing.assert_equal(loaded['prediction_kw'],power)
            self.assertEqual(set(task.score(split='later',**loaded)),{'power','probability'})

    def test_wrong_task_ids_and_intervals_rejected(self):
        task=self.task()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'p.npz';task.submission(path,split='later',prediction_kw=np.ones((2,6)))
            with np.load(path,allow_pickle=False) as p:original={k:p[k] for k in p.files}
            for change in [dict(target_name='requested_power'),dict(source='BPA'),dict(case_id=['c','c']),
                           dict(report_offsets_minutes=task.report_offsets+1)]:
                np.savez_compressed(path,**(original|change))
                with self.assertRaises(ValueError):task.read_submission(path,split='later')
        with self.assertRaises(ValueError):task.score(split='later',positive_probability=np.full((2,6),1.1))

    def test_acn_observation_mask_and_groups(self):
        data=acn();data._x[4,119]=0
        with patch('field_control.operational.AcnDataset',return_value=data):
            task=OperationalForecastTask('.','ACN',site='caltech')
        power,observed=task.current_power(task.indices('winter'))
        self.assertFalse(observed[0]);self.assertTrue(np.isnan(power[0]))
        scores=task.score(split='winter',positive_probability=np.full((2,10),.5))
        counts=scores['group_activity'][0];np.testing.assert_allclose(counts[counts.scale=='positive_count'].mse,1)
        task.base.population=pd.concat([data.population,data.population.iloc[-1:].assign(station_id='c')])
        self.assertTrue(task.score(split='winter',positive_probability=np.full((2,10),.5))['group_activity'][0].empty)

    def test_bpa_saved_split_and_information_filter(self):
        meta=pd.DataFrame(dict(case_id=list('abc'),local_date=['a','b','c'],device_index=[0]*3,origin_minute=[40,80,120]))
        base=SimpleNamespace(meta=meta,grid={'power_kw':np.ones((1,200)),'online':np.ones((1,200),dtype=bool)},
            power_windows=lambda ix,starts:(np.ones((len(ix),len(starts))),np.ones((len(ix),len(starts)))))
        def inputs(ix,observed_minutes):
            n=len(ix);out={'current_state':np.ones((n,6)),'current_state_observed':np.ones((n,6),dtype=bool),
                'past_power_kw':np.ones((n,96)),'past_power_observed_fraction':np.ones((n,96))}
            for key in ['past_state','past_state_observed','past_command_type','past_command_parameter','past_command_observed',
                        'current_command','current_command_observed']:out[key]=np.ones((n,1))
            return out
        native=SimpleNamespace(meta=meta,target=np.ones((3,10)),base=base,refreshed_inputs=inputs)
        with tempfile.TemporaryDirectory() as tmp:
            split=Path(tmp)/'update_validation';split.mkdir();np.savez(split/'split.npz',train=[0],validation=[2])
            with patch('field_control.operational.BpaResponseTask',return_value=native):
                task=OperationalForecastTask(tmp,'BPA',season='winter')
            self.assertEqual(task.meta.split.tolist(),['train','boundary_excluded','validation'])
            np.testing.assert_equal(task.report_offsets[[0,-1]],[[1,15],[136,150]])
            p=task.inputs([0],condition='power')['native_inputs']
            self.assertIn('query_power_kw',p);self.assertNotIn('current_state',p);self.assertNotIn('current_command',p)
            self.assertEqual(p['previous_week_profile_kw'].shape,(1,12))
            self.assertTrue((p['previous_week_profile_report_offsets']<0).all())
            self.assertEqual(base.meta.origin_minute.iloc[0],40)
            p=task.inputs([0],condition='power_state')['native_inputs'];self.assertIn('current_state',p);self.assertNotIn('current_command',p)


if __name__=='__main__':unittest.main()
