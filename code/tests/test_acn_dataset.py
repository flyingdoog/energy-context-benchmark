import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_local
import numpy as np
import pandas as pd
from field_control import AcnDataset
from field_control.acn_dataset import TARGET_MINUTES


class AcnDatasetTests(unittest.TestCase):
    def dataset(self):
        data=AcnDataset.__new__(AcnDataset)
        data._x=np.zeros((2,1445),dtype=np.float32);data.target=np.zeros((2,10),dtype=np.float32)
        data.receipt={'feature_groups':{'power':[0,120],'control':[120,240],'state':[240,1440],'calendar':[1440,1445]}}
        data.meta=pd.DataFrame(dict(case_id=['a','b'],site=['s','s'],split=['winter','winter'],
            local_date=['2020-01-01']*2,query_utc=['q']*2,station_id=['a','b']))
        data.population=data.meta.copy()
        data._x[:,:60]=3;data._x[:,60:120]=1
        return data

    def test_typed_inputs_keep_missing_values_and_information_conditions(self):
        data=self.dataset();inputs=data.inputs(condition='power')
        self.assertEqual(inputs['power_kw'].shape,(2,60,1))
        self.assertTrue((inputs['power_kw']==3).all())
        self.assertNotIn('pilot_a',inputs)
        rich=data.inputs(condition='full')
        self.assertTrue(np.isnan(rich['numeric_state']).all())
        self.assertEqual(rich['operating_state_fractions'].shape,(2,60,7))
        np.testing.assert_array_equal(rich['target_intervals_minutes'][0],[0,15])
        self.assertNotIn('target',rich)

    def test_submission_reorders_ids_and_rejects_wrong_targets(self):
        data=self.dataset()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'test.npz'
            values=np.repeat(np.array([2.,1.])[:,None],10,axis=1)
            np.savez_compressed(path,case_id=['b','a'],target_minutes=TARGET_MINUTES,prediction_kw=values)
            self.assertEqual(data.read_submission(path)[0,0],1)
            np.savez_compressed(path,case_id=['b','a'],target_minutes=TARGET_MINUTES+1,prediction_kw=values)
            with self.assertRaises(ValueError):data.read_submission(path)

    def test_public_scorer_uses_whole_query_known_cohorts(self):
        data=self.dataset();summary,_=data.score(np.ones((2,10)))
        group=summary[(summary.scale=='cohort_mean')&(summary.span_minutes==30)].iloc[0]
        self.assertEqual(group['mse'],1)
        data.population=pd.concat([data.population,data.population.iloc[:1].assign(case_id='c',station_id='c')])
        summary,_=data.score(np.ones((2,10)))
        self.assertNotIn('cohort_mean',summary.scale.tolist())

    def test_optional_declarations_align_and_check_availability(self):
        data=self.dataset();data.meta['query_utc']='2020-01-01T12:00:00Z'
        with tempfile.TemporaryDirectory() as directory:
            data.root=Path(directory);folder=data.root/'request_metadata_v1';folder.mkdir()
            path=folder/'query_requests.csv.gz'
            rows=pd.DataFrame(dict(case_id=['b','a'],site=['s','s'],requested_kwh=[5,10],
                requested_departure_hours=[-1,2],modified_at=[None,'2020-01-01T11:00:00Z']))
            rows.to_csv(path,index=False)
            values=data.inputs(declared=True)['declarations']
            np.testing.assert_allclose(values['requested_kwh'],[10,np.nan])
            self.assertNotIn('declarations',data.inputs())
            del data._requests
            rows.loc[1,'modified_at']='2020-01-01T13:00:00Z';rows.to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,'after'):data.declarations()


if __name__=='__main__':unittest.main()
