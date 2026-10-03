import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import run_local
import numpy as np
import pandas as pd
from field_control import EpflDataset
from field_control.epfl_dataset import TARGET_MINUTES


class EpflDatasetTests(unittest.TestCase):
    def dataset(self,target="actual_power"):
        data=EpflDataset.__new__(EpflDataset)
        data.target_name=target
        data._x=np.zeros((3,94),dtype=float)
        data._x[:,:10]=2
        data._x[:,65:94:3]=3500
        data._x[:,64:94:3]=4500
        data._x[:,66:94:3]=1
        data._targets=np.zeros((3,6,2));data._targets[:,:,1]=8
        data.target=data._targets[:,:,0 if target=="actual_power" else 1]
        data.meta=pd.DataFrame(dict(case_id=["a","b","c"],split=["later"]*3,
                                   date=["2023-04-01","2023-04-01","2023-04-02"]))
        return data

    def test_actual_target_power_only_does_not_expose_other_fields(self):
        data=self.dataset();inputs=data.inputs(condition="power")
        self.assertEqual(inputs["power_kw"].shape,(3,10,1))
        self.assertNotIn("numeric_state",inputs)
        self.assertNotIn("requested_kw",inputs)
        self.assertNotIn("controlled",inputs)
        self.assertNotIn("target",inputs)
        np.testing.assert_array_equal(inputs["target_history_kw"],2*np.ones((3,10,1)))

    def test_requested_target_always_has_own_history_in_aligned_views(self):
        data=self.dataset("requested_power")
        for condition in ["power","power_state","full"]:
            inputs=data.inputs(condition=condition)
            np.testing.assert_array_equal(inputs["requested_kw"],3.5*np.ones((3,10,1)))
            np.testing.assert_array_equal(inputs["target_history_kw"],inputs["requested_kw"])
            self.assertTrue((data.features(condition=condition)[:,65:94:3]==3500).all())
        legacy=data.inputs(condition="power",include_target_history=False)
        self.assertNotIn("requested_kw",legacy)
        self.assertNotIn("target_history_kw",legacy)

    def test_full_input_units_and_copies(self):
        data=self.dataset();inputs=data.inputs(condition="full")
        np.testing.assert_array_equal(inputs["setpoint_kw"],4.5*np.ones((3,10,1)))
        self.assertEqual(inputs["numeric_state_units"],("%","source_native","source_native","source_native","Wh"))
        self.assertTrue(inputs["controlled"].all())
        inputs["power_kw"][:]=999
        self.assertTrue((data._x[:,:10]==2).all())

    def test_target_changes_do_not_change_inputs(self):
        data=self.dataset();first=data.inputs()
        data._targets[:]=999
        later=data.inputs()
        for key,value in first.items():
            if isinstance(value,np.ndarray):np.testing.assert_array_equal(value,later[key])

    def test_scoring_equal_weights_dates_not_sessions(self):
        data=self.dataset()
        prediction=np.repeat(np.array([1.,1.,3.])[:,None],6,axis=1)
        summary,daily=data.score(prediction)
        self.assertTrue((summary.mse==5).all())
        self.assertTrue((summary.mae==2).all())
        row=summary[summary.span_minutes==30].iloc[0]
        self.assertAlmostEqual(row.energy_ae_kwh,1)
        self.assertEqual(len(daily),6)

    def test_submissions_reorder_and_distinguish_targets(self):
        data=self.dataset()
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/"submission.npz"
            values=np.repeat(np.array([3.,1.,2.])[:,None],6,axis=1)
            np.savez_compressed(path,case_id=["c","a","b"],target_name="actual_power",
                                target_minutes=TARGET_MINUTES,prediction_kw=values)
            np.testing.assert_array_equal(data.read_submission(path)[:,0],[1,2,3])
            requested=self.dataset("requested_power")
            with self.assertRaisesRegex(ValueError,"target"):requested.read_submission(path)
            data.submission(path,np.ones((3,6)))
            np.testing.assert_array_equal(data.read_submission(path),np.ones((3,6)))

    def test_wrong_intervals_and_missing_target_name_rejected(self):
        data=self.dataset()
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/"submission.npz"
            for edges,target in [(TARGET_MINUTES+1,"actual_power"),(TARGET_MINUTES,None)]:
                values=dict(case_id=["a","b","c"],target_minutes=edges,prediction_kw=np.ones((3,6)))
                if target is not None:values["target_name"]=target
                np.savez_compressed(path,**values)
                with self.assertRaises(ValueError):data.read_submission(path)

    def test_inputs_allow_sampling_but_scoring_requires_unique_cases(self):
        data=self.dataset()
        self.assertEqual(data.inputs([1,1])["power_kw"].shape,(2,10,1))
        self.assertEqual(data.inputs(np.array([False,True,False]))["power_kw"].shape,(1,10,1))
        with self.assertRaises(ValueError):data.score(np.ones((2,6)),indices=[1,1])
        with self.assertRaises(ValueError):data.inputs([.5])
        with self.assertRaises(ValueError):data.score(np.full((3,6),np.nan))


if __name__=="__main__":unittest.main()

