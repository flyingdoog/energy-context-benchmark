import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_local
import numpy as np
from field_control import OperationalForecastTask
from field_control.telemetry_availability import blackout_native_inputs,validate_blackout
from test_operational_task import acn,epfl


def bpa_native():
    n=2
    return dict(past_power_kw=np.ones((n,96)),past_power_observed_fraction=np.ones((n,96)),
        past_state=np.ones((n,96,5)),past_state_observed=np.ones((n,96,5),dtype=bool),
        past_command_type=np.ones((n,96),dtype=np.int16),past_command_parameter=np.ones((n,96)),
        past_command_observed=np.ones((n,96),dtype=bool),
        current_state=np.ones((n,6)),current_state_observed=np.ones((n,6),dtype=bool),
        current_command=np.ones((n,2)),current_command_observed=np.ones(n,dtype=bool),
        query_power_kw=np.ones(n),query_power_observed=np.ones(n,dtype=bool),
        previous_day_profile_kw=np.ones((n,12)),previous_week_profile_kw=np.ones((n,12)),
        target_query_offsets=np.c_[np.arange(1,151,15),np.arange(15,151,15)])


class OperationalAvailabilityTests(unittest.TestCase):
    def equal(self,a,b):
        self.assertEqual(a.keys(),b.keys())
        for key in a:
            if isinstance(a[key],dict):self.equal(a[key],b[key])
            elif isinstance(a[key],np.ndarray):np.testing.assert_equal(a[key],b[key])
            else:self.assertEqual(a[key],b[key])

    def epfl_task(self):
        data=epfl();data._x[:,14:]=np.arange(80)+1
        with patch('field_control.operational.EpflDataset',return_value=data):
            return OperationalForecastTask('.','EPFL')

    def test_default_zero_and_power_only_are_unchanged(self):
        task=self.epfl_task()
        self.equal(task.inputs([1,2]),task.inputs([1,2],auxiliary_blackout_minutes=0))
        self.equal(task.inputs([1,2],'power'),task.inputs([1,2],'power',auxiliary_blackout_minutes=5))

    def test_epfl_clock_masks_and_boolean_unknown(self):
        task=self.epfl_task();before=task.inputs([2,3])
        after=task.inputs([2,3],auxiliary_blackout_minutes=5)
        p=after['native_inputs'];old=before['native_inputs']
        self.assertEqual(after['availability']['retained_auxiliary_report_offset_minutes_less_than'],-5)
        self.equal(before['contract'],after['contract'])
        for key in ['power_kw','power_observed','target_history_kw','calendar','target_intervals_minutes']:
            np.testing.assert_equal(p[key],old[key])
        self.assertTrue(np.isnan(p['numeric_state'][:,5:]).all())
        self.assertFalse(p['numeric_state_observed'][:,5:].any())
        self.assertFalse(p['controlled'][:,5:].any())
        self.assertFalse(p['controlled_observed'][:,5:].any())
        np.testing.assert_equal(p['numeric_state'][:,:5],old['numeric_state'][:,:5])
        self.equal(before,task.inputs([2,3]))

    def test_hidden_tail_and_target_changes_cannot_change_available_inputs(self):
        task=self.epfl_task();before=task.inputs([2,3],auxiliary_blackout_minutes=5)
        task.base.target[:]=999
        task.base._x[:,14:64].reshape(4,10,5)[:,5:]=1e8
        task.base._x[:,64:94].reshape(4,10,3)[:,5:]=1e8
        self.equal(before,task.inputs([2,3],auxiliary_blackout_minutes=5))

    def test_acn_native_pack_reconstructs_original_mask_convention(self):
        data=acn();data._x[:,120:1440]=1
        with patch('field_control.operational.AcnDataset',return_value=data):
            task=OperationalForecastTask('.','ACN',site='caltech')
        original=task.inputs([4,5]);out=task.inputs([4,5],auxiliary_blackout_minutes=5)
        p=out['native_inputs']
        for value,mask in [('pilot_a','pilot_observed'),('numeric_state','numeric_state_observed'),
                            ('operating_state_fractions','operating_state_observed')]:
            self.assertTrue(np.isnan(p[value][:,55:]).all())
            self.assertFalse(p[mask][:,55:].any())
            self.assertTrue((p[value][:,:55]==1).all())
            self.assertTrue((np.where(p[mask],p[value],0)[:,55:]==0).all())
        np.testing.assert_equal(p['power_kw'],original['native_inputs']['power_kw'])
        np.testing.assert_equal(p['calendar'],original['native_inputs']['calendar'])
        self.equal(original,task.inputs([4,5]))

    def test_bpa_bin_end_clock_snapshot_power_and_profiles(self):
        original=bpa_native();saved=copy.deepcopy(original)
        out=blackout_native_inputs(original,'BPA',5)
        # Latest bin's state report is at -1; the preceding one is at -16.
        self.assertTrue(np.isnan(out['past_state'][:,-1]).all())
        np.testing.assert_equal(out['past_state'][:,:95],original['past_state'][:,:95])
        self.assertFalse(out['past_state_observed'][:,-1].any())
        self.assertTrue((out['past_command_type'][:,-1]==-1).all())
        self.assertFalse(out['past_command_observed'][:,-1].any())
        self.assertTrue(np.isnan(out['current_state'][:,1:]).all())
        np.testing.assert_equal(out['current_state'][:,0],original['current_state'][:,0])
        for key in ['past_power_kw','past_power_observed_fraction','query_power_kw','query_power_observed',
                    'previous_day_profile_kw','previous_week_profile_kw','target_query_offsets']:
            np.testing.assert_equal(out[key],original[key])
        self.equal(original,saved)

    def test_bpa_cutoff_boundary_uses_strict_retention(self):
        p=bpa_native()
        a=blackout_native_inputs(p,'BPA',15);b=blackout_native_inputs(p,'BPA',16)
        self.assertTrue(a['past_state_observed'][:,-2].all())
        self.assertFalse(b['past_state_observed'][:,-2].any())

    def test_power_state_filter_does_not_reintroduce_control(self):
        task=self.epfl_task();p=task.inputs([2],'power_state',auxiliary_blackout_minutes=5)['native_inputs']
        for name in ['requested_kw','setpoint_kw','controlled']:self.assertNotIn(name,p)
        self.assertTrue(np.isnan(p['numeric_state'][:,5:]).all())

    def test_bad_cutoffs_rejected_and_all_history_can_be_unavailable(self):
        for value in [-1,1.5,float('nan'),float('inf'),'5',True,11]:
            with self.assertRaises(ValueError):validate_blackout('EPFL',value)
        task=self.epfl_task();p=task.inputs([2],auxiliary_blackout_minutes=10)['native_inputs']
        self.assertFalse(p['numeric_state_observed'].any())
        np.testing.assert_equal(p['power_kw'],task.inputs([2])['native_inputs']['power_kw'])


if __name__=='__main__':unittest.main()
