"""Model-neutral observed-response queries with fixed absolute future targets."""
from pathlib import Path
import copy
import numpy as np
import pandas as pd
from .dataset import BpaDataset,HISTORY_STATE_NAMES,calendar_features

TARGET_ENDS=(60,120,180)
ABSOLUTE_BINS=np.arange(2,12)


class BpaResponseTask:
    """Same devices and minute31..180 labels at query0,15or30.

    Neither inputs nor scoring import a neural or tree library. Raw categorical
    fields retain their codes and explicit masks; inspect base.schema for types.
    """
    def __init__(self,root,season):
        self.base=BpaDataset(root,season);self.season=season

    @property
    def meta(self):return self.base.meta

    @property
    def target(self):return self.base.target[:,2:]

    @property
    def schema(self):return self.base.schema

    def __len__(self):return len(self.base)

    def refreshed_inputs(self,indices,observed_minutes=30,lookback_bins=96):
        """Ordinary latest-window inputs; fixed original targets at every query.

        The current report is at query time; history ends one minute earlier.
        Future calendar/offsets refer to the original ten fixed target bins, not
        the first ten bins after an earlier query. No original snapshot is added.
        """
        if observed_minutes not in (0,15,30):raise ValueError('Queries are0,+15,+30minutes')
        ix=np.atleast_1d(indices).astype(np.int64)
        self.base._power_prefix()
        view=copy.copy(self.base);view.meta=self.meta.iloc[ix].copy()
        original=view.meta.origin_minute.to_numpy().copy()
        view.meta['origin_minute']=original+observed_minutes
        p=view.inputs(np.arange(len(ix)),lookback_bins=lookback_bins)
        online=view.grid['online'][view.meta.device_index.to_numpy(),view.meta.origin_minute.to_numpy()]
        p['current_state'][~online]=np.nan;p['current_state_observed'][~online]=False
        p['current_command'][~online]=np.nan
        p['current_command_observed']=online&np.isfinite(p['current_command']).all(axis=1)
        starts=31+15*np.arange(10)
        p['future_calendar']=calendar_features(view.receipt['first_timestamp'],original[:,None]+starts[None,:])
        p['target_relative_intervals']=np.c_[starts,starts+14]
        p['target_query_offsets']=p['target_relative_intervals']-observed_minutes
        p['observed_minutes']=observed_minutes
        return p

    def inputs(self,indices,observed_minutes=30,lookback_bins=96):
        if observed_minutes not in (0,15,30):raise ValueError('Queries are0,+15,+30minutes')
        ix=np.atleast_1d(indices).astype(np.int64);meta=self.meta.iloc[ix]
        initial=self.base.inputs(ix,lookback_bins=lookback_bins)
        relative=np.arange(1,observed_minutes+1)
        device=meta.device_index.to_numpy()[:,None]
        minute=meta.origin_minute.to_numpy()[:,None]+relative[None,:]
        grid=self.base.grid;online=grid['online'][device,minute]
        power=np.array(grid['power_kw'][device,minute],dtype=np.float32,copy=True)
        power[~online]=np.nan
        state=np.stack([grid[n][device,minute] for n in HISTORY_STATE_NAMES],axis=-1).astype(np.float32)
        state_observed=online[...,None]&np.isfinite(state);state[~state_observed]=np.nan
        parameter=np.array(grid['command_numeric'][device,minute],dtype=np.float32,copy=True)
        command_observed=online&np.isfinite(parameter);parameter[~command_observed]=np.nan
        command=np.array(grid['command_type'][device,minute],copy=True)
        command[~command_observed]=-1
        feedback=dict(relative_minutes=relative,power_kw=power,power_observed=np.isfinite(power),
            state=state,state_observed=state_observed,command_type=command,
            command_numeric=parameter,command_observed=command_observed)
        return dict(initial=initial,feedback=feedback,observed_minutes=observed_minutes,
            target_relative_intervals=np.c_[31+15*np.arange(10),45+15*np.arange(10)],
            target_calendar=initial['future_calendar'][:,2:])

    def submission(self,path,prediction):
        prediction=np.asarray(prediction,dtype=np.float32)
        if prediction.shape!=(len(self),10) or not np.isfinite(prediction).all():
            raise ValueError('Expected all cases andtenfinite fixed-target predictions')
        np.savez_compressed(path,case_id=self.meta.case_id.to_numpy(dtype=str),
            absolute_bins=ABSOLUTE_BINS,prediction_kw=prediction)

    def score(self,prediction,method):
        return score_response_predictions(self.meta,self.target,prediction,method,self.season)


def read_response_submission(path,case_ids):
    with np.load(Path(path),allow_pickle=False) as p:
        ids=pd.Index(p['case_id'].astype(str));expected=pd.Index(np.asarray(case_ids).astype(str))
        if not ids.is_unique or not expected.is_unique or len(ids)!=len(expected):
            raise ValueError('Submission must contain the exact case set once')
        order=ids.get_indexer(expected)
        if (order<0).any():raise ValueError('Submission case set differs')
        if not np.array_equal(p['absolute_bins'],ABSOLUTE_BINS):raise ValueError('Wrong absolute target bins')
        pred=p['prediction_kw'][order]
    if pred.shape!=(len(expected),10) or not np.isfinite(pred).all():raise ValueError('Invalid predictions')
    return pred


def score_response_predictions(meta,target,prediction,method,season):
    """Date-balanced scores, equivalent to the original fixed-target pilot."""
    target=np.asarray(target,dtype=np.float64);prediction=np.asarray(prediction,dtype=np.float64)
    if target.shape!=(len(meta),10) or prediction.shape!=target.shape:raise ValueError('Expected ten fixed target bins')
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():raise ValueError('Incomplete predictions')
    if meta.duplicated(['cohort_id','alias']).any():raise ValueError('Repeated device within group')
    keys=['local_date','cohort_id','command_type'];cols=[f'r{i}' for i in range(10)]
    frame=meta[keys].copy();frame[cols]=prediction-target
    grouped=frame.groupby(keys,sort=False);group=grouped[cols].mean().reset_index()
    group['devices']=grouped.size().to_numpy();group=group[group.devices>=10].copy()
    total=group.copy();total[cols]=total[cols].to_numpy()*total.devices.to_numpy()[:,None]
    rows=[];days=[]
    for scale,table in [('device',frame),('cohort_mean',group),('cohort_total',total)]:
        for end in TARGET_ENDS:
            residual=table[cols[:(end-30)//15]].to_numpy()
            errors=table[['local_date','command_type']].copy()
            errors['mse']=np.mean(residual**2,axis=1);errors['mae']=np.mean(np.abs(residual),axis=1)
            errors['reported_energy_ae_kwh']=np.abs(residual.sum(axis=1)*.25)
            for code in ['all',1,5,8]:
                part=errors if code=='all' else errors[errors.command_type==code]
                if part.empty:continue
                daily=part.groupby('local_date')[['mse','mae','reported_energy_ae_kwh']].mean()
                shared=dict(method=method,season=season,scale=scale,command_type=str(code),
                    target_start_minutes=31,target_end_minutes=end,forecasted_bins=(end-30)//15)
                rows.append(dict(**shared,units=len(part),dates=len(daily),**daily.mean().to_dict()))
                days.append(daily.reset_index().assign(**shared))
    return pd.DataFrame(rows),pd.concat(days,ignore_index=True)
