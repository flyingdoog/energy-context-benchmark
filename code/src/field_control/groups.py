"""Synchronous fleet membership, pooled inputs, and direct fleet scoring."""
from pathlib import Path
import numpy as np
import pandas as pd
from .response import ABSOLUTE_BINS,TARGET_ENDS


class GroupLayout:
    """Group membership depends on published case metadata, never predictions."""
    def __init__(self,meta,min_devices=10):
        # Only BPA's sparse pooling needs scipy. Other source adapters can be
        # imported and scored with numpy/pandas alone.
        from scipy.sparse import csr_matrix
        keys=['local_date','cohort_id','command_type']
        if min_devices<1 or meta.duplicated(['cohort_id','alias']).any():
            raise ValueError('Invalid group size or duplicate device in a group')
        codes,unique=pd.factorize(pd.MultiIndex.from_frame(meta[keys]),sort=False)
        counts=np.bincount(codes,minlength=len(unique));keep=counts>=min_devices
        mapping=np.full(len(counts),-1,dtype=int);mapping[keep]=np.arange(keep.sum())
        self.meta=unique.set_names(keys).to_frame(index=False).loc[keep].reset_index(drop=True)
        self.meta['devices']=counts[keep]
        if not self.meta.cohort_id.is_unique:raise ValueError('Group IDs must be unique')
        self.case_count=len(meta);self.codes=mapping[codes]
        take=self.codes>=0
        self.members=csr_matrix((np.ones(take.sum()),(self.codes[take],np.flatnonzero(take))),
                               shape=(len(self.meta),len(meta)))

    def pool(self,values):
        """Column means over finite member values; existing masks pool too.

        One-hot means are category fractions. All-missing columns remain NaN.
        No category code is averaged as a numerical magnitude.
        """
        values=np.asarray(values)
        if values.ndim!=2 or len(values)!=self.case_count:raise ValueError('Wrong case matrix')
        good=np.isfinite(values)
        numerator=self.members@np.where(good,values,0)
        denominator=self.members@good.astype(np.float64)
        return np.divide(numerator,denominator,out=np.full(numerator.shape,np.nan),where=denominator>0)

    def submission(self,path,prediction):
        prediction=np.asarray(prediction,dtype=np.float32)
        if prediction.shape!=(len(self.meta),10) or not np.isfinite(prediction).all():
            raise ValueError('Expected finite group mean predictions for ten bins')
        np.savez_compressed(path,cohort_id=self.meta.cohort_id.to_numpy(dtype=str),
            absolute_bins=ABSOLUTE_BINS,prediction_kw=prediction)

    def read_submission(self,path):
        with np.load(Path(path),allow_pickle=False) as p:
            ids=pd.Index(p['cohort_id'].astype(str));expected=pd.Index(self.meta.cohort_id.astype(str))
            if not ids.is_unique or len(ids)!=len(expected):raise ValueError('Wrong group set')
            order=ids.get_indexer(expected)
            if (order<0).any() or not np.array_equal(p['absolute_bins'],ABSOLUTE_BINS):
                raise ValueError('Wrong group set or absolute target bins')
            prediction=p['prediction_kw'][order]
        if prediction.shape!=(len(expected),10) or not np.isfinite(prediction).all():
            raise ValueError('Invalid group forecast')
        return prediction


def score_group_predictions(meta,target,prediction,method,season):
    target=np.asarray(target,dtype=float);prediction=np.asarray(prediction,dtype=float)
    if target.shape!=(len(meta),10) or prediction.shape!=target.shape:
        raise ValueError('Wrong group forecast shape')
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():raise ValueError('Nonfinite group values')
    if not meta.cohort_id.is_unique:raise ValueError('Repeated group')
    rows=[];days=[]
    for scale in ['cohort_mean','cohort_total']:
        residual=prediction-target
        if scale=='cohort_total':residual=residual*meta.devices.to_numpy()[:,None]
        for end in TARGET_ENDS:
            error=residual[:,:(end-30)//15]
            frame=meta[['local_date','command_type']].copy()
            frame['mse']=(error**2).mean(axis=1);frame['mae']=np.abs(error).mean(axis=1)
            frame['reported_energy_ae_kwh']=np.abs(error.sum(axis=1)*.25)
            for code in ['all',1,5,8]:
                part=frame if code=='all' else frame[frame.command_type==code]
                if part.empty:continue
                daily=part.groupby('local_date')[['mse','mae','reported_energy_ae_kwh']].mean()
                shared=dict(method=method,season=season,scale=scale,command_type=str(code),
                    target_start_minutes=31,target_end_minutes=end,forecasted_bins=(end-30)//15)
                rows.append(dict(**shared,units=len(part),dates=len(daily),**daily.mean().to_dict()))
                days.append(daily.reset_index().assign(**shared))
    return pd.DataFrame(rows),pd.concat(days,ignore_index=True)
