"""Common model-neutral power and optional occurrence task interface."""
from pathlib import Path
import copy
import numpy as np
import pandas as pd
from .response import BpaResponseTask,score_response_predictions
from .acn_dataset import AcnDataset
from .acn_scoring import score_acn_residuals
from .epfl_dataset import EpflDataset
from .cu_bems_dataset import CuBemsDataset
from .probability_scoring import activity_probability_scores
from .activity_references import bpa_query_power,score_activity_cohorts
from .telemetry_availability import validate_blackout,blackout_native_inputs


class OperationalForecastTask:
    """Wrap prepared packets without rebuilding data or importing model libraries.

    Score/submit an entire named split. Arbitrary input sampling is supported,
    but partial prediction cohorts cannot silently replace the scoring population.
    """
    def __init__(self,root,source,*,season=None,site=None):
        self.root=Path(root);self.source=source.upper();self.season=season or ''
        if self.source=='BPA':
            if season not in ('winter','spring','summer') or site is not None:
                raise ValueError('BPA requires a native season and no site argument')
            self.base=BpaResponseTask(self.root,season);self.meta=self.base.meta.copy()
            self.meta['split']=season;self.site='Field study';self.width=10;self.step=15
            if season=='winter':
                with np.load(self.root/'update_validation/split.npz',allow_pickle=False) as p:
                    train=p['train'];validation=p['validation']
                if np.intersect1d(train,validation).size:raise ValueError('Training and validation overlap')
                if ((np.r_[train,validation]<0)|(np.r_[train,validation]>=len(self.meta))).any():
                    raise ValueError('Split indices outside data')
                self.meta['split']='boundary_excluded'
                self.meta.loc[self.meta.index[train],'split']='train'
                self.meta.loc[self.meta.index[validation],'split']='validation'
            self._native=np.arange(len(self.meta));self.target_name='reported_power'
            first=1;self.population_scope='Original label-eligible devices in synchronous groups of at least ten'
        elif self.source=='ACN':
            if season is not None or site not in ('caltech','office001'):
                raise ValueError('ACN requires site caltech or office001 and no season')
            self.base=AcnDataset(self.root)
            self._native=np.flatnonzero(self.base.meta.site==site)
            self.meta=self.base.meta.iloc[self._native].copy();self.site=site
            self.width=10;self.step=15;self.target_name='reported_power';first=0
            self.population_scope='Complete query-known cohorts with at least two distinct stations'
        elif self.source=='EPFL':
            if season is not None or site is not None:raise ValueError('EPFL has no season/site selector')
            self.base=EpflDataset(self.root,target='actual_power');self.meta=self.base.meta.copy()
            self.meta['local_date']=self.meta.date;self._native=np.arange(len(self.meta))
            self.site='Fast charging';self.width=6;self.step=5;self.target_name='actual_power';first=0
            self.population_scope='Eligible fast-charging sessions; device-only'
        elif self.source=='CU-BEMS':
            if season is not None or site not in tuple(f'floor{i}' for i in range(1,8)):
                raise ValueError('CU-BEMS requires site floor1 through floor7 and no season')
            self.base=CuBemsDataset(self.root,floor=int(site[5:]))
            self.meta=self.base.meta.copy();self._native=np.arange(len(self.meta));self.site=site
            self.width=10;self.step=15;self.target_name='reported_power';first=0
            self.population_scope='Eligible AC queries; complete fixed floor/zone membership for group scores'
        else:raise ValueError('Source must be BPA, ACN, EPFL or CU-BEMS')
        self.meta=self.meta.reset_index(drop=True)
        if not len(self.meta) or not self.meta.case_id.is_unique:raise ValueError('Unique nonempty source cases required')
        self.meta['case_id']=self.meta.case_id.astype(str)
        self.report_offsets=np.c_[first+self.step*np.arange(self.width),first+self.step*np.arange(self.width)+self.step-1]
        self.supports_groups=self.source!='EPFL'

    def __len__(self):return len(self.meta)

    @property
    def contract(self):
        return dict(source=self.source,site=self.site,season=self.season,target_name=self.target_name,
            power_unit='kW',bin_minutes=self.step,report_offsets_minutes=self.report_offsets.copy(),
            report_offsets_convention='Inclusive offsets of native minute labels relative to query',
            native_time_semantics='Point reports after query; bin mean of observed reports' if self.source=='BPA' else
                'Minute-duration means on left-labelled [minute, minute+1) intervals' if self.source=='ACN' else
                'Native minute records, averaged in blocks beginning at query',
            query_shift_from_original_minutes=30 if self.source=='BPA' else 0,
            population_scope=self.population_scope,supports_groups=self.supports_groups,
            available_conditions=getattr(self.base,'available_conditions',('power','power_state','full')))

    def indices(self,split):
        if split=='boundary_excluded':raise ValueError('Boundary cases are excluded from the experiment')
        ix=np.flatnonzero(self.meta.split==split)
        if not len(ix):raise ValueError(f'No supported cases for split {split}')
        return ix

    def _indices(self,indices):
        if indices is None:return np.arange(len(self))
        ix=np.atleast_1d(indices)
        if ix.dtype.kind=='b':
            if ix.shape!=(len(self),):raise ValueError('Boolean index shape differs')
            ix=np.flatnonzero(ix)
        if ix.ndim!=1 or ix.dtype.kind not in 'iu' or ((ix<0)|(ix>=len(self))).any():
            raise ValueError('Use in-range integer indices or a complete boolean mask')
        return ix.astype(int)

    def targets(self,indices=None):
        ix=self._indices(indices)
        return np.array(self.base.target[self._native[ix]],copy=True)

    def current_power(self,indices=None):
        ix=self._indices(indices);native=self._native[ix]
        if self.source=='BPA':return bpa_query_power(self.base.base,native)
        p=self.base.inputs(native,condition='power')
        return p['power_kw'][:,-1,0].copy(),p['power_observed'][:,-1,0].copy()

    def inputs(self,indices=None,condition='full',*,auxiliary_blackout_minutes=0):
        """Query-time inputs, optionally without the latest auxiliary reports.

        A positive D withholds state/control timestamps >= query-D. Power,
        calendar, targets and case selection are unchanged. The default and
        power-only output are unchanged; positive auxiliary views describe the
        synthetic cutoff in their returned availability metadata.
        """
        if condition not in ('power','power_state','full'):raise ValueError('Unknown common input condition')
        blackout=validate_blackout(self.source,auxiliary_blackout_minutes)
        ix=self._indices(indices);native=self._native[ix]
        if self.source=='BPA':
            p=self.base.refreshed_inputs(native,observed_minutes=30)
            # The existing tabular reference also receives lead-aligned day/week
            # histories. Preserve that information in every common input view.
            view=copy.copy(self.base.base);view.meta=self.base.meta.copy()
            view.meta['origin_minute']=view.meta.origin_minute+30
            for label,lag in [('day',1440),('week',10080)]:
                starts=1+15*np.arange(12)-lag
                values,fraction=view.power_windows(native,starts)
                p[f'previous_{label}_profile_kw']=values
                p[f'previous_{label}_profile_observed_fraction']=fraction
                p[f'previous_{label}_profile_report_offsets']=np.c_[starts,starts+14]
            p['query_power_kw']=p['current_state'][:,0].copy()
            p['query_power_observed']=p['current_state_observed'][:,0].copy()
            if condition=='power':
                for key in ['past_state','past_state_observed','current_state','current_state_observed']:p.pop(key)
            if condition!='full':
                for key in ['past_command_type','past_command_parameter','past_command_observed',
                            'current_command','current_command_observed']:p.pop(key)
        else:p=self.base.inputs(native,condition=condition)
        result=dict(case_id=self.meta.case_id.to_numpy()[ix].copy(),condition=condition,
                    contract=self.contract,native_inputs=p)
        if blackout and condition!='power':
            result['native_inputs']=blackout_native_inputs(p,self.source,blackout)
            result['availability']=dict(auxiliary_blackout_minutes=blackout,
                retained_auxiliary_report_offset_minutes_less_than=-blackout,
                synthetic=True,power_inputs_unchanged=True)
        return result

    def _values(self,values,n,probability=False):
        if values is None:return None
        value=np.asarray(values,dtype=float)
        if value.shape!=(n,self.width) or not np.isfinite(value).all():raise ValueError('Forecast shape or values differ')
        if probability and ((value<0)|(value>1)).any():raise ValueError('Probability must be within [0,1]')
        return value

    def submission(self,path,*,split,prediction_kw=None,positive_probability=None):
        ix=self.indices(split);power=self._values(prediction_kw,len(ix));prob=self._values(positive_probability,len(ix),True)
        if power is None and prob is None:raise ValueError('At least one prediction track required')
        payload=dict(case_id=self.meta.case_id.to_numpy(dtype=str)[ix],source=self.source,site=self.site,season=self.season,
            split=split,target_name=self.target_name,report_offsets_minutes=self.report_offsets,
            schema_version='operational-forecast-v1')
        if power is not None:payload['prediction_kw']=power
        if prob is not None:payload['positive_probability']=prob
        np.savez_compressed(path,**payload)

    def read_submission(self,path,*,split):
        ix=self.indices(split)
        with np.load(path,allow_pickle=False) as p:
            expected=dict(source=self.source,site=self.site,season=self.season,split=split,
                target_name=self.target_name,schema_version='operational-forecast-v1')
            if any(k not in p or str(p[k].item())!=v for k,v in expected.items()):raise ValueError('Submission task metadata differs')
            if not np.array_equal(p['report_offsets_minutes'],self.report_offsets):raise ValueError('Submission target times differ')
            ids=pd.Index(p['case_id'].astype(str));order=ids.get_indexer(self.meta.case_id.iloc[ix]) if ids.is_unique else np.array([-1])
            if not ids.is_unique or len(ids)!=len(ix) or (order<0).any():raise ValueError('Submission case set differs')
            values={name:self._values(p[name][order],len(ix),name=='positive_probability')
                    for name in ['prediction_kw','positive_probability'] if name in p}
        if not values:raise ValueError('Submission contains no prediction track')
        return values

    def _annotate(self,frames,split):
        return tuple(frame.assign(source=self.source,site=self.site,period=split) for frame in frames)

    def score(self,*,split,prediction_kw=None,positive_probability=None,method='submission',require_groups=False):
        if require_groups and not self.supports_groups:raise ValueError('EPFL has no supported cohort track')
        ix=self.indices(split);meta=self.meta.iloc[ix];y=self.targets(ix)
        power=self._values(prediction_kw,len(ix));prob=self._values(positive_probability,len(ix),True);result={}
        if power is None and prob is None:raise ValueError('At least one prediction track required')
        if power is not None:
            if self.source=='BPA':
                summary,daily=score_response_predictions(meta,y,power,method,self.season)
                summary=summary[summary.command_type=='all'].copy();daily=daily[daily.command_type=='all'].copy()
                for frame in [summary,daily]:
                    frame['span_minutes']=frame.target_end_minutes-30
                    frame.rename(columns={'reported_energy_ae_kwh':'energy_ae_kwh','local_date':'date'},inplace=True)
            elif self.source=='ACN':
                summary,daily=score_acn_residuals(meta,power-y,method,self.base.population)
                daily=daily.rename(columns={'local_date':'date'})
            elif self.source=='CU-BEMS':
                summary,daily=self.base.score(power,method,indices=self._native[ix])
            else:
                summary,daily=self.base.score(power,method,indices=self._native[ix])
                summary=summary.assign(scale='device',units=len(ix),dates=meta.local_date.nunique())
                daily=daily.assign(scale='device')
            keys=['scale','span_minutes','method','mse','mae','energy_ae_kwh']
            summary=summary[keys+['units','dates']];daily=daily[keys+['date']]
            for frame in [summary,daily]:
                frame['mse_unit']='kW squared';frame['mae_unit']='kW';frame['energy_ae_unit']='kWh'
            result['power']=self._annotate((summary,daily),split)
        if prob is not None:
            days=[]
            for bins in ([1,3,6] if self.source=='EPFL' else [2,6,10]):
                daily,_=activity_probability_scores(meta.local_date,y,prob,bins)
                days.append(daily.assign(span_minutes=bins*self.step,method=method,scale='device'))
            daily=pd.concat(days,ignore_index=True)
            summary=daily.groupby(['span_minutes','method','scale']).brier.mean().reset_index()
            summary=summary.assign(units=len(ix),dates=meta.local_date.nunique(),brier_unit='dimensionless')
            result['probability']=self._annotate((summary,daily),split)
            if self.supports_groups:
                if self.source=='CU-BEMS':
                    frames=self.base.score(prob,method,indices=self._native[ix],activity=True)
                else:
                    frames=score_activity_cohorts(self.source,meta,y,prob,method,split,
                        self.base.population if self.source=='ACN' else None)
                result['group_activity']=self._annotate(frames,split)
        return result
