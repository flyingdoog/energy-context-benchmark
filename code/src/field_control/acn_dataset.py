"""Prepared ACN packets, typed history inputs and portable submissions."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .acn import STATES
from .acn_scoring import score_acn_residuals

TARGET_MINUTES=np.arange(0,151,15)
CONDITIONS=('power','power_commands','power_state','full')


class AcnDataset:
    def __init__(self,root):
        self.root=Path(root)
        self.receipt=json.loads((self.root/'packet_receipt.json').read_text())
        self.meta=pd.read_csv(self.root/'cases.csv.gz')
        self.population=pd.read_csv(self.root/'query_population.csv.gz')
        with np.load(self.root/'packet.npz',allow_pickle=False) as packet:
            self._x,self.target=packet['x'],packet['y']
        if self._x.shape[0]!=len(self.meta) or self.target.shape!=(len(self.meta),10):
            raise ValueError('Packet shapes do not match case metadata')
        if not self.meta.case_id.is_unique:raise ValueError('Repeated case IDs')

    def __len__(self):return len(self.meta)

    @property
    def feature_groups(self):return self.receipt['feature_groups']

    def features(self,indices=None,condition='full'):
        if condition not in CONDITIONS:raise ValueError('Unknown information condition')
        ix=np.arange(len(self)) if indices is None else np.atleast_1d(indices).astype(int)
        x=self._x[ix].copy()
        for name,omit in [('control',condition in ('power','power_state')),('state',condition in ('power','power_commands'))]:
            if omit:
                a,b=self.feature_groups[name];x[:,a:b]=0
        return x

    def inputs(self,indices=None,condition='full',declared=False):
        if condition not in CONDITIONS:raise ValueError('Unknown information condition')
        ix=np.arange(len(self)) if indices is None else np.atleast_1d(indices).astype(int)
        x=self._x[ix];groups=self.feature_groups
        out=dict(history_intervals_minutes=np.c_[np.arange(-60,0),np.arange(-59,1)],
                 target_intervals_minutes=np.c_[TARGET_MINUTES[:-1],TARGET_MINUTES[1:]],
                 calendar=x[:,slice(*groups['calendar'])].copy())
        def unpack(name,channels):
            a,b=groups[name];raw=x[:,a:b];n=60*channels
            values=raw[:,:n].reshape(len(ix),60,channels).copy()
            observed=raw[:,n:2*n].reshape(len(ix),60,channels).astype(bool)
            values[~observed]=np.nan
            return values,observed
        out['power_kw'],out['power_observed']=unpack('power',1)
        if condition in ('power_commands','full'):
            out['pilot_a'],out['pilot_observed']=unpack('control',1)
        if condition in ('power_state','full'):
            out['numeric_state'],out['numeric_state_observed']=unpack('state',3)
            out['numeric_state_names']=('current_a','voltage_v','reported_cumulative_energy_kwh')
            raw=x[:,slice(*groups['state'])]
            categories=raw[:,360:780].reshape(len(ix),60,len(STATES)).copy()
            observed=raw[:,780:1200].reshape(len(ix),60,len(STATES)).astype(bool)
            categories[~observed]=np.nan
            out.update(operating_state_fractions=categories,operating_state_observed=observed,operating_state_names=STATES)
        if declared:
            out['declarations']=self.declarations(ix)
        return out

    def declarations(self,indices=None):
        """Optional as-of user requests; no realized departure or final energy."""
        if not hasattr(self,'_requests'):
            path=self.root/'request_metadata_v1/query_requests.csv.gz'
            requests=pd.read_csv(path)
            if not requests.case_id.is_unique:raise ValueError('Repeated declaration case IDs')
            requests=requests.set_index('case_id').reindex(self.meta.case_id)
            if requests.site.isna().any():raise ValueError('Missing declaration metadata row')
            modified=pd.to_datetime(requests.modified_at,utc=True,format='mixed',errors='coerce')
            query=pd.to_datetime(self.meta.query_utc,utc=True,format='mixed')
            if (modified.to_numpy()>query.to_numpy()).any():
                raise ValueError('Declaration modified after its forecast query')
            self._requests=requests
        ix=np.arange(len(self)) if indices is None else np.atleast_1d(indices).astype(int)
        requests=self._requests.iloc[ix]
        energy=requests.requested_kwh.to_numpy(dtype=float)
        departure=requests.requested_departure_hours.to_numpy(dtype=float)
        timestamped=pd.to_datetime(requests.modified_at,utc=True,format='mixed',errors='coerce').notna().to_numpy()
        energy_known=timestamped&np.isfinite(energy)&(energy>=0)
        departure_known=timestamped&np.isfinite(departure)
        return dict(requested_kwh=np.where(energy_known,energy,np.nan),
            requested_departure_hours=np.where(departure_known,departure,np.nan),
            request_observed=energy_known,departure_observed=departure_known)

    def submission(self,path,prediction):
        prediction=np.asarray(prediction,dtype=np.float32)
        if prediction.shape!=self.target.shape or not np.isfinite(prediction).all():
            raise ValueError('Expected a finite forecast for every case and target bin')
        np.savez_compressed(path,case_id=self.meta.case_id.to_numpy(dtype=str),target_minutes=TARGET_MINUTES,prediction_kw=prediction)

    def read_submission(self,path):
        with np.load(path,allow_pickle=False) as stored:
            ids=pd.Index(stored['case_id'].astype(str))
            if len(ids)!=len(self) or not ids.is_unique:raise ValueError('Submission case set differs')
            order=ids.get_indexer(self.meta.case_id)
            if (order<0).any() or not np.array_equal(stored['target_minutes'],TARGET_MINUTES):
                raise ValueError('Wrong case IDs or future intervals')
            values=stored['prediction_kw'][order]
        if values.shape!=self.target.shape or not np.isfinite(values).all():raise ValueError('Invalid predictions')
        return values

    def score(self,prediction,method='submission'):
        prediction=np.asarray(prediction,dtype=float)
        if prediction.shape!=self.target.shape:raise ValueError('Wrong prediction shape')
        return score_acn_residuals(self.meta,prediction-self.target,method,self.population)
