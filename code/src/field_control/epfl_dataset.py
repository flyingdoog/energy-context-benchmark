"""Model-independent EPFL history inputs, target-labelled submissions and scores."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .epfl_preparation import STATE_FIELDS,TARGET_NAMES,TARGET_MINUTES

CONDITIONS=("power","power_state","full")


class EpflDataset:
    def __init__(self,root,target="actual_power"):
        if target not in TARGET_NAMES:raise ValueError("Unknown EPFL target")
        self.root=Path(root);self.target_name=target
        self.receipt=json.loads((self.root/"receipt.json").read_text())
        self.meta=pd.read_csv(self.root/"cases.csv")
        with np.load(self.root/"packet.npz",allow_pickle=False) as packet:
            self._x=packet["x"];self._targets=packet["y"]
            ids=packet["case_id"].astype(str)
        if not self.meta.case_id.is_unique or not np.array_equal(ids,self.meta.case_id.to_numpy(dtype=str)):
            raise ValueError("EPFL packet/case IDs differ")
        if self._x.shape!=(len(self.meta),94) or self._targets.shape!=(len(self.meta),6,2):
            raise ValueError("EPFL v1 packet dimensions differ")
        if not (np.isfinite(self._x).all() and np.isfinite(self._targets).all()):
            raise ValueError("EPFL complete-session packet contains missing values")
        self.target=self._targets[:,:,TARGET_NAMES.index(target)]

    def __len__(self):return len(self.meta)

    def _indices(self,indices):
        if indices is None:return np.arange(len(self))
        raw=np.atleast_1d(indices)
        if raw.dtype.kind=="b":
            if raw.shape!=(len(self),):raise ValueError("Boolean case mask has wrong length")
            return np.flatnonzero(raw)
        if raw.ndim!=1 or raw.dtype.kind not in "iu":
            raise ValueError("Use integer indices or a full-length boolean mask")
        if ((raw<0)|(raw>=len(self))).any():raise ValueError("Case index outside packet")
        return raw.astype(int)

    @property
    def feature_groups(self):
        return dict(power=[0,10],calendar=[10,14],state=[14,64],operational=[64,94],
                    requested_history=list(range(65,94,3)))

    def features(self,indices=None,condition="full",include_target_history=True):
        if condition not in CONDITIONS:raise ValueError("Unknown information condition")
        x=self._x[self._indices(indices)].copy()
        if condition=="power":x[:,14:64]=0
        if condition in ("power","power_state"):
            x[:,64:94]=0
            if self.target_name=="requested_power" and include_target_history:
                ix=self._indices(indices)
                x[:,65:94:3]=self._x[ix,65:94:3]
        return x

    def inputs(self,indices=None,condition="full",include_target_history=True):
        if condition not in CONDITIONS:raise ValueError("Unknown information condition")
        x=self._x[self._indices(indices)]
        n=len(x)
        out=dict(target_name=self.target_name,history_offsets_minutes=np.arange(-10,0),
                 target_intervals_minutes=np.c_[TARGET_MINUTES[:-1],TARGET_MINUTES[1:]],
                 power_kw=x[:,:10,None].copy(),power_observed=np.ones((n,10,1),dtype=bool),
                 calendar=x[:,10:14].copy(),calendar_names=("hour_sin","hour_cos","weekday","month"))
        if condition in ("power_state","full"):
            out["numeric_state"]=x[:,14:64].reshape(n,10,5).copy()
            out["numeric_state_observed"]=np.ones((n,10,5),dtype=bool)
            out["numeric_state_names"]=STATE_FIELDS
            out["numeric_state_units"]=("%","source_native","source_native","source_native","Wh")
        requested_available=condition=="full" or (self.target_name=="requested_power" and include_target_history)
        control=x[:,64:94].reshape(n,10,3)
        if requested_available:
            out["requested_kw"]=control[:,:,1:2].copy()/1000
            out["requested_observed"]=np.ones((n,10,1),dtype=bool)
        if condition=="full":
            out["setpoint_kw"]=control[:,:,:1].copy()/1000
            out["setpoint_observed"]=np.ones((n,10,1),dtype=bool)
            out["controlled"]=control[:,:,2:3].astype(bool)
            out["controlled_observed"]=np.ones((n,10,1),dtype=bool)
        if self.target_name=="actual_power":
            out["target_history_kw"]=out["power_kw"].copy()
        elif requested_available:
            out["target_history_kw"]=out["requested_kw"].copy()
        return out

    def _prediction(self,prediction,n):
        values=np.asarray(prediction,dtype=float)
        if values.shape!=(n,6) or not np.isfinite(values).all():
            raise ValueError("Expected a finite forecast for every requested case and six bins")
        return values

    def submission(self,path,prediction):
        values=self._prediction(prediction,len(self))
        np.savez_compressed(path,case_id=self.meta.case_id.to_numpy(dtype=str),
                            target_name=self.target_name,target_minutes=TARGET_MINUTES,prediction_kw=values)

    def read_submission(self,path):
        with np.load(path,allow_pickle=False) as stored:
            ids=pd.Index(stored["case_id"].astype(str))
            if len(ids)!=len(self) or not ids.is_unique:raise ValueError("Submission case set differs")
            order=ids.get_indexer(self.meta.case_id)
            if (order<0).any() or not np.array_equal(stored["target_minutes"],TARGET_MINUTES):
                raise ValueError("Wrong case IDs or future intervals")
            if "target_name" not in stored or str(stored["target_name"].item())!=self.target_name:
                raise ValueError("Submission target differs")
            values=stored["prediction_kw"][order]
        return self._prediction(values,len(self))

    def score(self,prediction,method="submission",indices=None):
        ix=self._indices(indices);values=self._prediction(prediction,len(ix))
        if not len(ix) or len(np.unique(ix))!=len(ix):raise ValueError("Score unique nonempty cases")
        meta=self.meta.iloc[ix];target=self.target[ix];rows=[]
        for bins in [1,3,6]:
            error=values[:,:bins]-target[:,:bins]
            frame=meta[["split","date"]].reset_index(drop=True).copy()
            frame["mse"]=np.mean(error**2,axis=1)
            frame["mae"]=np.mean(abs(error),axis=1)
            frame["energy_ae_kwh"]=abs(error.sum(axis=1)*5/60)
            daily=frame.groupby(["split","date"]).mean().reset_index()
            rows.append(daily.assign(method=method,target=self.target_name,span_minutes=bins*5))
        daily=pd.concat(rows,ignore_index=True)
        summary=daily.groupby(["split","target","span_minutes","method"])[["mse","mae","energy_ae_kwh"]].mean().reset_index()
        return summary,daily

