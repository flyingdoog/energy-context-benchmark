"""Build query-time EPFL packets from the compact source cache, without fitting."""
from pathlib import Path
import numpy as np
import pandas as pd

STATE_FIELDS=("SOC (%)","TotalCapacity","RemainingCapacity","BulkCapacity","Energy (Wh)")
CONTROL_FIELDS=("Pset (W)","Preq (W)","Ctrl_t (0 = False, 1 = True)")
TARGET_NAMES=("actual_power","requested_power")
TARGET_MINUTES=np.arange(0,31,5)


def build_packet(source):
    source=Path(source)
    frame=pd.read_csv(source/"source_active_rows.csv.gz")
    frame["time"]=pd.to_datetime(frame["Date and time"])
    frame["key"]=frame.CCS.astype(str)+":"+frame.Session.astype(int).astype(str)
    support=pd.read_csv(source/"session_support.csv").set_index("key")
    rows=[];features=[];targets=[]
    for key,group in frame.groupby("key"):
        group=group.sort_values("time").set_index("time")
        start=group.index.min();end=group.index.max();query=start+pd.Timedelta(minutes=10)
        history=pd.date_range(start,periods=10,freq="min")
        if query>end or not history.isin(group.index).all():continue
        target=pd.date_range(query,periods=30,freq="min")
        if not (target.isin(group.index)|(target>end)).all():continue
        split="train" if query<pd.Timestamp("2023-01-01") else ("validation" if query<pd.Timestamp("2023-03-01") else "later")
        boundary=pd.Timestamp("2023-01-01") if split=="train" else (pd.Timestamp("2023-03-01") if split=="validation" else pd.Timestamp.max)
        if target.max()>=boundary:continue
        if support.loc[key,"start_offset_seconds"]!=0 or support.loc[key,"end_offset_seconds"]!=0:
            raise ValueError("Unverified session endpoint")
        h=group.loc[history]
        power=h["P (W)"].to_numpy()/1000
        state=h[list(STATE_FIELDS)].to_numpy().ravel()
        control=h[list(CONTROL_FIELDS)].to_numpy().ravel()
        calendar=[np.sin(2*np.pi*query.hour/24),np.cos(2*np.pi*query.hour/24),query.dayofweek,query.month]
        x=np.r_[power,calendar,state,control]
        if not np.isfinite(x).all():raise ValueError("Unexpected missing history field")
        actual=group[["P (W)","Preq (W)"]].reindex(target)
        actual.loc[target>end]=0
        y=actual.to_numpy().reshape(6,5,2).mean(axis=1)/1000
        if not np.isfinite(y).all():raise ValueError("Missing within-session target")
        rows.append(dict(case_id=key,plug=group.CCS.iloc[0],split=split,date=query.strftime("%Y-%m-%d"),
                         query=str(query),history_controlled=bool(h[CONTROL_FIELDS[-1]].iloc[-1]),
                         last_actual_kw=power[-1],last_requested_kw=h["Preq (W)"].iloc[-1]/1000))
        features.append(x);targets.append(y)
    return pd.DataFrame(rows),np.asarray(features),np.asarray(targets)

