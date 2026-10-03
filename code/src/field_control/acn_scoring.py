"""Model-independent ACN device/cohort scoring."""
import numpy as np
import pandas as pd


def complete_cohort_mask(meta,population):
    keys=['site','split','query_utc']
    expected=population.groupby(keys).size()
    present=meta.groupby(keys).size()
    good=present.index[present==expected.reindex(present.index)]
    return pd.MultiIndex.from_frame(meta[keys]).isin(good)


def score_acn_residuals(meta,residual,method,population=None):
    residual=np.asarray(residual,dtype=float)
    if residual.shape!=(len(meta),10) or not np.isfinite(residual).all():
        raise ValueError('Expected ten finite residuals for each case')
    rows,daily_rows=[],[];cols=[f'e{j}' for j in range(10)]
    frame=meta[['site','split','local_date','query_utc','station_id']].copy()
    frame[cols]=residual
    frame['cohort_available']=True if population is None else complete_cohort_mask(meta,population)
    for (site,split),period in frame.groupby(['site','split'],sort=False):
        duplicates=period.duplicated(['query_utc','station_id'],keep=False)
        bad=set(period.loc[duplicates,'query_utc'])
        clean=period[(~period.query_utc.isin(bad))&period.cohort_available]
        grouped=clean.groupby(['local_date','query_utc'])
        mean=grouped[cols].mean();size=grouped.size();mean=mean[size>=2]
        total=mean.mul(size[size>=2],axis=0)
        for scale,table in [('device',period),('cohort_mean',mean.reset_index()),('cohort_total',total.reset_index())]:
            if table.empty:continue
            for bins in (2,6,10):
                errors=table[cols[:bins]].to_numpy()
                metrics=pd.DataFrame(dict(local_date=table.local_date.to_numpy(),mse=np.mean(errors**2,axis=1),
                    mae=np.mean(abs(errors),axis=1),energy_ae_kwh=abs(errors.sum(axis=1)*.25)))
                day=metrics.groupby('local_date').mean()
                info=dict(site=site,split=split,method=method,scale=scale,span_minutes=bins*15)
                rows.append(dict(**info,units=len(table),dates=len(day),**day.mean().to_dict()))
                daily_rows.append(day.reset_index().assign(**info))
    return pd.DataFrame(rows),pd.concat(daily_rows,ignore_index=True) if daily_rows else pd.DataFrame()
