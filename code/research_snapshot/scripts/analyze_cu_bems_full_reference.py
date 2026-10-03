"""All-floor tables and date-paired comparisons; no forecast/model changes."""
import argparse
import json
from pathlib import Path
import time
import run_local
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from run_cu_bems_full_reference import OUT as BASE
from run_cu_bems_reference import dump,frame_json
from analyze_count_distributions import calendar_weights,paired_statistics

OUT=BASE/'analysis'
METRICS=['mse','mae','energy_ae_kwh']
METHODS=['Persistence','Ridge: power','Ridge: thermal','Tree: power','Tree: thermal']
PERIODS=[('later_1','Sep-Oct'),('later_2','Nov-Dec')]
COHORTS={'power_six_floors':([1,2,3,4,5,7],['Persistence','Ridge: power','Tree: power']),
         'thermal_five_floors':([2,3,4,5,7],METHODS)}


def common_daily(block,floors,methods,metric):
    selected=block[block.floor.isin(floors)&block.method.isin(methods)]
    pivot=selected.pivot(index='date',columns=['floor','method'],values=metric).sort_index()
    expected=pd.MultiIndex.from_product([floors,methods],names=['floor','method'])
    pivot=pivot.reindex(columns=expected).dropna(how='any')
    if not len(pivot):raise ValueError('No dates shared by all prescribed floors and methods')
    # Same calendar dates and equal floor weights, not independent floor resamples.
    return pd.DataFrame({m:pivot.xs(m,axis=1,level='method').mean(axis=1) for m in methods})


def main(include_neural=False):
    started=time.monotonic()
    if not json.loads((BASE/'receipt.json').read_text())['completed']:
        raise ValueError('Complete all planned default fits first')
    out=OUT if not include_neural else BASE.parent/'cu_bems_neural_reference_v1/analysis'
    methods=list(METHODS);families=['Ridge','Tree']
    cohorts={name:(list(floors),list(models)) for name,(floors,models) in COHORTS.items()}
    out.mkdir(parents=True,exist_ok=True)
    native=pd.read_json(BASE/'summary.json')
    daily=pd.read_json(BASE/'daily_scores.json',convert_dates=False)
    if include_neural:
        neural=BASE.parent/'cu_bems_neural_reference_v1/scores'
        status=json.loads((neural/'receipt.json').read_text())
        if not status['completed'] or status['completed_training_runs']!=39:
            raise ValueError('Complete and score all 39 N-HiTS fits first')
        native=pd.concat([native,pd.read_json(neural/'summary.json')],ignore_index=True)
        daily=pd.concat([daily,pd.read_json(neural/'daily_scores.json',convert_dates=False)],ignore_index=True)
        methods+=['N-HiTS: power','N-HiTS: thermal'];families+=['N-HiTS']
        cohorts['power_six_floors'][1].append('N-HiTS: power')
        cohorts['thermal_five_floors'][1].extend(['N-HiTS: power','N-HiTS: thermal'])
    daily['date']=pd.to_datetime(daily.date).dt.strftime('%Y-%m-%d')
    keys=['floor','split','scale','method','date']
    average=daily.groupby(keys,as_index=False)[METRICS[:2]].mean()
    average['span_minutes']=0
    scores=pd.concat([daily,average],ignore_index=True)
    point=average.groupby(['floor','split','scale','method'],as_index=False)[METRICS[:2]].mean()
    frame_json(out/'span_average.json',point)
    weights_cache={};contrasts=[]
    def weights_for(dates):
        key=tuple(dates)
        if key not in weights_cache:weights_cache[key]=calendar_weights(key)
        return weights_cache[key]
    for (floor,split,scale,span),block in scores.groupby(['floor','split','scale','span_minutes']):
        if floor==1:continue
        for metric in (METRICS[:2] if span==0 else METRICS):
            pivot=block.pivot(index='date',columns='method',values=metric).sort_index()
            if pivot.isna().any().any() or set(pivot.columns)!=set(methods):
                raise ValueError('Methods differ in floor/date support')
            weights,weeks=weights_for(pivot.index)
            for family in families:
                contrasts.append(dict(floor=int(floor),split=split,scale=scale,span_minutes=int(span),metric=metric,
                    family=family,dates=len(pivot),weeks=weeks,
                    **paired_statistics(pivot[f'{family}: power'],pivot[f'{family}: thermal'],weights)))
    contrasts=pd.DataFrame(contrasts);frame_json(out/'paired_intervals.json',contrasts)
    common=[];common_contrasts=[]
    for cohort,(floors,cohort_methods) in cohorts.items():
        for (split,scale,span),block in scores[scores.scale.isin(['device','floor_total','floor_mean'])].groupby(['split','scale','span_minutes']):
            for metric in (METRICS[:2] if span==0 else METRICS):
                pivot=common_daily(block,floors,cohort_methods,metric)
                weights,weeks=weights_for(pivot.index)
                for method in cohort_methods:
                    common.append(dict(cohort=cohort,split=split,scale=scale,span_minutes=int(span),metric=metric,
                        method=method,value=float(pivot[method].mean()),dates=len(pivot),floor_count=len(floors),
                        first_date=pivot.index[0],last_date=pivot.index[-1]))
                pairs=[('Persistence','Tree: power'),('Ridge: power','Tree: power')]
                if cohort=='thermal_five_floors':pairs+=[('Ridge: power','Ridge: thermal'),('Tree: power','Tree: thermal')]
                if include_neural:
                    pairs+=[('Tree: power','N-HiTS: power')]
                    if cohort=='thermal_five_floors':
                        pairs+=[('N-HiTS: power','N-HiTS: thermal'),('Tree: power','N-HiTS: thermal')]
                for a,b in pairs:
                    common_contrasts.append(dict(cohort=cohort,split=split,scale=scale,span_minutes=int(span),metric=metric,
                        reference_method=a,candidate_method=b,dates=len(pivot),weeks=weeks,
                        **paired_statistics(pivot[a],pivot[b],weights)))
    common=pd.DataFrame(common);frame_json(out/'common_cohort_scores.json',common)
    common_contrasts=pd.DataFrame(common_contrasts);frame_json(out/'common_cohort_intervals.json',common_contrasts)
    lines=['# CU-BEMS: full default-reference comparison','',
        'MSE (kW²) / MAE (kW), averaged over 30/90/150-minute spans with equal date weights within each floor.',
        'All supported floors and periods remain. Floor 1 has no thermal arm; Floor 6 has no Nov-Dec targets.','',
        '| Floor | Period | '+' | '.join(methods)+' |',
        '|---|---|'+ '|'.join(['---:']*len(methods))+'|']
    for floor in range(1,8):
        for split,label in PERIODS:
            cells=[]
            for method in methods:
                rows=point[(point.floor==floor)&(point.split==split)&(point.scale=='device')&(point.method==method)]
                cells.append('n.a.' if rows.empty else f'{rows.iloc[0].mse:.3f} / {rows.iloc[0].mae:.3f}')
            lines.append(f'| {floor} | {label} | '+' | '.join(cells)+' |')
    lines+=['','## Common-date floor averages','',
        'Power comparison: Floors 1-5 and 7. Thermal comparison: Floors 2-5 and 7.',
        'Within each comparison and scale, dates are intersected across every included floor and method. Floors, dates and spans receive equal weights.',
        'These are average per-floor errors, not a summed-building forecast error. Confidence intervals resample shared calendar weeks, not independent floors.','',
        '| Cohort | Period | Scale | Method | MSE | MAE | Common dates |',
        '|---|---|---|---|---:|---:|---:|']
    if include_neural:
        lines.insert(4,'N-HiTS entries average separately evaluated errors from three fixed seeds, not ensemble forecasts.')
    c=common[(common.span_minutes==0)&common.split.isin([x[0] for x in PERIODS])&common.scale.isin(['device','floor_total'])]
    table=c.pivot(index=['cohort','split','scale','method','dates'],columns='metric',values='value').reset_index()
    for row in table.itertuples():
        lines.append(f'| {row.cohort} | {row.split} | {row.scale} | {row.method} | {row.mse:.3f} | {row.mae:.3f} | {row.dates} |')
    lines+=['','## All native results','',
        '| Floor | Split | Scale | Minutes | Method | MSE (kW²) | MAE (kW) | Energy AE (kWh) | Cases/groups | Dates |',
        '|---|---|---|---:|---|---:|---:|---:|---:|---:|']
    for row in native.sort_values(['floor','split','scale','span_minutes','method']).itertuples():
        lines.append(f'| {row.floor} | {row.split} | {row.scale} | {row.span_minutes} | {row.method} | {row.mse:.3f} | {row.mae:.3f} | {row.energy_ae_kwh:.3f} | {row.units:,} | {row.dates} |')
    (out/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    selected=contrasts[(contrasts.metric=='mse')&(contrasts.span_minutes==0)&contrasts.split.isin([x[0] for x in PERIODS])]
    fig,axes=plt.subplots(len(families),2,figsize=(10.5,3.2*len(families)+.3),sharex=True,sharey=True,layout='constrained')
    floors=[2,3,4,5,6,7];colors=['#287d8e','#b36935'];markers=['o','^']
    for row,family in enumerate(families):
        for col,(scale,title) in enumerate([('device','Device MSE'),('floor_total','Floor-total MSE')]):
            axis=axes[row,col]
            for y,floor in enumerate(floors):
                for period,(split,_) in enumerate(PERIODS):
                    s=selected[(selected.floor==floor)&(selected.family==family)&(selected.scale==scale)&(selected.split==split)]
                    if s.empty:continue
                    s=s.iloc[0];offset=-.13 if period==0 else .13
                    axis.plot([1+s.change_low/100,1+s.change_high/100],[y+offset]*2,color=colors[period],lw=1.6)
                    axis.scatter([1+s.change_pct/100],[y+offset],color=colors[period],marker=markers[period],s=28,zorder=3)
            axis.axvline(1,ls='--',color='#666666',lw=1)
            axis.set_xscale('log');axis.set_yticks(range(6),[f'Floor {f}' for f in floors])
            axis.set_title(f'{family}: {title}',fontsize=11)
            axis.spines[['top','right']].set_visible(False);axis.grid(axis='x',alpha=.15)
    axes[0,0].invert_yaxis()
    fig.supxlabel('MSE ratio: thermal inputs / power inputs (log scale; below 1 is lower)',fontsize=10)
    handles=[Line2D([],[],color=c,marker=m,ls='-',label=label) for c,m,(_,label) in zip(colors,markers,PERIODS)]
    fig.legend(handles=handles,loc='outside upper center',ncol=2,frameon=False)
    fig.savefig(out/'thermal_mse_ratios.png',dpi=165);plt.close(fig)
    result=dict(completed=True,seconds=time.monotonic()-started,native_rows=len(native),paired_rows=len(contrasts),
        common_interval_rows=len(common_contrasts),bootstrap_draws=1500,figure_visually_checked=False,new_fits=0,
        include_neural=include_neural,
        aggregation='Equal floor/date/span averages on common dates; shared calendar-week resampling; N-HiTS uses mean seed errors')
    dump(out/'receipt.json',result)
    print(json.dumps(result),flush=True)
    print(point[(point.scale=='device')&point.split.isin([x[0] for x in PERIODS])&
        (point.method.str.startswith('N-HiTS') if include_neural else True)].to_string(index=False))
    print(common_contrasts[(common_contrasts.metric=='mse')&(common_contrasts.span_minutes==0)&(common_contrasts.scale=='device')&common_contrasts.split.isin([x[0] for x in PERIODS])][['cohort','split','reference_method','candidate_method','reference','candidate','change_pct','change_low','change_high','dates']].to_string(index=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--with-neural',action='store_true')
    main(include_neural=parser.parse_args().with_neural)
