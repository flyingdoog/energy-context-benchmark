"""Summarize every prescribed HVAC period/span and paired calendar-week changes."""
import json
from pathlib import Path
import time
import run_local
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_cu_bems_reference import OUT as BASE, dump, frame_json
from analyze_count_distributions import calendar_weights, paired_statistics

OUT=BASE/'analysis'
METRICS=['mse','mae','energy_ae_kwh']
METHODS=['Persistence','Ridge: power','Ridge: thermal','Tree: power','Tree: thermal']
PERIODS=[('later_1','September–October'),('later_2','November–December')]
SCALES=[('device','Device'),('floor_total','16-AC floor total'),('zone2_total','14-AC zone total')]


def main():
    started=time.monotonic()
    if not json.loads((BASE/'scoring.json').read_text())['completed']:
        raise ValueError('Complete prescribed fits and scoring first')
    OUT.mkdir(exist_ok=True)
    summary=pd.read_json(BASE/'summary.json')
    daily=pd.read_json(BASE/'daily_scores.json',convert_dates=False)
    daily['date']=pd.to_datetime(daily.date).dt.strftime('%Y-%m-%d')
    average=daily.groupby(['split','scale','method','date'],as_index=False)[['mse','mae']].mean()
    average['span_minutes']=0
    all_scores=pd.concat([daily,average],ignore_index=True)
    intervals=[];weight_cache={}
    for (split,scale,span),block in all_scores.groupby(['split','scale','span_minutes']):
        for metric in (['mse','mae'] if span==0 else METRICS):
            pivot=block.pivot(index='date',columns='method',values=metric).sort_index()
            if pivot.isna().any().any() or set(pivot.columns)!=set(METHODS):
                raise ValueError('Methods must share dates and scoring support')
            dates=tuple(pivot.index)
            if dates not in weight_cache:weight_cache[dates]=calendar_weights(dates)
            weights,weeks=weight_cache[dates]
            for family in ('Ridge','Tree'):
                intervals.append(dict(split=split,scale=scale,span_minutes=int(span),metric=metric,family=family,
                    dates=len(dates),weeks=weeks,
                    **paired_statistics(pivot[f'{family}: power'],pivot[f'{family}: thermal'],weights)))
    contrasts=pd.DataFrame(intervals)
    frame_json(OUT/'paired_intervals.json',contrasts)
    table=average.groupby(['split','scale','method'],as_index=False)[['mse','mae']].mean()
    frame_json(OUT/'span_average.json',table)
    lines=['# CU-BEMS Floor 2 default-reference results','',
        'MSE (kW²) / MAE (kW), averaged over the three cumulative prediction spans with equal date weights.',
        'Thermal adds the correctly linked past temperature/humidity and their observed masks. All methods share cases.', '',
        '| Later period | Reference | Device MSE / MAE | Floor-total MSE / MAE | Zone-2-total MSE / MAE |',
        '|---|---|---:|---:|---:|']
    for split,label in PERIODS:
        for method in METHODS:
            cells=[]
            for scale,_ in SCALES:
                row=table[(table.split==split)&(table.scale==scale)&(table.method==method)].iloc[0]
                cells.append(f'{row.mse:.3f} / {row.mae:.3f}')
            lines.append(f'| {label} | {method} | '+ ' | '.join(cells)+' |')
    lines += ['', 'These are one fixed default-tree run and deterministic Ridge, not multi-seed neural estimates.',
        'The two periods belong to the same floor/building. Nested floor/zone groups are not independent datasets.',
        'The paired-interval figure adds calendar-week uncertainty to the point-score table. Intervals condition on fitted models.',
        'No validation/test hyperparameter selection or performance-based device removal occurred.', '',
        '## All native spans, including validation','',
        '| Period | Scale | Minutes | Method | MSE (kW²) | MAE (kW) | Energy AE (kWh) | Cases/groups | Dates |',
        '|---|---|---:|---|---:|---:|---:|---:|---:|']
    for row in summary.sort_values(['split','scale','span_minutes','method']).itertuples():
        lines.append(f'| {row.split} | {row.scale} | {row.span_minutes} | {row.method} | {row.mse:.3f} | {row.mae:.3f} | {row.energy_ae_kwh:.3f} | {row.units:,} | {row.dates} |')
    (OUT/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    selected=contrasts[(contrasts.metric=='mse')&(contrasts.span_minutes==0)&contrasts.split.isin([p[0] for p in PERIODS])]
    fig,axes=plt.subplots(1,3,figsize=(11.5,4),sharex=True,sharey=True,layout='constrained')
    order=[('later_1','Ridge'),('later_1','Tree'),('later_2','Ridge'),('later_2','Tree')]
    for axis,(scale,title) in zip(axes,SCALES):
        for y,(split,family) in enumerate(order):
            r=selected[(selected.scale==scale)&(selected.split==split)&(selected.family==family)].iloc[0]
            color='#277f89' if family=='Tree' else '#b86c35'
            axis.plot([r.change_low,r.change_high],[y,y],color=color,lw=2)
            axis.scatter([r.change_pct],[y],color=color,s=36,zorder=3)
        axis.axvline(0,color='#777777',linestyle='--',lw=1)
        axis.set_title(title,fontsize=11)
        axis.grid(axis='x',alpha=.15)
        axis.spines[['top','right']].set_visible(False)
        axis.set_yticks(range(4),['Sep–Oct / Ridge','Sep–Oct / Tree','Nov–Dec / Ridge','Nov–Dec / Tree'])
    axes[0].invert_yaxis()
    fig.supxlabel('Thermal-context change in MSE (%)   ·   lower is better',fontsize=10)
    fig.suptitle('CU-BEMS: date-paired changes with 95% calendar-week intervals',fontsize=12)
    fig.savefig(OUT/'paired_mse.png',dpi=170)
    plt.close(fig)
    result=dict(completed=True,seconds=time.monotonic()-started,paired_rows=len(contrasts),
        native_rows=len(summary),bootstrap_draws=1500,figure_visually_checked=False,new_fits=0)
    dump(OUT/'receipt.json',result)
    print(json.dumps(result),flush=True)
    print(selected[selected.scale.isin([s[0] for s in SCALES])][['split','scale','family','reference','candidate','change_pct','change_low','change_high','weeks']].to_string(index=False))


if __name__=='__main__':main()
