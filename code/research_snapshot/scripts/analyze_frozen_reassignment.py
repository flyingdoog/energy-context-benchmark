"""Compare fixed-model permutation sensitivity with retrained reassignment."""
import json
from pathlib import Path
import time
import run_local
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_record_diagnostic_value import norm,KEYS,METRICS
from field_control.selection_utility import week_weights

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/frozen_reassignment_v1'
OUT=BASE/'analysis'


def main():
    started=time.perf_counter()
    receipt=json.loads((BASE/'receipt.json').read_text())
    if not receipt.get('completed') or len(receipt['sites'])!=3:raise ValueError('Full three-site batch required')
    OUT.mkdir(parents=True,exist_ok=True)
    frozen=[]
    for site in ['bpa','caltech','office001']:
        d=norm(pd.read_json(BASE/site/'daily.json.gz',convert_dates=False))
        frozen.append(d.groupby(KEYS+['method'],as_index=False)[METRICS].mean())
    d=pd.read_json(ROOT/'results/record_diagnostic_value_v1/matched_daily.json',convert_dates=False)
    d=d[d.representation.eq('Bottom-up Tree') & d.source.isin(['BPA','ACN'])]
    data=pd.concat([d[KEYS+['method']+METRICS],*frozen],ignore_index=True)
    avg=data.groupby(['source','site','period','scale','date','method'],as_index=False)[METRICS].mean()
    avg['span_minutes']=0;data=pd.concat([data,avg],ignore_index=True)
    rows=[];cache={}
    for key,part in data.groupby(['source','site','period','scale','span_minutes']):
        for metric in METRICS:
            p=part.pivot(index='date',columns='method',values=metric).sort_index()
            methods=['Base','Paired','Frozen reassigned','Reassigned']
            if set(p.columns)!=set(methods) or p.isna().any().any():raise ValueError('Four-arm date support differs')
            p=p[methods];dates=tuple(p.index)
            if dates not in cache:cache[dates]=week_weights(dates)
            w,nweeks=cache[dates];mean=p.mean().to_numpy();b=w@p.to_numpy()
            record=dict(zip(['source','site','period','scale','span_minutes'],key),metric=metric,
                dates=len(p),weeks=nweeks,power_error=float(mean[0]),paired_error=float(mean[1]),
                frozen_error=float(mean[2]),retrained_error=float(mean[3]))
            for name,i,j in [('frozen_penalty',2,1),('retrained_penalty',3,1),('relearning_recovery',2,3)]:
                samples=100*(b[:,i]-b[:,j])/b[:,1];lo,hi=np.quantile(samples,[.025,.975])
                record.update({name+'_pct':float(100*(mean[i]-mean[j])/mean[1]),name+'_low':float(lo),name+'_high':float(hi)})
            rows.append(record)
    effects=pd.DataFrame(rows)
    effects.to_json(OUT/'effects.json',orient='records',indent=2)
    data.to_json(OUT/'matched_daily.json',orient='records')
    e=effects[effects.scale.eq('group_total') & effects.span_minutes.eq(0)]
    lines=['# Fixed-model versus retrained reassignment','',
        'MSE / MAE, original group-total units, averaged over all native spans and dates.',
        'Both permuted arms use the same three cached mappings. Fixed permutation keeps the',
        'original Tree; retrained reassignment uses the existing fixed-recipe retrained models.',
        'The mapping moves auxiliary values and masks together within the original query population.','',
        '| Source / site | Period | Power only | Paired | Fixed-model reassignment | Retrained reassignment |',
        '|---|---|---:|---:|---:|---:|']
    for key,d in e.groupby(['source','site','period']):
        p=d.set_index('metric');cols=['power_error','paired_error','frozen_error','retrained_error']
        cells=[f'{p.loc["mse",c]:.3f} / {p.loc["mae",c]:.3f}' for c in cols]
        lines.append(f'| {key[0]} / {key[1]} | {key[2]} | '+' | '.join(cells)+' |')
    lines+=['','## Group-MSE penalty relative to correctly paired Tree','',
        'All percentages share the paired-error denominator. Relearning recovery = fixed penalty - retrained penalty.',
        'Intervals: 1,500 paired calendar-week draws, conditional on the fitted models.','',
        '| Site | Period | Fixed-model penalty (%) [95% CI] | Retrained penalty (%) [95% CI] | Relearning recovery (percentage points) |',
        '|---|---|---:|---:|---:|']
    for r in e[e.metric.eq('mse')].itertuples():
        lines.append(f'| {r.site} | {r.period} | {r.frozen_penalty_pct:+.2f} [{r.frozen_penalty_low:+.2f}, {r.frozen_penalty_high:+.2f}] | {r.retrained_penalty_pct:+.2f} [{r.retrained_penalty_low:+.2f}, {r.retrained_penalty_high:+.2f}] | {r.relearning_recovery_pct:+.2f} |')
    lines+=['','Fixed-model error sensitivity and recoverability after retraining are different quantities.',
        'This is a model-conditional benchmark control, not a new permutation-importance estimator,',
        'not conditional sampling given every protected variable, and not an identified causal effect.',
        'Validation and previously inspected later periods are retained. CU and EPFL are not newly',
        'evaluated here. Stronger ordinary validation references remain in the diagnostic-value readout.','']
    (OUT/'comparison.md').write_text('\n'.join(lines),encoding='utf8')
    fig,axes=plt.subplots(1,3,figsize=(11.7,4.3))
    for ax,(source,site) in zip(axes,[('BPA','Field study'),('ACN','Caltech'),('ACN','Office')]):
        order=['validation','spring','summer'] if source=='BPA' else ['validation','autumn','winter']
        block=e[e.source.eq(source)&e.site.eq(site)&e.metric.eq('mse')].set_index('period').loc[order]
        x=np.arange(3);width=.34
        for delta,name,label,color in [(-width/2,'frozen_penalty','Fixed model','#B16B38'),(width/2,'retrained_penalty','Retrained','#2D7894')]:
            y=block[name+'_pct'].to_numpy();lo=block[name+'_low'].to_numpy();hi=block[name+'_high'].to_numpy()
            ax.bar(x+delta,y,width,color=color,label=label)
            ax.vlines(x+delta,lo,hi,color='#444444',lw=1)
        ax.axhline(0,color='gray',lw=.8);ax.set_xticks(x,[s.capitalize() for s in order])
        ax.set_title('BPA' if source=='BPA' else site);ax.set_ylabel('MSE penalty vs paired Tree (%)')
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.12)
    axes[0].legend(frameon=False,fontsize=9);fig.tight_layout()
    fig.savefig(OUT/'fixed_vs_retrained.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(8.8,4.7))
    entries=[('BPA','Field study','spring','BPA spring'),('ACN','Caltech','autumn','Caltech autumn'),
             ('ACN','Office','autumn','Office autumn'),('BPA','Field study','summer','BPA summer'),
             ('ACN','Caltech','winter','Caltech winter'),('ACN','Office','winter','Office winter')]
    colors=['#84939C','#286F8C','#C68952','#76AFA1']
    labels=['Power only','Paired records','Fixed-model reassignment','Retrained reassignment']
    for ax,(source,site,period,title) in zip(axes.flat,entries):
        row=e[e.source.eq(source)&e.site.eq(site)&e.period.eq(period)&e.metric.eq('mse')].iloc[0]
        vals=[row.power_error,row.paired_error,row.frozen_error,row.retrained_error]
        ax.bar(range(4),vals,color=colors,width=.7)
        for i,v in enumerate(vals):ax.text(i,v,f'{v:.2f}',ha='center',va='bottom',fontsize=11.5)
        ax.set_title(title,fontsize=12);ax.set_xticks([]);ax.set_ylabel('MSE (kW²)',fontsize=11)
        ax.tick_params(axis='y',labelsize=10)
        ax.set_ylim(0,max(vals)*1.18);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.12)
    handles=[plt.Rectangle((0,0),1,1,color=c) for c in colors]
    fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False,fontsize=11.5)
    fig.tight_layout(rect=[0,.13,1,1]);fig.savefig(OUT/'four_arm_group_mse.png',dpi=220);plt.close(fig)
    result=dict(completed=True,seconds=time.perf_counter()-started,effect_rows=len(effects),
        new_fits=0,new_predictions=0,full_three_site_result=True,figure_checked=False)
    (OUT/'receipt.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result),flush=True);print('\n'.join(lines),flush=True)


if __name__=='__main__':main()
