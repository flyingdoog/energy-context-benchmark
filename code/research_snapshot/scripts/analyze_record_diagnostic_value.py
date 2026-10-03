"""Three-arm synthesis and chronological diagnostic readout from saved losses."""
import json
from pathlib import Path
import time
import run_local
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from field_control.record_diagnostic import three_arm_effects, chronological_choice
from field_control.selection_utility import week_weights

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/record_diagnostic_value_v1'
KEYS=['source','site','period','scale','span_minutes','date']
METRICS=['mse','mae']
ARMS=['Base','Paired','Reassigned']


def norm(d):
    d=d.rename(columns={'local_date':'date'}).copy()
    d['date']=pd.to_datetime(d.date).dt.strftime('%Y-%m-%d')
    d['site']=d.site.replace({'caltech':'Caltech','office001':'Office'})
    d['scale']=d.scale.replace({'cohort_mean':'group_mean','cohort_total':'group_total',
                               'floor_mean':'group_mean','floor_total':'group_total'})
    return d[d.scale.isin(['group_mean','group_total'])].copy()


def aligned_check(a,b):
    """Same group-loss contract, allowing serialized decimal precision."""
    m=a[KEYS+METRICS].merge(b[KEYS+METRICS],on=KEYS,how='outer',suffixes=('_a','_b'),indicator=True,validate='one_to_one')
    if not m._merge.eq('both').all():raise ValueError('Original paired score date/scale support differs')
    for metric in METRICS:
        np.testing.assert_allclose(m[metric+'_a'],m[metric+'_b'],rtol=1e-6,atol=1e-7)
    return len(m)


def sources():
    expanded=norm(pd.read_json(ROOT/'results/operational_selection_utility_v1/expanded_learned_references/matched_daily.json',convert_dates=False))
    linkage=[]
    for seed in (2027,2028,2029):
        d=norm(pd.read_csv(ROOT/f'results/record_linkage_v1/experiments/seed_{seed}/daily.csv.gz'))
        linkage.append(d[KEYS+['method']+METRICS])
    link=pd.concat(linkage).groupby(KEYS+['method'],as_index=False)[METRICS].mean()
    full=expanded[expanded.source.isin(['BPA','ACN']) & expanded.candidate.eq('Tree/full')]
    checks=aligned_check(link[link.method.eq('Paired')],full)
    power=expanded[expanded.source.isin(['BPA','ACN']) & expanded.candidate.eq('Tree/power')].copy()
    power['method']='Base'
    tree=pd.concat([link,power[KEYS+['method']+METRICS]],ignore_index=True)
    cu=pd.read_json(ROOT/'results/cu_bems_linkage_v1/daily_scores.json',convert_dates=False)
    cu=cu.rename(columns={'split':'period'}).assign(source='CU-BEMS',site=lambda d:'Floor '+d.floor.astype(str))
    cu=norm(cu)
    cu['method']=cu.method.replace({'Power only':'Base','Correct link':'Paired'})
    for arm,candidate in [('Base','Tree/power'),('Paired','Tree/thermal')]:
        checks+=aligned_check(cu[cu.method.eq(arm)],expanded[expanded.source.eq('CU-BEMS') &
            expanded.site.ne('Floor 1') & expanded.candidate.eq(candidate)])
    tree=pd.concat([tree,cu[KEYS+['method']+METRICS]],ignore_index=True).assign(representation='Bottom-up Tree',base_label='Power only')
    moments=[]; sets=[]
    for site in ('bpa','caltech','office001'):
        d=norm(pd.read_json(ROOT/f'results/group_moment_reference_v1/{site}/daily.json.gz',convert_dates=False))
        d['method']=d.method.replace({'Marginal':'Base'})
        d.loc[d.method.str.startswith('Reassigned'),'method']='Reassigned'
        moments.append(d.groupby(KEYS+['method'],as_index=False)[METRICS].mean())
        for seed in (2027,2028,2029):
            d=norm(pd.read_json(ROOT/f'results/complete_set_v1/experiments/{site}_seed{seed}/daily.json.gz',convert_dates=False))
            sets.append(d[KEYS+['method']+METRICS])
    moments=pd.concat(moments).assign(representation='Group moments',base_label='Marginal features')
    sets=pd.concat(sets).groupby(KEYS+['method'],as_index=False)[METRICS].mean().assign(representation='Deep Sets')
    data=pd.concat([tree,moments],ignore_index=True)
    if data.duplicated(KEYS+['method','representation']).any():raise ValueError('Repeated arm/date cell')
    return data,sets,expanded,checks


def intervals(data):
    cache={}; out=[]
    avg=data.groupby(['source','site','period','scale','date','method','representation','base_label'],as_index=False)[METRICS].mean()
    avg['span_minutes']=0
    all_rows=pd.concat([data,avg],ignore_index=True)
    group=['source','site','period','scale','span_minutes','representation','base_label']
    for key,part in all_rows.groupby(group):
        for metric in METRICS:
            p=part.pivot(index='date',columns='method',values=metric).reindex(columns=ARMS).sort_index()
            if p.isna().any().any():raise ValueError('Three-arm date support differs')
            dates=tuple(p.index)
            if dates not in cache:cache[dates]=week_weights(dates)
            w,nweeks=cache[dates]
            out.append(dict(zip(group,key),metric=metric,dates=len(p),weeks=nweeks,
                **three_arm_effects(p.Base,p.Paired,p.Reassigned,w)))
    return pd.DataFrame(out)


def transitions(effects):
    e=effects[effects.span_minutes.eq(0)]
    key=['source','site','scale','representation','base_label','metric']
    val=e[e.period.eq('validation')].drop(columns='period')
    later=e[~e.period.eq('validation')]
    return later.merge(val,on=key,suffixes=('_later','_validation'),validate='many_to_one')


def selectors(data,sets,expanded):
    cand=expanded.copy()
    eligible=data[data.representation.eq('Bottom-up Tree')][['source','site']].drop_duplicates()
    cand=cand.merge(eligible,on=['source','site'],validate='many_to_one')
    moment=data[data.representation.eq('Group moments') & data.method.ne('Reassigned')].copy()
    moment['candidate']=moment.method.map({'Base':'Group moments/marginal','Paired':'Group moments/paired'})
    ds=sets[sets.method.eq('Paired')].copy();ds['candidate']='Deep Sets/paired'
    cand=pd.concat([cand[KEYS+['candidate']+METRICS],moment[KEYS+['candidate']+METRICS],ds[KEYS+['candidate']+METRICS]],ignore_index=True)
    choices=[];selected=[]
    for (source,site,scale),d in cand.groupby(['source','site','scale']):
        for metric in METRICS:
            winner,vs=chronological_choice(d,metric)
            choices.append(dict(source=source,site=site,scale=scale,metric=metric,selected=winner,validation_scores=vs))
            for period,g in d[~d.period.eq('validation')].groupby('period'):
                p=g.pivot(index=['date','span_minutes'],columns='candidate',values=metric)
                if p.isna().any().any():raise ValueError('Later candidate support differs')
                default='Tree/thermal' if source=='CU-BEMS' else 'Tree/full'
                selected.append(dict(source=source,site=site,scale=scale,metric=metric,period=period,selected=winner,
                    selected_error=float(p[winner].mean()),paired_tree_error=float(p[default].mean()),
                    power_tree_error=float(p['Tree/power'].mean())))
    return choices,pd.DataFrame(selected)


def write_outputs(data,effects,change,choices,selected,sets):
    for name,frame in [('matched_daily',data),('effects',effects),('validation_to_later',change),('selected_later',selected),('deep_sets_daily',sets)]:
        frame.to_json(OUT/f'{name}.json',orient='records',indent=2 if name!='matched_daily' else None)
    (OUT/'ordinary_validation_choices.json').write_text(json.dumps(choices,indent=2)+'\n',encoding='utf8')
    lines=['# What does record association add?', '',
        'All errors use saved fits, matched group totals, and equal date/native-span weights.',
        'MSE / MAE units are kW squared / kW. Mapping errors are averaged, not forecast-ensembled.',
        'All three native spans (30/90/150 minutes) are shown in the JSON results.', '',
        '## Bottom-up Tree: three-arm comparison', '',
        '| Source / site | Period | Power only MSE / MAE | Paired MSE / MAE | Reassigned MSE / MAE |',
        '|---|---|---:|---:|---:|']
    e=effects[effects.scale.eq('group_total') & effects.span_minutes.eq(0)]
    for key,d in e[e.representation.eq('Bottom-up Tree')].groupby(['source','site','period']):
        m=d.set_index('metric');cells=[f'{m.loc["mse",c]:.3f} / {m.loc["mae",c]:.3f}' for c in ['base_error','paired_error','reassigned_error']]
        lines.append(f'| {key[0]} / {key[1]} | {key[2]} | '+' | '.join(cells)+' |')
    lines+=['','## Validation signals and later consequences','',
        'Positive values mean lower error. All percentages use the same power-only (or marginal-arm)',
        'denominator: total = (Base - Paired)/Base; unlinked = (Base - Reassigned)/Base;',
        'association = (Reassigned - Paired)/Base; total = unlinked + association.',
        'These are model-conditional error contrasts, not identified causal information quantities.', '',
        '| Representation / site | Later period | Total gain: validation → later (%) | Association gain: validation → later (%) |',
        '|---|---|---:|---:|']
    c=change[change.scale.eq('group_total') & change.metric.eq('mse')]
    for r in c.itertuples():
        lines.append(f'| {r.representation} / {r.site} | {r.period} | {r.total_gain_pct_validation:+.2f} → {r.total_gain_pct_later:+.2f} | {r.association_gain_pct_validation:+.2f} → {r.association_gain_pct_later:+.2f} |')
    lines+=['','## Ordinary validation selection with strong saved references','',
        'No proposed diagnostic selector is claimed. This is the comparator that any such rule must exceed.',
        'Each task/metric selects once on validation, including the existing stronger references and new fixed',
        'group representations where available. Caltech declarations have a larger information budget.', '',
        '| Site | Later period | Metric | Validation-selected model | Selected error | Full-input Tree error |',
        '|---|---|---|---|---:|---:|']
    for r in selected[selected.scale.eq('group_total')].itertuples():
        lines.append(f'| {r.site} | {r.period} | {r.metric.upper()} | {r.selected} | {r.selected_error:.3f} | {r.paired_tree_error:.3f} |')
    lines+=['','## Scope and interpretation','',
        'BPA is one collection, ACN has two sites, and all CU floors are one building. Floor 1 has no',
        'thermal record arm; Floor 6 has no second later period. EPFL has no synchronized fixed-population',
        'task in the present protocol, so it is not a missing three-arm group result.',
        'Paired calendar-week intervals are saved for both metrics and all spans. They condition on',
        'fitted models and do not create independent datasets from floors, mapping repeats or dates.',
        'Later outcomes were previously inspected; this is retrospective chronological analysis.',
        'Deep Sets has no no-record arm; group moments use a marginal-information arm, not power-only.',
        'No algorithm, maintenance-cost saving, causal effect or independent confirmation follows merely',
        'from the additive contrast identity. No source files, fits or original scores were altered.','']
    (OUT/'comparison.md').write_text('\n'.join(lines),encoding='utf8')
    plot=e[(e.representation=='Bottom-up Tree') & e.metric.eq('mse') & ~e.period.eq('validation')].copy()
    fig,axes=plt.subplots(1,2,figsize=(12,6.8),sharey=True)
    labels=[f'{"BPA" if r.source=="BPA" else r.site} · {dict(later_1="Sep–Oct",later_2="Nov–Dec").get(r.period,r.period)}' for r in plot.itertuples()]
    y=np.arange(len(plot))
    for ax,column,title,color in zip(axes,['total_gain','association_gain'],['Added records: total gain','Correct association: additional contrast'],['#247B91','#B96935']):
        v=plot[column+'_pct'].to_numpy();lo=plot[column+'_low'].to_numpy();hi=plot[column+'_high'].to_numpy()
        ax.hlines(y,lo,hi,color=color,linewidth=1.3);ax.scatter(v,y,color=color,s=22,zorder=3)
        ax.axvline(0,color='gray',ls='--',lw=.8);ax.set_title(title,fontsize=11)
        ax.set_xlabel('MSE reduction / power-only MSE (%)');ax.set_yticks(y,labels)
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.15)
    axes[0].invert_yaxis();fig.tight_layout()
    fig.savefig(OUT/'content_and_association.png',dpi=170);plt.close(fig)
    return lines


def main():
    started=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    data,sets,expanded,checks=sources()
    effects=intervals(data);change=transitions(effects)
    choices,selected=selectors(data,sets,expanded)
    lines=write_outputs(data,effects,change,choices,selected,sets)
    receipt=dict(completed=True,seconds=time.perf_counter()-started,matched_native_daily_rows=len(data),
        recovered_original_group_rows=checks,effect_rows=len(effects),transition_rows=len(change),
        validation_choices=len(choices),new_fits=0,new_predictions=0,gpu_used=False,
        validation_missing=False,diagnostic_selection_benefit_demonstrated=False,figure_checked=False)
    (OUT/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
    print(json.dumps(receipt),flush=True)
    print('\n'.join(lines),flush=True)


if __name__=='__main__':main()
