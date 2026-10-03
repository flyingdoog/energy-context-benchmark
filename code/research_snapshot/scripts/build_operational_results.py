"""One source of paper tables for both device families and three model classes."""
import json
from pathlib import Path
import run_local
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/operational_reference_comparison_v1'
CONDITIONS=('power','power_commands','power_state','full')
MODELS=('Ridge','LightGBM','N-HiTS')
CONTEXTS=[('BPA','Field study','spring'),('BPA','Field study','summer'),('ACN','Caltech','autumn'),('ACN','Caltech','winter'),('ACN','Office','autumn'),('ACN','Office','winter')]


def load_results():
    tables=[]
    bpa=ROOT/'results/benchmark_bpa_sequences_v1/telemetry_value_v1'
    acn=ROOT/'results/acn_fleet_months_v1'
    for model,path in [('Ridge',bpa/'ridge_reference_float64/summary.csv'),('LightGBM',bpa/'summary.csv'),('N-HiTS',bpa/'analysis_nhits_mean3/summary.csv')]:
        data=pd.read_csv(path,dtype={'command_type':str})
        data=data[(data.command_type=='all')&data.season.isin(['spring','summer'])].copy()
        data=data.rename(columns={'season':'period','method':'condition','reported_energy_ae_kwh':'energy_ae_kwh'})
        data['span_minutes']=data.target_end_minutes-30;data['source']='BPA';data['site']='Field study';data['model']=model
        tables.append(data)
    for model,path in [('Ridge',acn/'ridge_references_float64_v1/summary.csv'),('LightGBM',acn/'summary.csv'),('N-HiTS',acn/'neural_references_v1/analysis_nhits_mean3/summary.csv')]:
        data=pd.read_csv(path);data=data[data.split.isin(['autumn','winter'])].copy()
        if 'condition' not in data:data['condition']=data.method
        data=data[data.condition.isin(CONDITIONS)].rename(columns={'split':'period'})
        data['site']=data.site.map({'caltech':'Caltech','office001':'Office'});data['source']='ACN';data['model']=model
        tables.append(data)
    cols=['source','site','period','model','condition','scale','span_minutes','mse','mae','energy_ae_kwh','dates','units']
    result=pd.concat([data[cols] for data in tables],ignore_index=True)
    keys=cols[:7]
    if result.duplicated(keys).any() or not np.isfinite(result[['mse','mae','energy_ae_kwh']]).all().all():raise ValueError('Duplicate or invalid reference cells')
    expected=6*3*4*3*3
    if len(result)!=expected:raise ValueError(f'Incomplete reference matrix:{len(result)} vs {expected}')
    return result


def main():
    OUT.mkdir(exist_ok=True);data=load_results();data.to_csv(OUT/'complete_reference_scores.csv',index=False)
    group=data[data.scale=='cohort_mean'];keys=['source','site','period','model','condition']
    macro=group.groupby(keys)[['mse','mae','energy_ae_kwh']].mean().reset_index();macro.to_csv(OUT/'span_average_scores.csv',index=False)
    names=['BPA spring','BPA summer','Caltech autumn','Caltech winter','Office autumn','Office winter']
    lines=['# Operational-device forecasting: main model comparison','',
        'Cohort-mean MSE / MAE, averaged equally across the 30, 90 and 150 minute forecast spans after date-balanced scoring. History includes the corresponding calendar information. All inputs add observed control and device state. N-HiTS entries average three independently trained model errors; no ensembling or baseline hyperparameter search. The six columns are settings/periods from two source collections.', '',
        '| Model | Inputs | '+' | '.join(names)+' |','|---|---|'+'|'.join(['---:']*6)+'|']
    for model in MODELS:
        for condition,label in [('power','History'),('full','All inputs')]:
            cells=[]
            for source,site,period in CONTEXTS:
                row=macro[(macro.source==source)&(macro.site==site)&(macro.period==period)&(macro.model==model)&(macro.condition==condition)]
                if len(row)!=1:raise ValueError('Ambiguous main-table cell')
                row=row.iloc[0];cells.append(f'{row.mse:.3f} / {row.mae:.3f}')
            lines.append('| '+' | '.join([model,label,*cells])+' |')
    (ROOT/'paper/generated/operational_main_results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    full=['# Full operational-device results','',
        'Cohort-mean MSE / MAE for every information condition and forecast span. Model recipes, paired targets and seed averaging match the main comparison. Device, cohort-total and integrated-energy metrics remain in the complete result file.', '',
        '| Source / setting / period | Span (min) | Model | History | + control | + state | All inputs |',
        '|---|---:|---|---:|---:|---:|---:|']
    for source,site,period in CONTEXTS:
        for span in [30,90,150]:
            for model in MODELS:
                part=group[(group.source==source)&(group.site==site)&(group.period==period)&(group.span_minutes==span)&(group.model==model)].set_index('condition')
                cells=[f'{part.loc[c,"mse"]:.3f} / {part.loc[c,"mae"]:.3f}' for c in CONDITIONS]
                full.append('| '+' | '.join([source+' / '+site+' / '+period,str(span),model,*cells])+' |')
    (ROOT/'paper/generated/operational_full_results.md').write_text('\n'.join(full)+'\n',encoding='utf-8')
    effect_keys=['source','site','period','model','scale','span_minutes']
    effects=data[data.condition=='power'].merge(data[data.condition=='full'],on=effect_keys,suffixes=('_history','_full'))
    for metric in ['mse','mae']:effects[metric+'_reduction_pct']=100*(1-effects[metric+'_full']/effects[metric+'_history'])
    effects.to_csv(OUT/'information_effects.csv',index=False)
    (OUT/'receipt.json').write_text(json.dumps(dict(completed=True,source_collections=2,source_settings=3,evaluation_contexts=6,
        model_classes=3,conditions=4,forecast_spans=[30,90,150],score_scales=3,complete_cells=len(data),
        main_summary='Equal-weight mean of three per-span,date-balanced errors. No raw-score macro average across different source contexts.'),indent=2)+'\n')
    print('\n'.join(lines),flush=True)


if __name__=='__main__':main()
