"""Average errors across matched seeds, never ensemble their predictions."""
import json
from pathlib import Path
import subprocess
import sys
import run_local
import numpy as np
import pandas as pd
from bpa_update_validation_common import ROOT,DEST,CONDITIONS
from analyze_bpa_update_validation import contrasts

METRICS=['mse','mae','reported_energy_ae_kwh']
DAILY_KEYS=['season','scale','command_type','target_end_minutes','method','local_date']


def mean_seed_errors(daily,seeds):
    if daily.duplicated(DAILY_KEYS+['seed']).any():raise ValueError('Repeated seed/date score')
    counts=daily.groupby(DAILY_KEYS).seed.nunique()
    if not (counts==len(seeds)).all():raise ValueError('Incomplete common seed/date support')
    if set(daily.seed.unique())!=set(seeds):raise ValueError('Unexpected seeds')
    return daily.groupby(DAILY_KEYS,as_index=False)[METRICS].mean()


def main():
    base=DEST.parent/'update_validation_typed';seeds=[1,2,3]
    for seed in seeds:
        family=f'nhits_seed{seed}'
        for condition in CONDITIONS:
            receipt=base/f'{family}_{condition}'/'receipt.json'
            if not receipt.exists() or not json.loads(receipt.read_text()).get('completed'):
                raise ValueError(f'Finish {family}/{condition} before aggregation')
        folder=base/f'analysis_{family}'
        # Reuse scored runs, score newly completed seeds using the same pipeline.
        if not (folder/'receipt.json').exists():
            command=[sys.executable,str(ROOT/'scripts/analyze_bpa_update_validation.py'),
                     '--family',family,'--encoding','typed']
            result=subprocess.run(command,cwd=ROOT,text=True,capture_output=True)
            if result.returncode:
                raise RuntimeError(result.stderr[-3000:])
            print(json.dumps({'scored_seed':seed}),flush=True)
    days=[];summaries=[];selections=[]
    for seed in seeds:
        folder=base/f'analysis_nhits_seed{seed}'
        days.append(pd.read_csv(folder/'daily_scores.csv.gz',dtype={'command_type':str}).assign(seed=seed))
        summaries.append(pd.read_csv(folder/'summary.csv',dtype={'command_type':str}).assign(seed=seed))
        selections.append(dict(seed=seed,**json.loads((folder/'validation_selection.json').read_text())))
    daily=pd.concat(days,ignore_index=True);individual=pd.concat(summaries,ignore_index=True)
    averaged=mean_seed_errors(daily,seeds);groupkeys=DAILY_KEYS[:-1]
    summary=averaged.groupby(groupkeys,as_index=False)[METRICS].mean()
    summary=summary.merge(individual[individual.seed==1][groupkeys+['units','dates','target_start_minutes','forecasted_bins']],
                          on=groupkeys,validate='one_to_one')
    dispersion=individual.groupby(groupkeys,as_index=False)[['mse','mae']].std().rename(columns={'mse':'mse_seed_std','mae':'mae_seed_std'})
    summary=summary.merge(dispersion,on=groupkeys,validate='one_to_one')
    summary['family']='nhits_mean3';summary['encoding']='typed';summary['seeds']=len(seeds)
    averaged['family']='nhits_mean3';averaged['encoding']='typed'
    paired=contrasts(averaged);paired['family']='nhits_mean3';paired['encoding']='typed'
    folder=base/'analysis_nhits_mean3';folder.mkdir(exist_ok=True)
    summary.to_csv(folder/'summary.csv',index=False);averaged.to_csv(folder/'daily_scores.csv.gz',index=False)
    paired.to_csv(folder/'paired_contrasts.csv',index=False)
    individual.to_csv(folder/'per_seed_scores_internal.csv',index=False)
    (folder/'validation_selections.json').write_text(json.dumps(selections,indent=2)+'\n')
    (folder/'receipt.json').write_text(json.dumps(dict(completed=True,seeds=seeds,
        aggregation='Mean of errors from separately fitted models; predictions are NOT ensembled.',
        uncertainty='Paired calendar-week intervals on seed-averaged daily errors; seed dispersion retained separately.',
        conditional_models=12,information_conditions=list(CONDITIONS)),indent=2)+'\n')
    mainrows=summary[(summary.scale=='cohort_mean')&(summary.command_type=='all')]
    print(mainrows[['season','target_end_minutes','method','mse','mae','mse_seed_std']].to_string(index=False),flush=True)


if __name__=='__main__':main()
