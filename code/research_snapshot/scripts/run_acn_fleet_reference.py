"""Fixed default tree references at individual and shared-clock cohort scales."""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import run_local
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from prepare_acn_fleet_pilot import DEST

CONDITIONS = ('power', 'power_commands', 'power_state', 'full')


def restrict(x, condition, groups):
    selected = x.copy()
    if condition in ('power', 'power_state'):
        a, b = groups['control']; selected[:, a:b] = 0
    if condition in ('power', 'power_commands'):
        a, b = groups['state']; selected[:, a:b] = 0
    return selected


def complete_cohort_mask(meta, population):
    keys=['site','split','query_utc']
    # Cohort membership comes from observed history, before target availability.
    expected=population.groupby(keys).size()
    present=meta.groupby(keys).size()
    good=present.index[present==expected.reindex(present.index)]
    return pd.MultiIndex.from_frame(meta[keys]).isin(good)


def score(meta, residual, method, population=None):
    rows, daily_rows = [], []
    cols = [f'e{j}' for j in range(10)]
    frame = meta[['site', 'split', 'local_date', 'query_utc', 'station_id']].copy()
    frame[cols] = residual
    cohort_available = np.ones(len(meta),dtype=bool) if population is None else complete_cohort_mask(meta,population)
    frame['cohort_available']=cohort_available
    for dims, period in frame.groupby(['site', 'split'], sort=False):
        site, split = dims
        # Do not double count an EVSE when erroneous overlapping sessions occur.
        duplicates = period.duplicated(['query_utc', 'station_id'], keep=False)
        bad_queries = set(period.loc[duplicates, 'query_utc'])
        clean = period[(~period.query_utc.isin(bad_queries)) & period.cohort_available]
        grouped = clean.groupby(['local_date', 'query_utc'])
        mean = grouped[cols].mean()
        size = grouped.size()
        mean = mean[size >= 2]
        total = mean.mul(size[size >= 2], axis=0)
        for scale, table in [('device', period), ('cohort_mean', mean.reset_index()), ('cohort_total', total.reset_index())]:
            if table.empty:
                continue
            for bins in (2, 6, 10):
                errors = table[cols[:bins]].to_numpy()
                metrics = pd.DataFrame({'local_date': table.local_date.to_numpy(),
                                        'mse': np.mean(errors**2, axis=1),
                                        'mae': np.mean(abs(errors), axis=1),
                                        'energy_ae_kwh': abs(errors.sum(axis=1)*.25)})
                day = metrics.groupby('local_date').mean()
                info = dict(site=site, split=split, method=method, scale=scale, span_minutes=bins*15)
                rows.append(dict(**info, units=len(table), dates=len(day), **day.mean().to_dict()))
                daily_rows.append(day.reset_index().assign(**info))
    return pd.DataFrame(rows), pd.concat(daily_rows, ignore_index=True)


def main():
    global DEST
    parser=argparse.ArgumentParser();parser.add_argument('--rescore-only',action='store_true');parser.add_argument('--expanded',action='store_true');args=parser.parse_args()
    if args.expanded:DEST=DEST.parent/'acn_fleet_months_v1'
    receipt = json.loads((DEST / 'packet_receipt.json').read_text())
    meta = pd.read_csv(DEST / 'cases.csv.gz')
    population=pd.read_csv(DEST/'query_population.csv.gz')
    with np.load(DEST / 'packet.npz') as packet:
        x, y = packet['x'], packet['y']
    start = time.monotonic()
    models = DEST / 'references'; models.mkdir(exist_ok=True)
    if args.rescore_only:
        tables,days=[],[]
        for method in ['persistence',*CONDITIONS]:
            with np.load(models/(method+'.npz')) as stored:
                ix=pd.Index(stored['case_id'].astype(str)).get_indexer(meta.case_id)
                if (ix<0).any():raise ValueError('Saved predictions do not cover the current cases')
                prediction=stored['prediction_kw'][ix]
            table,daily=score(meta,prediction-y,method,population)
            tables.append(table);days.append(daily)
        pd.concat(tables,ignore_index=True).to_csv(DEST/'summary.csv',index=False)
        pd.concat(days,ignore_index=True).to_csv(DEST/'daily_scores.csv.gz',index=False)
        (DEST/'cohort_rescore_receipt.json').write_text(json.dumps(dict(completed=True,new_fits=0,
            query_known_population=len(population),complete_device_cases=len(meta),
            contract='Whole-cohort label availability; never regroup by removing individual future-missing members.'))+'\n')
        print('Rescored existing forecasts against query-known complete cohorts.',flush=True)
        return
    outputs, days, fits = [], [], []
    predictions = {'persistence': np.repeat(meta.persistence_kw.to_numpy()[:, None], 10, axis=1).astype(np.float32)}
    for condition in CONDITIONS:
        predictions[condition] = np.full_like(y, np.nan)

    def fit(site, condition):
        selection = np.flatnonzero(meta.site.to_numpy() == site)
        local = meta.iloc[selection]
        train = local.split.to_numpy() == 'train'
        if train.sum() < 30:
            raise ValueError(f'Insufficient pilot training support for {site}')
        features = restrict(x[selection], condition, receipt['feature_groups'])
        prediction = np.empty((len(selection), 10), dtype=np.float32)
        folder = models / site / condition; folder.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        for lead in range(10):
            model = LGBMRegressor(n_jobs=2, random_state=2027, verbosity=-1)
            model.fit(features[train], y[selection[train], lead])
            prediction[:, lead] = np.maximum(0, model.predict(features))
            model.booster_.save_model(str(folder / f'lead_{lead:02d}.txt'))
            (folder / 'progress.json').write_text(json.dumps(dict(leads=lead+1, seconds=time.monotonic()-started)))
        return dict(site=site, condition=condition, training_cases=int(train.sum()), seconds=time.monotonic()-started), selection, prediction

    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(fit, site, condition) for site in sorted(meta.site.unique()) for condition in CONDITIONS]
        for future in as_completed(jobs):
            details, selection, prediction = future.result()
            predictions[details['condition']][selection] = prediction
            fits.append(details); print(json.dumps(details), flush=True)
    for method, prediction in predictions.items():
        if not np.isfinite(prediction).all():
            raise ValueError('Missing predictions')
        np.savez_compressed(models / (method+'.npz'), case_id=meta.case_id.to_numpy(dtype=str), prediction_kw=prediction)
        table, daily = score(meta, prediction-y, method,population)
        outputs.append(table); days.append(daily)
    pd.concat(outputs, ignore_index=True).to_csv(DEST / 'summary.csv', index=False)
    pd.concat(days, ignore_index=True).to_csv(DEST / 'daily_scores.csv.gz', index=False)
    final = dict(completed=True, seconds=time.monotonic()-start, fit_count=10*len(fits), fits=fits,
                 recipe='Default LightGBM100trees/31leaves/MSE/seed2027/two threads. No HPO or validation selection.',
                 statistical_status='Fixed-calendar feasibility pilot, deterministic tree; not a multi-seed neural benchmark or a final release.')
    (DEST / 'fit_receipt.json').write_text(json.dumps(final, indent=2)+'\n')
    print(json.dumps(final, indent=2), flush=True)


if __name__ == '__main__':
    main()
