"""Fixed-target value of partial field response, using default tree recipes.

Targets always start at minute31 after the original command report. Models at
report/+15/+30 receive the original context plus only the newly available
observations. Original cases, command classes, groups and dates remain fixed.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import time
import run_local
import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap
from lightgbm import Booster, LGBMRegressor
from field_control import BpaDataset, read_submission
from field_control.dataset import HISTORY_STATE_NAMES
from run_bpa_sequence_references import OUT, packet, tabular
from prepare_bpa_model_packets import CODES, STATIC_NAMES

DEST = OUT / 'fixed_target_update'
ENDS = (60, 120, 180)
EARLY_NAMES = (['power_kw', 'power_observed'] + list(HISTORY_STATE_NAMES) +
               [n+'_observed' for n in HISTORY_STATE_NAMES] +
               [f'command_is_{c}' for c in CODES] + ['command_numeric', 'command_observed'])
PHYSICAL_WIDTH = 2 + 2*len(HISTORY_STATE_NAMES)
DIMENSIONS = ['season', 'scale', 'command_type', 'target_end_minutes']


def feedback(data, ix, offset):
    if offset not in (15, 30):
        raise ValueError('Only +15/+30 updates are specified')
    meta = data.meta.iloc[np.asarray(ix)]
    d = meta.device_index.to_numpy()[:, None]
    t = meta.origin_minute.to_numpy()[:, None] + np.arange(1, offset+1)[None, :]
    online = data.grid['online'][d, t]
    power = data.grid['power_kw'][d, t].astype(np.float32)
    power[~online] = np.nan
    state = np.stack([data.grid[n][d, t] for n in HISTORY_STATE_NAMES], axis=-1).astype(np.float32)
    mask = online[..., None] & np.isfinite(state)
    state[~mask] = np.nan
    code = data.grid['command_type'][d, t]
    parameter = data.grid['command_numeric'][d, t].astype(np.float32)
    command_mask = online & np.isfinite(parameter)
    onehot = (code[..., None] == np.asarray(CODES)) & command_mask[..., None]
    parameter[~command_mask] = np.nan
    return np.concatenate([power[..., None], np.isfinite(power)[..., None], state, mask,
                           onehot, parameter[..., None], command_mask[..., None]], axis=-1).astype(np.float32)


def design(base, observed, offset, condition):
    if offset not in (15, 30):
        raise ValueError('Unspecified query offset')
    if condition == 'repeat_initial_command':
        # Tabular layout:96 power +96x22 history exogenous +70 static +48 calendar.
        # Static entries12:22 are current command one-hots and raw parameter.
        # The initial command is observed for every eligible source case.
        static_start = 96 + 96*22
        initial = np.c_[base[:, static_start+12:static_start+22], np.ones(len(base), dtype=np.float32)]
        return np.concatenate([base, np.tile(initial, (1, offset))], axis=1)
    values = observed[:, :offset]
    if condition == 'physical':
        values = values[:, :, :PHYSICAL_WIDTH]
    elif condition == 'command':
        values = values[:, :, PHYSICAL_WIDTH:]
    elif condition != 'full':
        raise ValueError('Unknown information condition')
    return np.concatenate([base, values.reshape(len(base), -1)], axis=1)


def score(meta, target, prediction, method, season):
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if target.shape != (len(meta), 10) or prediction.shape != target.shape:
        raise ValueError('Expected ten fixed absolute target bins')
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError('Incomplete case predictions cannot be dropped')
    if meta.duplicated(['cohort_id', 'alias']).any():
        raise ValueError('Repeated device inside an original group')
    keys = ['local_date', 'cohort_id', 'command_type']
    cols = [f'r{i}' for i in range(10)]
    frame = meta[keys].copy();frame[cols] = prediction-target
    grouped = frame.groupby(keys, sort=False)
    group = grouped[cols].mean().reset_index()
    group['devices'] = grouped.size().to_numpy()
    group = group[group.devices >= 10].copy()
    total = group.copy();total[cols] = total[cols].to_numpy()*total.devices.to_numpy()[:, None]
    records, days = [], []
    for scale, table in [('device', frame), ('cohort_mean', group), ('cohort_total', total)]:
        for end in ENDS:
            residual = table[cols[:(end-30)//15]].to_numpy()
            values = table[['local_date', 'command_type']].copy()
            values['mse'] = np.mean(residual**2, axis=1)
            values['mae'] = np.mean(np.abs(residual), axis=1)
            values['reported_energy_ae_kwh'] = np.abs(residual.sum(axis=1)*.25)
            for code in ['all', 1, 5, 8]:
                part = values if code == 'all' else values[values.command_type == code]
                if part.empty:
                    continue
                daily = part.groupby('local_date')[['mse', 'mae', 'reported_energy_ae_kwh']].mean()
                shared = dict(method=method, season=season, scale=scale, command_type=str(code),
                              target_start_minutes=31, target_end_minutes=end, forecasted_bins=(end-30)//15)
                records.append(dict(**shared, units=len(part), dates=len(daily), **daily.mean().to_dict()))
                days.append(daily.reset_index().assign(**shared))
    return pd.DataFrame(records), pd.concat(days, ignore_index=True)


def prepare(data):
    folder = DEST / 'feedback';folder.mkdir(exist_ok=True)
    outputs = {}
    for season, dataset in data.items():
        path = folder / f'{season}.npy'
        receipt = folder / f'{season}.json'
        if not receipt.exists():
            start = time.perf_counter()
            array = open_memmap(path, mode='w+', dtype=np.float32, shape=(len(dataset), 30, len(EARLY_NAMES)))
            for begin in range(0, len(dataset), 2048):
                ix = np.arange(begin, min(begin+2048, len(dataset)))
                array[ix] = feedback(dataset, ix, 30)
            array.flush();del array
            online_fraction = []
            for offset in (15, 30):
                d = dataset.meta.device_index.to_numpy()
                t = dataset.meta.origin_minute.to_numpy()+offset
                online_fraction.append(dict(query_offset=offset, online_fraction=float(dataset.grid['online'][d, t].mean())))
            receipt.write_text(json.dumps(dict(completed=True, season=season, cases=len(dataset),
                shape=[len(dataset), 30, len(EARLY_NAMES)], names=EARLY_NAMES,
                current_report_availability=online_fraction, seconds=time.perf_counter()-start), indent=2)+'\n')
            print(json.dumps({'feedback_complete':season}), flush=True)
        outputs[season] = np.load(path, mmap_mode='r')
    return outputs


def fit(offset, condition, data, base, observed):
    name = f'lgbm_{condition}_q{offset}'
    folder = DEST / name;folder.mkdir(exist_ok=True)
    receipt = folder / 'receipt.json'
    if receipt.exists() and json.loads(receipt.read_text()).get('completed'):
        return json.loads(receipt.read_text())
    start = time.perf_counter()
    x = {s: design(base[s], observed[s], offset, condition) for s in data}
    y = np.asarray(data['winter'].target[:, 2:])
    predictions = {s: np.empty((len(data[s]), 10), dtype=np.float32) for s in ['spring', 'summer']}
    newly_fitted = 0
    for lead in range(10):
        model_path = folder / f'absolute_bin_{lead+2:02d}.txt'
        if model_path.exists():
            model = Booster(model_file=str(model_path))
        else:
            estimator = LGBMRegressor(n_jobs=2, random_state=2027, verbosity=-1)
            estimator.fit(x['winter'], y[:, lead])
            model = estimator.booster_;model.save_model(str(model_path));newly_fitted += 1
        for s in predictions:
            predictions[s][:, lead] = np.maximum(model.predict(x[s], num_threads=2), 0)
        print(json.dumps({'method':name, 'leads_complete':lead+1,
                          'seconds':round(time.perf_counter()-start, 1)}), flush=True)
    for s, pred in predictions.items():
        np.savez_compressed(folder/f'{s}.npz', case_id=data[s].meta.case_id.to_numpy(dtype=str),
                            absolute_bins=np.arange(2,12), prediction_kw=pred)
    result = dict(completed=True, method=name, query_offset_minutes=offset, condition=condition,
                  features=x['winter'].shape[1], train_cases=len(y), training_season='winter',
                  newly_fitted_leads=newly_fitted, seconds=time.perf_counter()-start,
                  recipe='Same default LightGBM 100trees/31leaves/MSE/seed2027 as initial model; no HPO.')
    receipt.write_text(json.dumps(result, indent=2)+'\n')
    return result


def read_updated(path, case_ids):
    with np.load(path, allow_pickle=False) as z:
        ids = pd.Index(z['case_id'].astype(str));order = ids.get_indexer(np.asarray(case_ids).astype(str))
        if not ids.is_unique or len(ids) != len(case_ids) or (order < 0).any():
            raise ValueError('Update case set differs')
        if not np.array_equal(z['absolute_bins'], np.arange(2,12)):
            raise ValueError('Wrong absolute target bins')
        pred = z['prediction_kw'][order]
    if pred.shape != (len(case_ids), 10) or not np.isfinite(pred).all():
        raise ValueError('Invalid update prediction')
    return pred


def references(dataset, p, observed, initial):
    current = np.asarray(p['static_exog'][:, 0])
    indices = [STATIC_NAMES.index(f'previous_day_lead_{j}') for j in range(12)]
    day = np.asarray(p['static_exog'][:, indices])
    day = np.where(np.isfinite(day), day, current[:, None])
    pred = dict(lgbm_initial=initial[:, 2:], previous_day=day[:, 2:],
                persistence_q0=np.repeat(current[:, None], 10, axis=1))
    for offset in (15, 30):
        response = observed[:, :offset, 0]
        valid = np.isfinite(response)
        latest = response[:, -1]
        latest = np.where(np.isfinite(latest), latest, current)
        pred[f'persistence_q{offset}'] = np.repeat(latest[:, None], 10, axis=1)
        for source, name in [(initial, 'initial_residual'), (day, 'day_residual')]:
            expected = np.repeat(source[:, :offset//15], 15, axis=1)
            residual = np.where(valid, response-expected, 0).sum(axis=1)
            residual = np.divide(residual, valid.sum(axis=1), out=np.zeros_like(residual), where=valid.sum(axis=1)>0)
            pred[f'{name}_q{offset}'] = np.maximum(source[:, 2:]+residual[:, None], 0)
    return pred


def paired_contrasts(daily):
    result, weight_cache = [], {}
    for dims, frame in daily.groupby(DIMENSIONS):
        for metric in ['mse', 'mae', 'reported_energy_ae_kwh']:
            table = frame.pivot(index='local_date', columns='method', values=metric).sort_index()
            if table.isna().any().any():
                raise ValueError('Methods do not share complete date support')
            key = tuple(table.index)
            if key not in weight_cache:
                dates = pd.to_datetime(table.index)
                weeks = (dates-pd.to_timedelta(dates.dayofweek, unit='D')).strftime('%Y-%m-%d')
                unique = sorted(set(weeks));index = np.array([unique.index(w) for w in weeks])
                counts = np.random.default_rng(2027).multinomial(len(unique), np.ones(len(unique))/len(unique), size=1500)
                weights = counts[:, index].astype(float)
                weights /= weights.sum(axis=1, keepdims=True)
                weight_cache[key] = weights, len(unique)
            weights, n_weeks = weight_cache[key]
            boot = weights @ table.to_numpy();means = table.mean()
            pairs = [(m, 'lgbm_initial') for m in table.columns if m != 'lgbm_initial']
            pairs += [(f'lgbm_full_q{q}', f'lgbm_{c}_q{q}') for q in (15,30) for c in ['physical','command']]
            for method, reference in pairs:
                a, b = table.columns.get_loc(method), table.columns.get_loc(reference)
                values = 100*(1-boot[:, a]/np.maximum(boot[:, b], np.finfo(float).tiny))
                lo, hi = np.quantile(values, [.025,.975])
                result.append(dict(zip(DIMENSIONS,dims), metric=metric, method=method, reference=reference,
                    error=means[method], reference_error=means[reference],
                    reduction_pct=100*(1-means[method]/means[reference]) if means[reference]>0 else np.nan,
                    ci_low=lo, ci_high=hi, dates=len(table), weeks=n_weeks))
    return pd.DataFrame(result)


def main():
    start = time.perf_counter();DEST.mkdir(exist_ok=True)
    data = {s:BpaDataset(OUT,s) for s in ['winter','spring','summer']}
    plan = dict(task='Fixed absolute target bins31to180min after original command report; endpoints60/120/180.',
        population='Same original cases, groups >=10, command classes and dates at every query; no update-availability filtering.',
        inputs='Original full-information context retained. Additional minute1to15/30 telemetry only through updated query. Separate physical/command/full branches. No later actions or targets enter inputs.',
        models='Initial ten target estimators reused from the existing full-information default LightGBM; six new conditions each fit ten default lead models on winter only.',
        controls='Current-power persistence, previous day, and untrained initial/day forecast plus mean observed early residual.',
        scope='Information-value pilot, not a new method. Public as-issued full control plans are unavailable. Both seasons already used for project development.',
        workers=2, threads_per_worker=2)
    (DEST/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    observed = prepare(data)
    packs = {s:packet(s) for s in data}
    base = {s:tabular(packs[s]) for s in data}
    fits=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(fit,q,c,data,base,observed) for c in ['full','physical','command'] for q in (15,30)]
        for future in as_completed(futures):
            receipt=future.result();fits.append(receipt)
            print(json.dumps({'completed':receipt['method'],'done':len(fits),'total':6}),flush=True)
    del base
    rows,days=[],[]
    for season in ['spring','summer']:
        d=data[season]
        initial=read_submission(OUT/'references'/'lightgbm'/f'{season}_lightgbm.npz',d.meta.case_id)
        predictions=references(d,packs[season],observed[season],initial)
        for f in fits:
            predictions[f['method']]=read_updated(DEST/f['method']/f'{season}.npz',d.meta.case_id)
        for method,pred in predictions.items():
            summary,daily=score(d.meta,d.target[:,2:],pred,method,season)
            rows.append(summary);days.append(daily)
    summary=pd.concat(rows,ignore_index=True);daily=pd.concat(days,ignore_index=True)
    summary.to_csv(DEST/'summary.csv',index=False);daily.to_csv(DEST/'daily_scores.csv.gz',index=False)
    paired=paired_contrasts(daily);paired.to_csv(DEST/'paired_contrasts.csv',index=False)
    (DEST/'receipt.json').write_text(json.dumps(dict(completed=True,plan=plan,fits=fits,
        wall_seconds=time.perf_counter()-start,cases={s:len(d) for s,d in data.items()},
        inference='Date-balanced errors. Paired calendar-week cluster intervals conditional on the fixed dataset and estimator recipe; no independent multi-seed claim.'),indent=2)+'\n')
    show=paired[(paired.scale=='cohort_mean')&(paired.command_type=='all')&(paired.metric=='mse')&
                (paired.reference=='lgbm_initial')&paired.method.str.startswith('lgbm_')]
    print(show.to_string(index=False),flush=True)


if __name__=='__main__':main()
