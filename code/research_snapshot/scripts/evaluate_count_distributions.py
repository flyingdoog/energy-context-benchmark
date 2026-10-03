"""Unchanged device marginals, validation-fitted count laws, all later groups."""
import gzip
import json
from pathlib import Path
import time
import run_local
import numpy as np
import pandas as pd
from field_control.groups import GroupLayout
from field_control.acn_scoring import complete_cohort_mask
from field_control.count_distribution import count_endpoints, count_scores, mixture_terms, fit_mixture
from pilot_count_distributions import validation_contexts, LEAD_WEIGHT, SPANS
from analyze_operational_activity import contexts

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/count_distribution_later_v1'
PROB = ROOT / 'results/operational_activity_references_v1'
MARGINALS = ('State frequency', 'Two-part')
JOINTS = ('Independent', 'Shared uniform', 'Calibrated mixture')
KEY = ['source', 'site', 'period', 'marginal', 'joint', 'span_minutes']
METRICS = list(count_scores(np.array([[.5, .5]]), np.array([0])))


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def save_frame(path, frame):
    frame.to_json(path, orient='records', indent=2, double_precision=12,
                  compression='gzip' if path.name.endswith('.gz') else None)


def progress(value):
    value = dict(at=pd.Timestamp.now(tz='UTC').isoformat(), **value)
    with (OUT / 'progress.jsonl').open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(value) + '\n')
    print(json.dumps(value), flush=True)


def full_validation_calibration():
    path = OUT / 'calibration.json'
    if path.exists():
        records = json.loads(path.read_text())
        if len(records) != 6 or any(r['scope'] != 'complete_original_validation' for r in records):
            raise ValueError('Existing coefficients are not this evaluation recipe')
        progress({'stage': 'reuse_full_validation_calibration'})
        return records
    records = []
    for source, site, _, target, forecasts, groups, rows, _ in validation_contexts():
        weights = (1 / groups.local_date.map(groups.local_date.value_counts())).to_numpy()
        for marginal in MARGINALS:
            num, den = [], []
            for member in rows:
                a, b = count_endpoints(forecasts[marginal][member])
                u, v = mixture_terms(a, b, np.sum(target[member] > 0, axis=0))
                num.append(u @ LEAD_WEIGHT); den.append(v @ LEAD_WEIGHT)
            record = dict(source=source, site=site, marginal=marginal,
                coefficient=fit_mixture(num, den, weights), groups=len(groups),
                dates=groups.local_date.nunique(), first_date=min(groups.local_date),
                last_date=max(groups.local_date), scope='complete_original_validation',
                objective='Equal-date, equal-native-span RPS divided by query-known group size',
                shared_across_all_leads_and_later_periods=True)
            records.append(record)
            progress(dict(stage='calibrated', **record))
    save_json(path, records)
    return records


def group_layout(source, meta, population=None):
    meta = meta.reset_index(drop=True)
    if source == 'BPA':
        layout = GroupLayout(meta)
        rows = [layout.members.indices[layout.members.indptr[i]:layout.members.indptr[i + 1]]
                for i in range(len(layout.meta))]
        return layout.meta.rename(columns={'cohort_id': 'query_group'}).copy(), rows
    if source != 'ACN' or population is None:
        raise ValueError('Only BPA or ACN with complete-population metadata is supported')
    duplicates = meta.loc[meta.duplicated(['query_utc', 'station_id'], keep=False)]
    keep = complete_cohort_mask(meta, population) & ~meta.query_utc.isin(duplicates.query_utc)
    groups, rows = [], []
    for (date, query), frame in meta.loc[keep].groupby(['local_date', 'query_utc'], sort=True):
        if len(frame) >= 2:
            groups.append(dict(local_date=date, query_group=query, devices=len(frame)))
            rows.append(frame.index.to_numpy())
    if not groups:
        raise ValueError('No originally supported groups')
    return pd.DataFrame(groups), rows


def read_marginals(source, site, period, meta):
    folder = '_'.join((source, site.replace(' ', '_'), period))
    with np.load(PROB / folder / 'probabilities.npz', allow_pickle=False) as packet:
        ids = pd.Index(packet['case_id'].astype(str))
        expected = meta.case_id.to_numpy(dtype=str)
        if not ids.is_unique or len(ids) != len(expected):
            raise ValueError('Saved marginal case set differs')
        order = ids.get_indexer(expected)
        offset = 30 if source == 'BPA' else 0
        if (order < 0).any() or not np.array_equal(packet['target_minutes'], offset + np.arange(11) * 15):
            raise ValueError('Saved marginal cases or native future bins differ')
        methods = packet['method'].tolist()
        return {name: packet['probability'][order, methods.index(name)] for name in MARGINALS}


def score_context(source, site, period, meta, target, population, coefficients):
    tag = '_'.join((source, site.replace(' ', '_'), period))
    dest = OUT / 'contexts' / tag
    receipt_path = dest / 'receipt.json'
    if receipt_path.exists() and json.loads(receipt_path.read_text()).get('completed'):
        progress(dict(stage='reuse_scored_context', context=tag))
        return pd.read_json(dest / 'daily_scores.json'), json.loads(receipt_path.read_text())
    dest.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    groups, members = group_layout(source, meta, population)
    predictions = read_marginals(source, site, period, meta)
    outcomes = [np.sum(target[ix] > 0, axis=0) for ix in members]
    rows = []; max_mean_difference = 0.
    for marginal in MARGINALS:
        probability = predictions[marginal]
        coefficient = coefficients[(source, site, marginal)]
        for i, (ix, truth) in enumerate(zip(members, outcomes)):
            a, b = count_endpoints(probability[ix])
            expected = probability[ix].sum(axis=0)
            for joint, pmf in [('Independent', a), ('Shared uniform', b),
                               ('Calibrated mixture', (1 - coefficient) * a + coefficient * b)]:
                max_mean_difference = max(max_mean_difference, float(np.max(np.abs(pmf @ np.arange(len(ix) + 1) - expected))))
                scores = count_scores(pmf, truth)
                for bins in SPANS:
                    rows.append(dict(source=source, site=site, period=period,
                        marginal=marginal, joint=joint, span_minutes=15 * bins,
                        date=groups.iloc[i].local_date, query_group=groups.iloc[i].query_group,
                        members=len(ix), **{name: float(values[:bins].mean()) for name, values in scores.items()}))
        progress(dict(stage='scored_marginal', source=source, site=site, period=period,
                      marginal=marginal, groups=len(groups)))
    if max_mean_difference > 1e-8:
        raise ValueError('A joint count distribution changed its expected count')
    frame = pd.DataFrame(rows)
    daily = frame.groupby(KEY + ['date'], as_index=False)[METRICS].mean()
    native = daily.groupby(KEY, as_index=False)[METRICS].mean()
    old = pd.read_csv(PROB / 'cohort_summary.csv')
    old = old[(old.source == source) & (old.site == site) & (old.period == period)
              & (old.scale == 'positive_count') & old.method.isin(MARGINALS)]
    new = native[native.joint == 'Independent']
    check = new.merge(old, left_on=['marginal', 'span_minutes'], right_on=['method', 'span_minutes'], validate='one_to_one')
    if len(check) != 6 or not np.allclose(check.count_mse, check.mse, rtol=1e-7, atol=1e-7):
        raise ValueError('Expected-count scores do not recover the original reference')
    if not (check.units == len(groups)).all() or not (check.dates == groups.local_date.nunique()).all():
        raise ValueError('Original group/date support changed')
    save_frame(dest / 'group_scores.json.gz', frame)
    save_frame(dest / 'daily_scores.json', daily)
    save_frame(dest / 'native_span_scores.json', native)
    info = dict(completed=True, source=source, site=site, period=period,
        source_cases=len(meta), groups=len(groups), dates=groups.local_date.nunique(),
        min_members=int(groups.devices.min()), max_members=int(groups.devices.max()),
        score_rows=len(native), original_count_cells_recovered=6,
        max_expected_count_difference=max_mean_difference, seconds=time.perf_counter() - start)
    save_json(receipt_path, info)
    return daily, info


def main():
    path = OUT / 'receipt.json'
    if path.exists() and json.loads(path.read_text()).get('completed'):
        print('Full later-period evaluation is complete; not repeating it.', flush=True)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    records = full_validation_calibration()
    coefficients = {(r['source'], r['site'], r['marginal']): r['coefficient'] for r in records}
    population = pd.read_csv(ROOT / 'results/acn_fleet_months_v1/query_population.csv.gz')
    pieces, receipts = [], []
    for source, site, period, meta, target, step, _ in contexts():
        if source == 'EPFL':
            continue
        if step != 15:
            raise ValueError('Expected existing ten-bin BPA/ACN target intervals')
        daily, receipt = score_context(source, site, period, meta.reset_index(drop=True), target, population, coefficients)
        pieces.append(daily); receipts.append(receipt)
    if len(receipts) != 6:
        raise ValueError('Incomplete later-period comparison')
    daily = pd.concat(pieces, ignore_index=True)
    native = daily.groupby(KEY, as_index=False)[METRICS].mean()
    average = native.groupby(KEY[:-1], as_index=False)[METRICS].mean()
    save_frame(OUT / 'daily_scores.json', daily)
    save_frame(OUT / 'native_span_scores.json', native)
    save_frame(OUT / 'span_average.json', average)
    result = dict(completed=True, seconds=time.perf_counter() - started,
        scalar_calibrations=6, new_ml_fits=0, gpu_used=False, periods=receipts,
        cases=sum(r['source_cases'] for r in receipts), groups=sum(r['groups'] for r in receipts),
        native_score_rows=len(native), original_count_cells_recovered=36,
        recipe='Unchanged saved marginals; six full-validation scalar mixtures; all six original later group periods')
    save_json(path, result)
    progress(dict(stage='evaluation_complete', seconds=result['seconds'], groups=result['groups'], cases=result['cases']))


if __name__ == '__main__': main()
