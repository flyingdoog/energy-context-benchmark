"""Validation-only count-distribution pilot from unchanged saved marginals."""
import gc
import json
from pathlib import Path
import time
import run_local
import numpy as np
import pandas as pd
from field_control import BpaDataset, AcnDataset
from field_control.groups import GroupLayout
from field_control.acn_scoring import complete_cohort_mask
from field_control.activity_references import bpa_query_power, state_reference_predictions, score_activity_cohorts
from field_control.count_distribution import count_endpoints, count_scores, mixture_terms, fit_mixture
from bpa_update_validation_common import load_split

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/count_distribution_pilot_v1'
TWO = ROOT / 'results/operational_two_part_v1'
SPANS = (2, 6, 10)
LEAD_WEIGHT = np.mean([np.r_[np.full(b, 1 / b), np.zeros(10 - b)] for b in SPANS], axis=0)
FREQUENCIES = ROOT / 'results/operational_activity_references_v1/training_frequency_tables.json'


def saved_probability(path, meta, offset=0):
    with np.load(path, allow_pickle=False) as packet:
        index = pd.Index(packet['case_id'].astype(str))
        order = index.get_indexer(meta.case_id.astype(str))
        if not index.is_unique or (order < 0).any():
            raise ValueError('Saved probability IDs do not cover the validation cases')
        if not np.array_equal(packet['target_minutes'], offset + np.arange(11) * 15):
            raise ValueError('Saved probabilities use different target intervals')
        values = packet['positive_probability'][order]
    if values.shape != (len(meta), 10):
        raise ValueError('Expected unchanged ten-bin marginal forecasts')
    return values


def validation_contexts():
    tables = {(v['source'], v['site']): v for v in json.loads(FREQUENCIES.read_text())}
    data = BpaDataset(ROOT / 'results/benchmark_bpa_sequences_v1', 'winter')
    _, ix = load_split()
    meta = data.meta.iloc[ix].reset_index(drop=True)
    target = np.asarray(data.target[ix, 2:])
    power, observed = bpa_query_power(data, ix)
    frequency = state_reference_predictions(power, observed, tables[('BPA', 'Field study')])['State frequency']
    layout = GroupLayout(meta)
    group_rows = [layout.members.indices[layout.members.indptr[i]:layout.members.indptr[i + 1]]
                  for i in range(len(layout.meta))]
    group_meta = layout.meta.rename(columns={'cohort_id': 'query_group'}).copy()
    yield 'BPA', 'Field study', meta, target, {
        'State frequency': frequency,
        'Two-part': saved_probability(TWO / 'bpa/winter_validation.npz', meta, offset=30)}, group_meta, group_rows, None
    del data; gc.collect()
    data = AcnDataset(ROOT / 'results/acn_fleet_months_v1')
    for site, label in [('caltech', 'Caltech'), ('office001', 'Office')]:
        ix = np.flatnonzero((data.meta.site == site) & (data.meta.split == 'validation'))
        meta = data.meta.iloc[ix].reset_index(drop=True)
        inputs = data.inputs(ix, condition='power')
        frequency = state_reference_predictions(inputs['power_kw'][:, -1, 0],
            inputs['power_observed'][:, -1, 0], tables[('ACN', label)])['State frequency']
        duplicate_queries = set(meta.loc[meta.duplicated(['query_utc', 'station_id'], keep=False), 'query_utc'])
        keep = complete_cohort_mask(meta, data.population) & ~meta.query_utc.isin(duplicate_queries)
        group_rows, records = [], []
        for (date, query), group in meta.loc[keep].groupby(['local_date', 'query_utc'], sort=True):
            if len(group) < 2:
                continue
            group_rows.append(group.index.to_numpy())
            records.append({'local_date': date, 'query_group': query, 'devices': len(group)})
        yield 'ACN', label, meta, data.target[ix], {
            'State frequency': frequency,
            'Two-part': saved_probability(TWO / f'acn_{site}/all.npz', meta)
        }, pd.DataFrame(records), group_rows, data.population


def chronological_parts(group_meta):
    dates = sorted(group_meta.local_date.unique())
    if len(dates) < 5:
        raise ValueError('Insufficient validation dates for separate calibration/gap/holdout')
    half = len(dates) // 2
    calibration, gap, holdout = dates[:half], dates[half], dates[half + 1:]
    parts = np.where(group_meta.local_date.isin(calibration), 'calibration',
                     np.where(group_meta.local_date.isin(holdout), 'holdout', 'gap'))
    return parts, {'calibration_dates': calibration, 'gap_date': gap, 'holdout_dates': holdout}


def main():
    receipt_path = OUT / 'receipt.json'
    if receipt_path.exists() and json.loads(receipt_path.read_text()).get('completed'):
        print('Completed validation pilot exists; not repeating it.', flush=True)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    output, coefficients, support, support_checks = [], [], [], []
    max_expected_count_difference = 0.
    for source, site, meta, target, predictions, groups, rows, population in validation_contexts():
        parts, split = chronological_parts(groups)
        support.append({'source': source, 'site': site, 'validation_cases': len(meta),
            'groups': len(groups), 'dates': groups.local_date.nunique(),
            'calibration_groups': int((parts == 'calibration').sum()),
            'gap_groups': int((parts == 'gap').sum()), 'holdout_groups': int((parts == 'holdout').sum()),
            'member_min': int(groups.devices.min()), 'member_max': int(groups.devices.max()), **split})
        truth = [np.sum(target[member] > 0, axis=0) for member in rows]
        for marginal, probability in predictions.items():
            endpoints = [count_endpoints(probability[member]) for member in rows]
            nums, dens = zip(*(mixture_terms(a, b, y) for (a, b), y in zip(endpoints, truth)))
            nums, dens = np.array(nums) @ LEAD_WEIGHT, np.array(dens) @ LEAD_WEIGHT
            cal = np.flatnonzero(parts == 'calibration')
            cal_dates = groups.iloc[cal].local_date
            weights = (1 / cal_dates.map(cal_dates.value_counts())).to_numpy()
            coefficient = fit_mixture(nums[cal], dens[cal], weights)
            coefficients.append({'source': source, 'site': site, 'marginal': marginal,
                'coefficient': coefficient, 'objective': 'equal-date, equal-native-span normalized RPS',
                'calibration_groups': len(cal), **split})
            native, _ = score_activity_cohorts(source, meta, target, probability, marginal,
                                               'winter_validation' if source == 'BPA' else 'validation', population)
            recovered = []
            for i, ((independent, common), observed) in enumerate(zip(endpoints, truth)):
                expected = probability[rows[i]].sum(axis=0)
                for pmf in (independent, common):
                    difference = np.max(np.abs(pmf @ np.arange(len(rows[i]) + 1) - expected))
                    max_expected_count_difference = max(max_expected_count_difference, float(difference))
                for bins in SPANS:
                    recovered.append({'date': groups.iloc[i].local_date, 'span_minutes': bins * 15,
                                      'count_mse': np.mean((expected[:bins] - observed[:bins]) ** 2)})
                if parts[i] == 'gap':
                    continue
                for joint, pmf in [('Independent', independent), ('Shared uniform', common),
                                   ('Calibrated mixture', (1 - coefficient) * independent + coefficient * common)]:
                    scores = count_scores(pmf, observed)
                    for bins in SPANS:
                        output.append({'source': source, 'site': site, 'marginal': marginal, 'joint': joint,
                            'part': parts[i], 'date': groups.iloc[i].local_date,
                            'query_group': groups.iloc[i].query_group, 'members': int(groups.iloc[i].devices),
                            'span_minutes': bins * 15, **{k: float(v[:bins].mean()) for k, v in scores.items()}})
            recovered = pd.DataFrame(recovered).groupby(['span_minutes', 'date']).count_mse.mean()
            recovered = recovered.groupby('span_minutes').mean()
            check = native[native.scale == 'positive_count'].set_index('span_minutes')
            if not np.allclose(check.mse.loc[recovered.index], recovered, rtol=1e-7, atol=1e-7):
                raise ValueError('Group construction disagrees with original count scorer')
            if not (check.units == len(groups)).all() or not (check.dates == groups.local_date.nunique()).all():
                raise ValueError('Original group/date support changed')
            support_checks.append({'source': source, 'site': site, 'marginal': marginal,
                                   'count_score_cells_recovered': len(recovered)})
            print(json.dumps({'completed_context': [source, site, marginal], 'groups': len(groups),
                              'mixture_coefficient': coefficient}), flush=True)
    if max_expected_count_difference > 1e-8:
        raise ValueError('A joint model changed marginal expected counts')
    frame = pd.DataFrame(output)
    keys = ['source', 'site', 'marginal', 'joint', 'part', 'span_minutes']
    metrics = list(count_scores(np.array([[.5, .5]]), np.array([0])))
    daily = frame.groupby(keys + ['date'], as_index=False)[metrics].mean()
    summary = daily.groupby(keys, as_index=False)[metrics].mean()
    means = summary.groupby(keys[:-1], as_index=False)[metrics].mean()
    for label, data in [('group_scores', frame), ('daily_scores', daily), ('native_span_scores', summary), ('span_average', means)]:
        (OUT / f'{label}.json').write_text(data.to_json(orient='records', indent=2, double_precision=12) + '\n')
    for label, data in [('support', support), ('calibration', coefficients)]:
        (OUT / f'{label}.json').write_text(json.dumps(data, indent=2) + '\n')
    lines = ['# Count-distribution validation pilot', '',
        'Only the later portion of the original validation block is shown; this is not test performance.', '',
        '| Source / site | Marginal predictor | Mixture weight | Independent nRPS | Mixture nRPS | Independent / mixture 90% coverage | Independent / mixture 90% width |',
        '|---|---|---:|---:|---:|---:|---:|']
    for entry in coefficients:
        part = means[(means.source == entry['source']) & (means.site == entry['site']) &
            (means.marginal == entry['marginal']) & (means.part == 'holdout')].set_index('joint')
        a, b = part.loc['Independent'], part.loc['Calibrated mixture']
        lines.append(f"| {entry['source']} / {entry['site']} | {entry['marginal']} | {entry['coefficient']:.3f} | "
            f"{a.normalized_rps:.5f} | {b.normalized_rps:.5f} | {a.coverage_90:.3f} / {b.coverage_90:.3f} | "
            f"{a.width_90:.3f} / {b.width_90:.3f} |")
    lines += ['', 'All three joint forecasts preserve the same device marginals and expected-count MSE.',
        'nRPS divides the count RPS by query-known group size. Width is in devices.',
        'Full endpoints, raw RPS,80%/90% intervals and all native spans are in the JSON results.',
        'These finite validation blocks do not establish distribution-free coverage or later-period improvement.']
    (OUT / 'readout.md').write_text('\n'.join(lines) + '\n')
    receipt = {'completed': True, 'seconds': time.perf_counter() - start,
        'new_ml_fits': 0, 'scalar_calibrations': len(coefficients), 'gpu_used': False,
        'later_periods_scored': 0, 'count_score_cells_recovered': sum(r['count_score_cells_recovered'] for r in support_checks),
        'max_expected_count_difference': max_expected_count_difference,
        'native_score_rows': len(summary), 'group_score_rows': len(frame), 'original_support_checks': support_checks}
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    print('\n'.join(lines), flush=True)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__': main()
