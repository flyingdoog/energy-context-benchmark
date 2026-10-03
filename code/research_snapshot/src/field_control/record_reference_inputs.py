"""Default record-comparison inputs built from explicit prepared source roots."""
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .operational import OperationalForecastTask
from .dataset import STATE_NAMES, HISTORY_STATE_NAMES, CALENDAR_NAMES
from .acn_record_inputs import recover_query_inputs
from .record_linkage import donor_mapping

SEEDS = (2027, 2028, 2029)
CODES = (0, 1, 2, 3, 5, 7, 8, 9, 13)
HIST_NAMES = (['power_observed_fraction'] + list(HISTORY_STATE_NAMES) +
    [n+'_observed' for n in HISTORY_STATE_NAMES] +
    [f'past_command_is_{c}' for c in CODES] + ['past_command_parameter', 'past_command_observed'])
STATIC_NAMES = (list(STATE_NAMES) + [n+'_observed' for n in STATE_NAMES] +
    [f'current_command_is_{c}' for c in CODES] + ['current_command_parameter'] +
    [f'previous_{lag}_lead_{j}' for lag in ['day', 'week'] for j in range(12)] +
    [f'previous_{lag}_lead_{j}_fraction' for lag in ['day', 'week'] for j in range(12)])


def category_names(name):
    if name == 'oper_state':
        return [f'oper_state_is_{i}' for i in range(13)] + ['oper_state_manufacturer_reserved', 'oper_state_other']
    if name in ('running', 'opt_out'):
        return [f'{name}_is_0', f'{name}_is_1', f'{name}_other']
    return [name]


def typed(values, names):
    blocks = []
    for j, name in enumerate(names):
        x = np.asarray(values[..., j], dtype=np.float32)
        if name == 'oper_state':
            normal = x[..., None] == np.arange(13)
            reserved = np.isfinite(x) & (x >= 126) & (x <= 255)
            blocks.extend([normal, reserved[..., None],
                           (np.isfinite(x) & ~normal.any(axis=-1) & ~reserved)[..., None]])
        elif name in ('running', 'opt_out'):
            blocks.append(np.stack([x == 0, x == 1, np.isfinite(x) & (x != 0) & (x != 1)], axis=-1))
        else:
            blocks.append(x[..., None])
    return np.concatenate(blocks, axis=-1).astype(np.float32)


def bpa_names_and_moved():
    hist = [new for name in HIST_NAMES for new in category_names(name)]
    static = [new for name in STATIC_NAMES for new in category_names(name)]
    names = ([f'history_power:{j}' for j in range(96)] +
        [f'history{j}:{name}' for j in range(96) for name in hist] +
        [f'snapshot:{name}' for name in static] +
        [f'calendar{j}:{name}' for j in range(10) for name in CALENDAR_NAMES])
    moved = []
    for i, name in enumerate(names):
        prefix, leaf = name.split(':', 1)
        protected = (prefix == 'history_power' or prefix.startswith('calendar') or
            leaf in ('power_kw', 'power_kw_observed', 'power_observed_fraction') or leaf.startswith('previous_'))
        if not protected:
            moved.append(i)
    return names, np.asarray(moved, dtype=int)


def bpa_features(base, indices, batch_size=1024, progress=None):
    """Original latest-window tabular recipe at query +30, without label access."""
    base._power_prefix()
    view = copy.copy(base)
    view.meta = base.meta.copy()
    view.meta['origin_minute'] = view.meta.origin_minute + 30
    result = np.empty((len(indices), len(bpa_names_and_moved()[0])), dtype=np.float32)
    for start in range(0, len(indices), batch_size):
        ix = indices[start:start+batch_size]
        p = view.inputs(ix)
        hist = np.concatenate([p['past_power_observed_fraction'][..., None], p['past_state'],
            p['past_state_observed'].astype(np.float32),
            (p['past_command_type'][..., None] == np.asarray(CODES)).astype(np.float32),
            p['past_command_parameter'][..., None], p['past_command_observed'][..., None]], axis=-1)
        day, df = view.power_windows(ix, 1+15*np.arange(12)-1440)
        week, wf = view.power_windows(ix, 1+15*np.arange(12)-10080)
        static = np.c_[p['current_state'], p['current_state_observed'],
            (p['current_command'][:, :1] == np.asarray(CODES)).astype(np.float32),
            p['current_command'][:, 1], day, week, df, wf].astype(np.float32)
        rows = view.meta.iloc[ix]
        online = view.grid['online'][rows.device_index.to_numpy(), rows.origin_minute.to_numpy()]
        static[~online, :6] = np.nan
        static[~online, 6:21] = 0
        static[~online, 21] = np.nan
        result[start:start+len(ix)] = np.concatenate([p['past_power_kw'],
            typed(hist, HIST_NAMES).reshape(len(ix), -1), typed(static, STATIC_NAMES),
            p['future_calendar'][:, :10].reshape(len(ix), -1)], axis=1)
        if progress:
            progress(dict(stage='prepare_features', cases=min(start+batch_size, len(indices)), total=len(indices)))
    return result


def task_for(root, source, site, period):
    if source == 'BPA':
        season = 'winter' if period in ('train', 'validation') else period
        return OperationalForecastTask(root, 'BPA', season=season)
    return OperationalForecastTask(root, 'ACN', site=site)


def describe_inputs(root, source, site=None):
    if source not in ('ACN', 'BPA'):
        raise ValueError('Record comparison supports ACN and BPA')
    if (source == 'ACN' and site not in ('caltech', 'office001')) or (source == 'BPA' and site is not None):
        raise ValueError('ACN needs a site; BPA does not use a site argument')
    periods = ['train', 'validation'] + (['spring', 'summer'] if source == 'BPA' else ['autumn', 'winter'])
    support = []
    for period in periods:
        task = task_for(root, source, site, period)
        ix = task.indices(period)
        if source == 'ACN':
            pop = task.base.population
            pop = pop[(pop.site == site) & (pop.split == period)]
            queries, input_only = len(pop), len(pop)-len(ix)
            width = task.base._x.shape[1]
        else:
            queries, input_only, width = len(ix), 0, len(bpa_names_and_moved()[0])
        support.append(dict(period=period, labeled_cases=len(ix), query_known_cases=queries,
                            input_only_donors=input_only))
    return dict(source=source, site=site, support=support, feature_columns=width,
                feature_bytes=sum(r['query_known_cases'] for r in support)*width*4,
                required_original_source=source == 'ACN' and any(r['input_only_donors'] for r in support),
                fitted_heads=50, mapping_seeds=list(SEEDS), model_seed=2027,
                native_spans_minutes=[30, 90, 150], gpu_used=False, hpo=False)


def save_period(folder, meta, x, moved, task, period, minimum):
    folder.mkdir(parents=True, exist_ok=True)
    labeled_rows = np.flatnonzero(meta.native_index >= 0)
    labeled_rows = labeled_rows[np.argsort(meta.native_index.iloc[labeled_rows], kind='stable')]
    native = meta.native_index.iloc[labeled_rows].to_numpy(dtype=int)
    if not np.array_equal(native, task.indices(period)):
        raise ValueError('The prepared comparison must retain the full named split')
    meta.to_csv(folder/'metadata.csv.gz', index=False)
    np.save(folder/'features.npy', x, allow_pickle=False)
    np.savez_compressed(folder/'labeled.npz', rows=labeled_rows, target_kw=task.targets(native), moved=moved)
    for seed in SEEDS:
        donor, groups = donor_mapping(meta, minimum, seed)
        np.save(folder/f'donor_{seed}.npy', donor, allow_pickle=False)
        groups.to_json(folder/f'groups_{seed}.json', orient='records', indent=2)
    record = dict(completed=True, period=period, query_known_cases=len(meta), labeled_cases=len(native),
                  feature_columns=x.shape[1], moved_columns=len(moved), minimum_members=minimum)
    (folder/'complete.json').write_text(json.dumps(record, indent=2)+'\n')
    return record


def prepare_inputs(root, source, site, output, original_source=None, progress=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    original_source = Path(original_source).resolve() if original_source else None
    for source_path in [root] + ([original_source] if original_source else []):
        if output == source_path or output in source_path.parents or source_path in output.parents:
            raise ValueError('Keep input caches separate from source/prepared directories')
    plan = dict(data_root=str(root), source=source, site=site,
                original_source=str(original_source) if original_source else None,
                input_recipe='record-reference-v1', mapping_seeds=list(SEEDS))
    if (output/'plan.json').exists():
        if json.loads((output/'plan.json').read_text()) != plan:
            raise ValueError('Input cache belongs to a different source or recipe')
    elif output.exists() and any(output.iterdir()):
        raise ValueError('Use an empty input-cache folder')
    output.mkdir(parents=True, exist_ok=True)
    (output/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    description = describe_inputs(root, source, site)
    pending = [r['period'] for r in description['support'] if not (output/r['period']/'complete.json').exists()]
    if source == 'ACN' and pending:
        task = task_for(root, source, site, 'train')
        pop = task.base.population
        pop = pop[(pop.site == site) & pop.split.isin(pending)].reset_index(drop=True)
        known = pd.Index(task.base.meta.case_id).get_indexer(pop.case_id)
        if (known < 0).any():
            if original_source is None:
                raise ValueError('ACN input-only donors require --original-source with manifest.csv and sessions')
            x, recovery = recover_query_inputs(pop, task.base.meta, task.base._x, original_source, progress)
            if progress:
                progress(dict(stage='input_only_donors_recovered', **recovery))
        else:
            x = task.base._x[known].copy()
        native = pd.Index(task.meta.case_id).get_indexer(pop.case_id)
        moved = np.r_[np.arange(*task.base.feature_groups['control']), np.arange(*task.base.feature_groups['state'])]
        meta = pd.DataFrame(dict(source='ACN', site=pop.site, period=pop.split, query_group=pop.query_utc,
            member=pop.station_id, case_id=pop.case_id,
            date=pd.to_datetime(pop.query_utc, utc=True, format='mixed').dt.tz_convert('America/Los_Angeles').dt.date.astype(str),
            native_index=native))
        for period in pending:
            ix = np.flatnonzero(meta.period == period)
            save_period(output/period, meta.iloc[ix].reset_index(drop=True), x[ix], moved, task, period, 2)
    elif source == 'BPA':
        moved = bpa_names_and_moved()[1]
        for period in pending:
            task = task_for(root, source, site, period)
            ix = task.indices(period)
            raw = task.meta.iloc[ix].reset_index(drop=True)
            meta = pd.DataFrame(dict(source='BPA', site='Field study', period=period,
                query_group=raw.cohort_id.astype(str)+'|'+raw.command_type.astype(str),
                member=raw.alias.astype(str), case_id=raw.case_id, date=raw.local_date, native_index=ix))
            x = bpa_features(task.base.base, ix, progress=progress)
            save_period(output/period, meta, x, moved, task, period, 10)
            del task, x
    for item in description['support']:
        folder = output/item['period']
        required = ['complete.json', 'metadata.csv.gz', 'features.npy', 'labeled.npz'] + [f'donor_{s}.npy' for s in SEEDS]
        if not all((folder/name).is_file() for name in required):
            raise ValueError('A completed input period has missing files')
    (output/'complete.json').write_text(json.dumps(dict(completed=True, **description), indent=2)+'\n')
    return description
