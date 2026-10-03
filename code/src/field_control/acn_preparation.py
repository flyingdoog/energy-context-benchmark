"""Reusable raw ACN session preparation; no historical outputs or model library."""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from .acn import read_static, minute_grid, history_features, future_labels

GROUPS = ('power', 'control', 'state', 'calendar')
CASE_COLUMNS = ['case_id', 'site', 'split', 'filename', 'station_id', 'query_utc',
                'local_date', 'connection_utc', 'persistence_kw']
POPULATION_COLUMNS = ['case_id', 'site', 'split', 'query_utc', 'station_id', 'target_complete']


def read_manifest(source):
    """Read explicit calendar contexts without inferring a split from labels."""
    source = Path(source).resolve()
    path = source / 'receipt.json'
    if path.exists() and not json.loads(path.read_text()).get('completed'):
        raise ValueError('Resolve incomplete source acquisition before preparation')
    manifest = pd.read_csv(source / 'manifest.csv')
    required = {'site', 'split', 'station_id', 'connection_utc', 'filename',
                'query_start_local', 'query_end_exclusive_local'}
    if not required.issubset(manifest.columns) or manifest.empty:
        raise ValueError('Manifest needs source identities and explicit query calendar bounds')
    if manifest[list(required)].isna().any().any():
        raise ValueError('Source identities and query calendar bounds must not be missing')
    if manifest.duplicated(['site', 'split', 'filename']).any():
        raise ValueError('Duplicate session/calendar contexts')
    if not manifest.site.isin(['caltech', 'office001']).all():
        raise ValueError('Unknown ACN site')
    if not manifest.split.isin(['train', 'validation', 'autumn', 'winter']).all():
        raise ValueError('Unknown ACN split')
    for row in manifest.to_dict('records'):
        filename = row['filename']
        if not isinstance(filename, str) or '/' in filename or '\\' in filename or not filename.endswith('.csv.gz'):
            raise ValueError('Session filename must name one local compressed CSV')
        path = (source / 'sessions' / filename).resolve()
        if path.parent != (source / 'sessions').resolve() or not path.is_file():
            raise ValueError(f'Missing source session: {filename}')
        connected = pd.Timestamp(row['connection_utc'])
        if connected.tzinfo is None:
            raise ValueError('Connection time requires its source timezone')
        lower = pd.Timestamp(row['query_start_local'], tz='America/Los_Angeles')
        upper = pd.Timestamp(row['query_end_exclusive_local'], tz='America/Los_Angeles')
        if upper <= lower:
            raise ValueError('Empty or reversed query calendar interval')
    return manifest


def build_acn_packet(source, destination, shards=1, shard_index=0, manifest=None):
    """Preserve the existing minute coverage, history and target construction."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError('Keep the source and preparation output in separate directories')
    if not 1 <= shards <= 4 or not 0 <= shard_index < shards:
        raise ValueError('Invalid preparation shard')
    manifest = read_manifest(source) if manifest is None else manifest.copy()
    manifest = manifest.iloc[shard_index::shards].copy()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('Use an empty preparation destination')
    destination.mkdir(parents=True, exist_ok=True)
    meta, features, targets, summaries, population = [], [], [], [], []
    starts = time.monotonic()
    boundaries = None
    for count, row in enumerate(manifest.to_dict('records'), 1):
        path = source / 'sessions' / row['filename']
        summary = dict(site=row['site'], split=row['split'], filename=row['filename'],
                       query_candidates=0, usable_history=0, complete_targets=0)
        try:
            if not path.exists():
                raise ValueError('Source file unavailable')
            frame = read_static(path)
            for name in ['Power (kW)', 'Actual Pilot (A)', 'Charging State']:
                summary[name + ' observed_fraction'] = float(frame[name].notna().mean())
            if not frame['Power (kW)'].notna().any():
                raise ValueError('No reported power in this session')
            minutes, coverage, receipt = minute_grid(frame)
            summary.update(terminal_unplugged=receipt['terminal_unplugged'],
                           first=receipt['first'], last=receipt['last'])
            connected = pd.Timestamp(row['connection_utc'])
            queries = pd.date_range((connected+pd.Timedelta(minutes=30)).ceil('h'), frame.index[-1].floor('h'), freq='h')
            if pd.notna(row.get('query_start_local')):
                lower=pd.Timestamp(row['query_start_local'],tz='America/Los_Angeles')
                upper=pd.Timestamp(row['query_end_exclusive_local'],tz='America/Los_Angeles')
                queries=queries[(queries>=lower)&(queries+pd.Timedelta(minutes=150)<=upper)]
            summary['query_candidates'] = len(queries)
            for query in queries:
                past_index = frame.index.searchsorted(query, side='right') - 1
                if past_index < 0 or (query-frame.index[past_index]).total_seconds() > 10:
                    continue
                if frame['Charging State'].iloc[past_index] == 'UNPLUGGED':
                    continue
                recent = minutes.reindex(pd.date_range(query-pd.Timedelta(minutes=30), periods=30, freq='min'))
                if recent['Power (kW)'].notna().sum() < 24:
                    continue
                summary['usable_history'] += 1
                target = future_labels(minutes, query)
                population.append(dict(case_id=row['filename']+'@'+query.isoformat(),site=row['site'],
                                       split=row['split'],query_utc=query.isoformat(),station_id=row['station_id'],
                                       target_complete=bool(np.isfinite(target).all())))
                if not np.isfinite(target).all():
                    continue
                feature = history_features(minutes, query, connected)
                if boundaries is None:
                    lengths = [len(feature[name]) for name in GROUPS]
                    ends = np.cumsum([0] + lengths)
                    boundaries = {name: [int(a), int(b)] for name, a, b in zip(GROUPS, ends[:-1], ends[1:])}
                vector = np.concatenate([feature[name] for name in GROUPS]).astype(np.float32)
                if not np.isfinite(vector).all():
                    raise ValueError('Nonfinite model feature')
                recent_power = recent['Power (kW)'].dropna()
                local = query.tz_convert('America/Los_Angeles')
                meta.append(dict(case_id=row['filename']+'@'+query.isoformat(), site=row['site'], split=row['split'],
                                 filename=row['filename'], station_id=row['station_id'], query_utc=query.isoformat(),
                                 local_date=str(local.date()), connection_utc=row['connection_utc'],
                                 persistence_kw=float(recent_power.iloc[-1])))
                features.append(vector)
                targets.append(target.astype(np.float32))
                summary['complete_targets'] += 1
            summary['status'] = 'processed'
        except (ValueError, OSError) as error:
            summary.update(status='unusable source', error=str(error))
        summaries.append(summary)
        if count % 100 == 0:
            print(json.dumps(dict(files=count, cases=len(meta), seconds=time.monotonic()-starts)), flush=True)
    if not meta and shards == 1:
        raise ValueError('No supported forecast cases')
    meta = pd.DataFrame(meta, columns=CASE_COLUMNS)
    population = pd.DataFrame(population, columns=POPULATION_COLUMNS)
    if not meta.case_id.is_unique or not population.case_id.is_unique:
        raise ValueError('Duplicate query cases across source contexts')
    x = np.stack(features) if features else np.empty((0, 1445), dtype=np.float32)
    y = np.stack(targets) if targets else np.empty((0, 10), dtype=np.float32)
    boundaries = boundaries or {'power': [0,120], 'control': [120,240],
                                'state': [240,1440], 'calendar': [1440,1445]}
    meta.to_csv(destination / 'cases.csv.gz', index=False)
    pd.DataFrame(population).to_csv(destination / 'query_population.csv.gz', index=False)
    np.savez_compressed(destination / 'packet.npz', x=x, y=y)
    pd.DataFrame(summaries).to_csv(destination / 'source_coverage.csv', index=False)
    counts = meta.groupby(['site', 'split']).agg(cases=('case_id', 'size'), sessions=('filename', 'nunique'),
                                               dates=('local_date', 'nunique'), stations=('station_id', 'nunique')).reset_index()
    counts.to_csv(destination / 'support.csv', index=False)
    receipt = dict(completed=True, source_files=int(manifest.filename.nunique()),manifest_contexts=len(manifest),cases=len(meta), query_known_cases=len(population), input_dimensions=x.shape[1],
                   feature_groups=boundaries, seconds=time.monotonic()-starts,
                   task='Hourly queries of the currently observed source-session cohort; no future arrivals.',
                   history='Sixty closed minutes; at least24/30 recent minutes with reported power; latest source report<=10s old.',
                   target='Ten 15min power means after query; same cases for all methods/conditions/spans.',
                   zero_extension='Only after explicit terminal UNPLUGGED, for the same departed session.',
                   missing='Input masks retained; incomplete target support counted before exclusion; no missing-tail zero filling.',
                   grouping='Define membership from query-available history before target inspection. Score a cohort only if every member has complete labels; do not shrink it after future missingness. Minimum two devices.')
    (destination / 'packet_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(counts.to_string(index=False), flush=True)
    print(json.dumps(receipt, indent=2), flush=True)
    return receipt

