"""Prepare the original BPA archives for model-independent forecasting."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import gc
import json
import time
import zipfile
import numpy as np
import pandas as pd

SEASONS = ('winter', 'spring', 'summer')


def enable_deflate64():
    """Process-local support for the publisher's method-9 ZIP; no disk changes."""
    if getattr(zipfile._get_decompressor, '_bpa_method9', False):
        return
    original = zipfile._get_decompressor
    class Reader:
        def __init__(self):
            import inflate64
            self.decoder = inflate64.Inflater()
        def decompress(self, data):
            return self.decoder.inflate(data)
        @property
        def eof(self):
            return self.decoder.eof
    def decompressor(method):
        return Reader() if method == 9 else original(method)
    decompressor._bpa_method9 = True
    zipfile._get_decompressor = decompressor


def scan_archive(source, season):
    """Read only time/identity columns to size the original minute grids."""
    enable_deflate64()
    archive_path = Path(source) / f'{season}2018.zip'
    devices = set()
    first = last = None
    rows = 0
    with zipfile.ZipFile(archive_path) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith('.csv')]
        if len(csv_names) != 1:
            raise ValueError('Expected one original CSV per BPA seasonal archive')
        with archive.open(csv_names[0]) as stream:
            for frame in pd.read_csv(stream, usecols=['local_time', 'alias'], chunksize=500_000):
                devices.update(frame.alias.unique().tolist())
                stamps = frame.local_time.astype(str)
                lo, hi = stamps.min(), stamps.max()
                first = min(first, lo) if first else lo
                last = max(last, hi) if last else hi
                rows += len(frame)
                if rows % 5_000_000 == 0:
                    print(json.dumps(dict(stage='dimensions', season=season, rows=rows)), flush=True)
    if not rows or not devices:
        raise ValueError('Empty BPA archive')
    return dict(season=season, rows=rows, devices=len(devices),
                first_timestamp=first, last_timestamp=last, members=csv_names)


def archive_info(source, output, season):
    folder = Path(output) / 'source_metadata'
    folder.mkdir(exist_ok=True)
    path = folder / f'{season}.json'
    if path.exists():
        return json.loads(path.read_text())
    info = scan_archive(source, season)
    path.write_text(json.dumps(info, indent=2) + '\n')
    return info

FIELDS = {'curr_watts': ('power', np.float32, np.nan),
          'curr_curtail_id': ('command_id', np.int32, -1),
          'curr_curtail_type': ('command_type', np.int16, -1),
          'curr_curtail_numeric': ('command_numeric', np.float32, np.nan),
          'wifi_online': ('wifi', np.uint8, 255),
          'opt_out': ('opt_out', np.uint8, 255),
          'present_whr': ('present', np.float32, np.nan),
          'total_whr': ('capacity', np.float32, np.nan)}

EXTRA_FIELDS = dict(FIELDS, oper_state=('oper_state', np.float32, np.nan),
                    running=('running', np.float32, np.nan))

def segment_mean(values, starts, width=15, min_count=12):
    """Mean over [start,start+width), never replacing absent samples with zero."""
    starts = np.asarray(starts, dtype=np.int64)
    valid = np.isfinite(values)
    sums = np.r_[0.0, np.cumsum(np.where(valid, values, 0.0), dtype=np.float64)]
    counts = np.r_[0, np.cumsum(valid, dtype=np.int64)]
    count = counts[starts + width] - counts[starts]
    total = sums[starts + width] - sums[starts]
    return np.where(count >= min_count, total / np.maximum(count, 1), np.nan)


def command_onsets(command_id, command_type, command_numeric, seen, wifi):
    """New commands need an observed previous minute: reconnects aren't onsets."""
    changed = ((command_id[1:] != command_id[:-1]) |
               (command_type[1:] != command_type[:-1]) |
               (command_numeric[1:] != command_numeric[:-1]))
    adjacent = seen[1:] & seen[:-1] & (wifi[1:] == 1) & (wifi[:-1] == 1)
    return np.flatnonzero(changed & adjacent) + 1


def read_season(source, season, fields=None, info=None):
    fields = FIELDS if fields is None else fields
    source = Path(source)
    enable_deflate64()
    if info is None:
        info = scan_archive(source, season)
    first = pd.Timestamp(info['first_timestamp'])
    last = pd.Timestamp(info['last_timestamp'])
    length = int((last - first).total_seconds() // 60) + 1
    width = info['devices']
    arrays = {name: np.full((width, length), fill, dtype=dtype)
              for name, dtype, fill in fields.values()}
    seen = np.zeros((width, length), dtype=np.bool_)
    conflicted = np.zeros((width, length), dtype=np.bool_)
    aliases = {}
    raw_rows = duplicate_rows = off_grid = 0
    conflict_examples = []
    with zipfile.ZipFile(source / f'{season}2018.zip') as archive:
        csv_name, = [n for n in archive.namelist() if n.lower().endswith('.csv')]
        with archive.open(csv_name) as stream:
            for frame in pd.read_csv(stream, usecols=['local_time', 'alias'] + list(fields),
                                     chunksize=500_000):
                raw_count = len(frame)
                duplicate_rows += raw_count - len(frame.drop_duplicates())
                frame = frame.drop_duplicates()
                # Only unique minute strings need parsing, not every device row.
                codes, times = pd.factorize(frame.local_time, sort=False)
                times = pd.to_datetime(times, utc=True)
                seconds = (times - first).total_seconds().to_numpy()
                off_grid += int(np.count_nonzero((seconds[codes] % 60) != 0))
                if np.any(seconds % 60):
                    raise ValueError('Timestamp not on the one-minute source grid')
                minute = (seconds // 60).astype(np.int64)[codes]
                for alias in frame.alias.unique():
                    if alias not in aliases:
                        aliases[alias] = len(aliases)
                dev = frame.alias.map(aliases).to_numpy(dtype=np.int64)
                linear = dev * length + minute
                # Nonidentical rows at one device/minute must not be averaged into
                # a new label. Preserve their existence; mark the minute missing.
                within_conflict = pd.Series(linear).duplicated(keep=False).to_numpy()
                across = seen[dev, minute]
                same = across.copy()
                for source, (name, dtype, _) in fields.items():
                    old = arrays[name][dev, minute]
                    new = frame[source].to_numpy(dtype=dtype)
                    same &= (old == new) | (pd.isna(old) & pd.isna(new))
                duplicate_rows += int(same.sum())
                bad = within_conflict | (across & ~same) | conflicted[dev, minute]
                if bad.any():
                    for pos in np.flatnonzero(bad)[:max(0, 20-len(conflict_examples))]:
                        conflict_examples.append(dict(timestamp=str(frame.local_time.iloc[pos]),
                            alias=str(frame.alias.iloc[pos]),
                            previous_power=float(arrays['power'][dev[pos], minute[pos]]),
                            incoming_power=float(frame.curr_watts.iloc[pos])))
                    conflicted[dev[bad], minute[bad]] = True
                    seen[dev[bad], minute[bad]] = False
                good = ~bad
                frame = frame.iloc[np.flatnonzero(good)]
                dev, minute = dev[good], minute[good]
                seen[dev, minute] = True
                for source, (name, dtype, _) in fields.items():
                    arrays[name][dev, minute] = frame[source].to_numpy(dtype=dtype)
                raw_rows += raw_count
                if raw_rows % 5_000_000 == 0:
                    print(json.dumps({'stage': 'read', 'season': season, 'rows': raw_rows}), flush=True)
    return arrays, seen, list(aliases), first, dict(raw_rows=raw_rows,
        identical_duplicate_rows=duplicate_rows,
        conflicting_device_minutes=int(conflicted.sum()), conflict_examples=conflict_examples,
        off_grid_rows=off_grid, minutes=length,
        devices=len(aliases))


def build_season(source, output, season):
    folder = Path(output)/season
    if (folder/'receipt.json').exists():
        saved = json.loads((folder/'receipt.json').read_text())
        required = ['cases.csv.gz', 'target_kw.npy', 'target_observed_fraction.npy'] + [f'grid/{name}.npy' for name in saved['grid_fields']]
        if not all((folder/name).is_file() for name in required):
            raise ValueError('Completed BPA season has missing output files')
        print(json.dumps({'season': season, 'completed_release_reused': True}), flush=True)
        return json.loads((folder/'receipt.json').read_text())
    tick = time.perf_counter()
    folder.mkdir(parents=True, exist_ok=True)
    info = archive_info(source, output, season)
    arrays, seen, aliases, first, coverage = read_season(source, season, fields=EXTRA_FIELDS, info=info)
    online = seen & (arrays['wifi'] == 1)
    power = arrays.pop('power')/1000.0
    power[~online] = np.nan
    arrays.update(power_kw=power, online=online, seen=seen)
    rows, targets, fractions = [], [], []
    excluded = Counter()
    future_offsets = 1 + 15*np.arange(12)
    for dev, alias in enumerate(aliases):
        onsets = command_onsets(arrays['command_id'][dev], arrays['command_type'][dev],
            arrays['command_numeric'][dev], seen[dev], arrays['wifi'][dev])
        eligible = onsets[np.isin(arrays['command_type'][dev, onsets], [1, 5, 8])]
        excluded['other_command_types'] += len(onsets)-len(eligible)
        bounds = eligible[(eligible >= 10080) & (eligible+181 <= power.shape[1])]
        excluded['seven_day_context_or_180minute_boundary'] += len(eligible)-len(bounds)
        if not len(bounds):
            continue
        y = segment_mean(power[dev], bounds[:, None]+future_offsets, width=15, min_count=12)
        history_mean = segment_mean(power[dev], bounds-1440, width=1440, min_count=1152)
        count_prefix = np.r_[0, np.cumsum(np.isfinite(power[dev]), dtype=np.int32)]
        starts = bounds[:, None]+future_offsets
        fraction = (count_prefix[starts+15]-count_prefix[starts])/15.0
        good = (np.isfinite(y).all(axis=1) & np.isfinite(history_mean) &
                np.isfinite(power[dev, bounds]) & np.isfinite(arrays['command_numeric'][dev, bounds]))
        excluded['current_power_command_or_history_target_coverage'] += int((~good).sum())
        for j in np.flatnonzero(good):
            t = int(bounds[j]);stamp = first + pd.Timedelta(minutes=t)
            code = int(arrays['command_type'][dev, t]);parameter = float(arrays['command_numeric'][dev, t])
            command_id = int(arrays['command_id'][dev, t])
            rows.append(dict(case_id=f'bpa-v1|{season}|{alias}|{t}', season=season, alias=str(alias),
                device_index=dev, origin_minute=t, origin_utc=stamp.isoformat(),
                local_date=str(stamp.tz_convert('America/Los_Angeles').date()),
                command_type=code, command_numeric=parameter, command_id=command_id,
                cohort_id=f'{stamp.isoformat()}|{command_id}|{code}|{parameter}'))
            targets.append(y[j]);fractions.append(fraction[j])
    meta = pd.DataFrame(rows)
    meta.to_csv(folder/'cases.csv.gz', index=False)
    np.save(folder/'target_kw.npy', np.asarray(targets, dtype=np.float32), allow_pickle=False)
    np.save(folder/'target_observed_fraction.npy', np.asarray(fractions, dtype=np.float32), allow_pickle=False)
    (folder/'grid').mkdir(exist_ok=True)
    for name, values in arrays.items():
        np.save(folder/'grid'/f'{name}.npy', values, allow_pickle=False)
    group_sizes = meta.groupby('cohort_id').size()
    receipt = dict(schema='bpa-sequences-v1', season=season, **coverage,
        first_timestamp=first.isoformat(), aliases=list(map(str, aliases)),
        grid_fields=list(arrays), cases=len(meta), eligible_devices=int(meta.alias.nunique()),
        dates=int(meta.local_date.nunique()), synchronous_groups=len(group_sizes),
        groups_at_least10=int((group_sizes >= 10).sum()),
        exclusions={k: int(v) for k,v in excluded.items()}, wall_seconds=time.perf_counter()-tick,
        context='Up to seven days of source history; at least 80% online power reports in the last 24h.',
        labels='12 successive 15-sample bins at origin+1..origin+180 minutes. Each requires >=12 online finite power reports. Current report excluded.',
        availability='Observed changed-command report; current power and state plus command code/raw parameter. Future realized commands/state never inputs. Calendar is known.',
        population='All qualifying type1/5/8 transitions, not only the earlier pilot cases. Matched 180-minute eligible population across all horizons/models. Synchronous groups are not independent dispatch counts.',
        interpretation='Reported estimated/calibrated device power in kW; observational response forecasting, not causal treatment effects or independently metered savings.',
        source_url='https://www.bpa.gov/energy-and-services/conservation/emerging-technologies/portfolio/cta-2045-water-heaters',
        license_status='Source-public download; redistribution terms not established. This is a local prototype, not a public release.')
    (folder/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({k: receipt[k] for k in ['season','cases','eligible_devices','dates','groups_at_least10','wall_seconds']}), flush=True)
    del arrays, seen, power, online
    gc.collect()
    return receipt


def split_winter(meta):
    dates=sorted(meta.local_date.unique())
    if len(dates)<=14:
        raise ValueError('Need more than two weeks for training')
    cutoff=pd.Timestamp(dates[-14]).tz_localize('America/Los_Angeles').tz_convert('UTC')
    origin=pd.to_datetime(meta.origin_utc,utc=True)
    train=np.flatnonzero((origin+pd.Timedelta(minutes=180)<cutoff).to_numpy())
    validation=np.flatnonzero((origin>=cutoff).to_numpy())
    return train,validation,cutoff.isoformat()


def prepare_split(output):
    OUT = Path(output)
    DEST = OUT/'update_validation'
    DEST.mkdir(exist_ok=True)
    meta=pd.read_csv(OUT/'winter/cases.csv.gz',dtype={'alias':str})
    train,validation,cutoff=split_winter(meta)
    np.savez(DEST/'split.npz',train=train,validation=validation)
    origin=pd.to_datetime(meta.origin_utc,utc=True)
    info=dict(training_cases=len(train),validation_cases=len(validation),
        omitted_boundary_cases=len(meta)-len(train)-len(validation),cutoff_utc=cutoff,
        training_dates=int(meta.iloc[train].local_date.nunique()),
        validation_dates=int(meta.iloc[validation].local_date.nunique()),
        training_devices=int(meta.iloc[train].alias.nunique()),validation_devices=int(meta.iloc[validation].alias.nunique()),
        latest_training_target_end=(origin.iloc[train].max()+pd.Timedelta(minutes=180)).isoformat(),
        first_validation_origin=origin.iloc[validation].min().isoformat(),
        rule='Last14winter local dates validate. Training targets must end before the start of that block. No split chosen from prediction performance.')
    (DEST/'split.json').write_text(json.dumps(info,indent=2)+'\n')
    return info



def prepare(source, output, seasons=SEASONS, workers=1):
    """Original ZIPs -> grids/cases and original winter split; no model fits."""
    source, output = Path(source).resolve(), Path(output).resolve()
    seasons = tuple(seasons)
    if not seasons or len(set(seasons)) != len(seasons) or not set(seasons) <= set(SEASONS):
        raise ValueError('Choose unique original BPA seasons')
    if workers not in (1, 2):
        raise ValueError('Use one or two seasonal preparation workers')
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('Keep source archives and output in separate directories')
    for season in seasons:
        if not (source / f'{season}2018.zip').is_file():
            raise FileNotFoundError(f'Missing original archive: {season}2018.zip')
    output.mkdir(parents=True, exist_ok=True)
    plan = output / 'preparation.json'
    identity = dict(source=str(source), seasons=list(seasons))
    if plan.exists():
        if json.loads(plan.read_text()) != identity:
            raise ValueError('Output directory belongs to another source or season selection')
    elif any(output.iterdir()):
        raise ValueError('Use an empty preparation directory')
    else:
        plan.write_text(json.dumps(identity, indent=2) + '\n')
    started = time.monotonic()
    if workers == 1:
        receipts = [build_season(source, output, season) for season in seasons]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(seasons))) as pool:
            futures = [pool.submit(build_season, source, output, season) for season in seasons]
            receipts = [future.result() for future in futures]
    split = prepare_split(output) if 'winter' in seasons else None
    result = dict(completed=True, seasons=receipts, split=split,
                  seconds=time.monotonic() - started, workers=workers,
                  new_model_fits=0, gpu_used=False, source_directory=str(source))
    (output / 'preparation_receipt.json').write_text(json.dumps(result, indent=2) + '\n')
    return result
