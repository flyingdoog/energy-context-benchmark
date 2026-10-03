"""Construct real-load forecasting cases at newly observed BPA commands.

Public raw archives remain unchanged. This is a bounded pilot, not causal
response estimation. Timestamp, availability and support choices are explicit.
"""
from collections import Counter
from datetime import datetime, timezone
import argparse
import gc
import json
from pathlib import Path
import time
import zipfile

import run_local
import census_bpa_public_archives  # Process-local Deflate64 support.
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'results/benchmark_source_feasibility_v1/bpa_cta2045'
OUT = ROOT / 'results/benchmark_bpa_command_pilot_v1'
SEASONS = ('winter', 'spring', 'summer')
FIELDS = {'curr_watts': ('power', np.float32, np.nan),
          'curr_curtail_id': ('command_id', np.int32, -1),
          'curr_curtail_type': ('command_type', np.int16, -1),
          'curr_curtail_numeric': ('command_numeric', np.float32, np.nan),
          'wifi_online': ('wifi', np.uint8, 255),
          'opt_out': ('opt_out', np.uint8, 255),
          'present_whr': ('present', np.float32, np.nan),
          'total_whr': ('capacity', np.float32, np.nan)}
CODE_LEVELS = (0, 1, 2, 3, 5, 7, 8, 9, 13)


def json_write(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')


from field_control.bpa_preparation import segment_mean, command_onsets
from field_control.bpa_preparation import read_season as _shared_read_season


def read_season(season, fields=None):
    """Compatibility entry retaining the historical RAW override."""
    info = json.loads((RAW / f'{season}_census.json').read_text())
    return _shared_read_season(RAW, season, fields=FIELDS if fields is None else fields, info=info)


def build_season(season, force=False):
    folder = OUT / season
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'receipt.json').exists() and not force:
        print(json.dumps({'season': season, 'completed_packet_reused': True}), flush=True)
        return json.loads((folder / 'receipt.json').read_text())
    started = time.perf_counter()
    arrays, seen, aliases, first, coverage = read_season(season)
    rows, histories, commands, updates, targets, transitions = [], [], [], [], [], []
    exclusions = Counter()
    prior_starts = -15 * np.arange(1, 9)  # Most recent 2h, strictly before command.
    lead_starts = 15 + 15 * np.arange(12)  # Common target: +15min to +195min.
    feature_names = ([f'past_15m_{k}' for k in range(1, 9)] +
        [f'previous_day_lead_{k}' for k in range(12)] +
        [f'previous_week_lead_{k}' for k in range(12)] +
        ['past_6h_mean', 'past_24h_mean', 'hour_sin', 'hour_cos',
         'weekday_sin', 'weekday_cos', 'previous_present_whr', 'previous_total_whr',
         'previous_opt_out', 'previous_command_numeric'] +
        [f'previous_command_is_{code}' for code in CODE_LEVELS])
    command_names = ([f'current_command_is_{code}' for code in CODE_LEVELS] +
                     ['current_command_numeric', 'current_opt_out'])
    update_names = ['early_mean_kw', 'early_first5_kw', 'early_last5_kw',
                    'early_std_kw', 'early_online_fraction']
    for dev, alias in enumerate(aliases):
        power = arrays['power'][dev].astype(np.float64) / 1000.0
        power[(~seen[dev]) | (arrays['wifi'][dev] != 1)] = np.nan
        onsets = command_onsets(arrays['command_id'][dev], arrays['command_type'][dev],
            arrays['command_numeric'][dev], seen[dev], arrays['wifi'][dev])
        for t in onsets:
            transitions.append((alias, int(t), int(arrays['command_id'][dev, t]),
                int(arrays['command_type'][dev, t]),
                float(arrays['command_numeric'][dev, t])))
        eligible = onsets[np.isin(arrays['command_type'][dev, onsets], [1, 5, 8])]
        exclusions['other_command_codes'] += int(len(onsets) - len(eligible))
        bounded = eligible[(eligible >= 7 * 1440) & (eligible + 195 <= len(power))]
        exclusions['insufficient_season_boundary_history_or_target'] += int(len(eligible) - len(bounded))
        if not len(bounded):
            continue
        past = segment_mean(power, bounded[:, None] + prior_starts)
        day = segment_mean(power, bounded[:, None] + lead_starts - 1440)
        week = segment_mean(power, bounded[:, None] + lead_starts - 10080)
        y = segment_mean(power, bounded[:, None] + lead_starts)
        early = power[bounded[:, None] + np.arange(15)]
        early_first = segment_mean(power, bounded, width=5, min_count=4)
        early_last = segment_mean(power, bounded + 10, width=5, min_count=4)
        six = segment_mean(power, bounded - 360, width=360, min_count=288)
        daymean = segment_mean(power, bounded - 1440, width=1440, min_count=1152)
        required = np.c_[past, day, week, y, early_first, early_last, six, daymean]
        supported = np.isfinite(required).all(axis=1) & (np.isfinite(early).sum(axis=1) >= 12)
        exclusions['power_history_target_or_early_coverage'] += int((~supported).sum())
        for j in np.flatnonzero(supported):
            t = bounded[j]
            stamp = first + pd.Timedelta(minutes=int(t))
            local = stamp.tz_convert('America/Los_Angeles')
            hour = local.hour + local.minute / 60.0
            previous_code = arrays['command_type'][dev, t - 1]
            current_code = arrays['command_type'][dev, t]
            h = np.r_[past[j], day[j], week[j], six[j], daymean[j],
                np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24),
                np.sin(2*np.pi*local.dayofweek/7), np.cos(2*np.pi*local.dayofweek/7),
                arrays['present'][dev, t-1], arrays['capacity'][dev, t-1],
                arrays['opt_out'][dev, t-1], arrays['command_numeric'][dev, t-1],
                [int(previous_code == code) for code in CODE_LEVELS]]
            c = np.r_[[int(current_code == code) for code in CODE_LEVELS],
                arrays['command_numeric'][dev, t], arrays['opt_out'][dev, t]]
            u = [np.nanmean(early[j]), early_first[j], early_last[j],
                 np.nanstd(early[j]), np.isfinite(early[j]).mean()]
            command_id = int(arrays['command_id'][dev, t])
            # Synchronous group labels are descriptive, not independent-event IDs.
            cohort = f'{stamp.isoformat()}|{command_id}|{int(current_code)}|{float(c[-2])}'
            rows.append(dict(season=season, alias=alias, origin_utc=stamp.isoformat(),
                local_date=str(local.date()), command_id=command_id,
                command_type=int(current_code), command_numeric=float(c[-2]),
                cohort_id=cohort, origin_minute=int(t)))
            histories.append(h); commands.append(c); updates.append(u); targets.append(y[j])
    frame = pd.DataFrame(rows)
    frame.to_csv(folder / 'cases.csv.gz', index=False)
    changes = pd.DataFrame(transitions, columns=['alias', 'origin_minute', 'command_id',
                                               'command_type', 'command_numeric'])
    changes.to_csv(folder / 'observed_transitions.csv.gz', index=False)
    np.savez_compressed(folder / 'packet.npz', history=np.array(histories, dtype=np.float32),
        command=np.array(commands, dtype=np.float32), update=np.array(updates, dtype=np.float32),
        target=np.array(targets, dtype=np.float32),
        history_names=np.array(feature_names), command_names=np.array(command_names),
        update_names=np.array(update_names))
    receipt = dict(season=season, **coverage, first_timestamp=str(first),
        observed_device_command_transitions=len(changes), cases=len(frame),
        eligible_devices=int(frame.alias.nunique()), dates=int(frame.local_date.nunique()),
        synchronous_cohorts=int(frame.cohort_id.nunique()),
        command_case_counts={str(k): int(v) for k,v in frame.command_type.value_counts().items()},
        exclusions=dict(exclusions), wall_seconds=time.perf_counter()-started,
        target='12 successive 15-minute mean reported kW, [+15,+195) minutes from observed command',
        availability='H strictly before first observed command; S adds current code/parameter/opt-out; S+15 adds power [0,15). No subsequent command/end is an input.',
        quality='Observed current and preceding minute online; power only wifi_online==1; >=12/15 target/history minutes; >=80% longer histories. comm_status not interpreted.',
        support='Matched device cases for all conditions. Requires previous-week/day profiles. Counts are NOT independent dispatches.',
        interpretation='Reported calibrated/estimated port power, not independently metered savings. Codes 1/5/8 pilot; all transition codes saved.')
    json_write(folder / 'receipt.json', receipt)
    print(json.dumps(receipt, indent=2), flush=True)
    del arrays, seen
    gc.collect()
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--season', choices=SEASONS)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    receipts = [build_season(s, args.force) for s in ([args.season] if args.season else SEASONS)]
    json_write(OUT / 'construction_receipt.json', dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        seasons=receipts, models_trained=0, stage='constructed_current_command_pilot'))


if __name__ == '__main__':
    main()
