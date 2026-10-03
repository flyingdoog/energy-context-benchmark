"""Full public-data census only; raw command IDs are not independent events."""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import urllib.request
import zipfile

import run_local
import pandas as pd
import inflate64


class Deflate64Reader:
    """Adapter for the already installed decoder; ZipExtFile retains its CRC check."""
    def __init__(self):
        self.decoder = inflate64.Inflater()

    def decompress(self, data):
        return self.decoder.inflate(data)

    @property
    def eof(self):
        return self.decoder.eof


# Process-local read support for the publisher's method-9 summer ZIP.
# The Python installation and the public archive are not modified.
_standard_decompressor = zipfile._get_decompressor
def _bpa_decompressor(method):
    return Deflate64Reader() if method == 9 else _standard_decompressor(method)
zipfile._get_decompressor = _bpa_decompressor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/benchmark_source_feasibility_v1/bpa_cta2045'
BASE = 'https://www.bpa.gov/-/media/Aep/energy-efficiency/emerging-technologies/'
SEASONS = ('winter', 'spring', 'summer')
USE = ['local_time', 'alias', 'curr_curtail_id', 'curr_curtail_type', 'curr_curtail_numeric',
       'curr_watts', 'opt_out', 'groupA', 'groupB', 'comm_status', 'wifi_online']


def download(season):
    path = OUT / f'{season}2018.zip'
    if path.exists():
        return path
    request = urllib.request.Request(BASE + path.name, headers={'User-Agent': 'ResearchMetadataCensus/1.0'})
    temporary = path.with_suffix('.partial')
    with urllib.request.urlopen(request, timeout=45) as response, temporary.open('wb') as stream:
        total = 0
        while part := response.read(4_000_000):
            total += len(part)
            if total > 500_000_000:
                raise ValueError('Public archive exceeds 500 MB bound')
            stream.write(part)
    # Archive integrity, not a copy-identity audit.
    if not zipfile.is_zipfile(temporary):
        raise ValueError('Response is not a complete ZIP archive')
    temporary.rename(path)
    return path


def census(season):
    saved = OUT / f'{season}_census.json'
    if saved.exists() and (OUT / f'{season}2018.zip').exists():
        print(json.dumps({'season':season,'existing_completed_census':True}), flush=True)
        return json.loads(saved.read_text())
    path = download(season)
    devices = set()
    counters = {f: Counter() for f in ['curr_curtail_type','opt_out','groupA','groupB','comm_status','wifi_online']}
    codes = {}
    dates = Counter()
    rows = missing_power = negative_power = 0
    first = last = None
    with zipfile.ZipFile(path) as archive:
        members = archive.namelist()
        csv_name, = [name for name in members if name.lower().endswith('.csv')]
        with archive.open(csv_name) as stream:
            for frame in pd.read_csv(stream, usecols=USE, chunksize=250_000):
                rows += len(frame)
                devices.update(frame.alias.unique().tolist())
                stamp = frame.local_time.astype(str)
                lo, hi = stamp.min(), stamp.max()
                first = min(first, lo) if first else lo
                last = max(last, hi) if last else hi
                dates.update(stamp.str[:10].value_counts().to_dict())
                missing_power += int(frame.curr_watts.isna().sum())
                negative_power += int((frame.curr_watts < 0).sum())
                for name in counters:
                    counters[name].update({str(k): int(v) for k,v in frame[name].value_counts(dropna=False).items()})
                # A code can persist for many hours/days; retain time support without treating it as an event.
                grouped = frame.groupby(['curr_curtail_id','curr_curtail_type'], dropna=False).agg(
                    first_seen=('local_time','min'), last_seen=('local_time','max'), rows=('alias','size'))
                for key, value in grouped.iterrows():
                    key = tuple(str(k) for k in key)
                    existing = codes.setdefault(key, {'first_seen':value.first_seen,'last_seen':value.last_seen,'rows':0})
                    existing['first_seen'] = min(existing['first_seen'], value.first_seen)
                    existing['last_seen'] = max(existing['last_seen'], value.last_seen)
                    existing['rows'] += int(value.rows)
                if rows % 2_000_000 == 0:
                    print(json.dumps({'season':season,'rows_scanned':rows}), flush=True)
    code_rows = [dict(command_id=key[0], type_code=key[1], **value) for key,value in codes.items()]
    pd.DataFrame(code_rows).to_csv(OUT / f'{season}_raw_command_support.csv', index=False)
    pd.DataFrame(sorted(dates.items()), columns=['timestamp_date_literal','rows']).to_csv(OUT / f'{season}_raw_date_support.csv', index=False)
    receipt = dict(season=season, rows=rows, devices=len(devices), date_literals=len(dates),
        first_timestamp=first,last_timestamp=last, command_id_type_pairs=len(codes),
        missing_power=missing_power,negative_power=negative_power, raw_value_counts=counters,
        archive_bytes=path.stat().st_size, members=members,
        time_semantics='Raw local_time strings retained. Z suffix not yet independently reconciled with dispatch-local clock.',
        scope='Raw telemetry coverage, not event counts, effective sample size, or verified field-code semantics.')
    (OUT / f'{season}_census.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['raw_value_counts']}, indent=2), flush=True)
    return receipt


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results, errors = [], []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {season: pool.submit(census, season) for season in SEASONS}
        for season, future in futures.items():
            try:
                results.append(future.result())
            except Exception as exc:
                errors.append({'season':season,'error':str(exc)})
    (OUT / 'census.json').write_text(json.dumps({'completed_utc':datetime.now(timezone.utc).isoformat(),
        'sources':results,'errors':errors,'all_seasons_complete':len(results)==len(SEASONS),
        'models_trained':0,'event_segmentation_complete':False}, indent=2) + '\n')
    print(json.dumps({'completed_seasons':[r['season'] for r in results],'errors':errors}), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
