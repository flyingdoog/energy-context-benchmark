"""Pinned ACN source acquisition and the original calendar manifest.

Planning reads provider indexes, never session outcomes. A sample download is
an acquisition check, not a smaller benchmark. Preparation requires all files.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime as dt
import json
from pathlib import Path
import re
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen
import pandas as pd

REPOSITORY = 'tongxin-li/ACN-Data-Static'
REVISION = 'abb7cf15cc7108913e965d4375cc7270d532b65f'
SOURCE_URL = f'https://github.com/{REPOSITORY}'
RAW_BASE = f'https://raw.githubusercontent.com/{REPOSITORY}/{REVISION}/'
BLOCKS = [('train', '2019-04-01', '2019-06-30'),
          ('validation', '2019-07-01', '2019-07-31'),
          ('autumn', '2019-10-01', '2019-10-31'),
          ('winter', '2020-01-01', '2020-01-31')]
SITES = [('caltech', 'caltech', 'California_Garage_01'),
         ('office001', 'office_01', 'Parking_Lot_01')]
LEAD_IN_DAYS = 14
FILE_LIMIT = 4 * 1024 ** 2
COLLECTION_LIMIT = 512 * 1024 ** 2
PATTERN = re.compile(r'^(\d+-\d+-\d+-\d+)-(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}-\d+)\.csv\.gz$')


def _write_json(path, value):
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def _get(url, limit):
    """Two attempts only for transient failures; no retry of access/rate errors."""
    for attempt in range(2):
        try:
            request = Request(url, headers={'User-Agent': 'OperationalForecastBenchmark/0.1'})
            with urlopen(request, timeout=30) as response:
                if response.status != 200:
                    raise ValueError(f'Unexpected source response: {response.status}')
                content = response.read(limit + 1)
            if not content or len(content) > limit:
                raise ValueError('Empty source response or download size limit exceeded')
            return content
        except HTTPError as error:
            if error.code not in (502, 503, 504) or attempt:
                raise
        except (TimeoutError, URLError):
            if attempt:
                raise
        time.sleep(1)


def _workspace(output):
    output = Path(output).resolve()
    identity = dict(source_url=SOURCE_URL, revision=REVISION,
                    blocks=BLOCKS, lead_in_days=LEAD_IN_DAYS)
    marker = output / 'source_plan.json'
    if marker.exists():
        if json.loads(marker.read_text()) != json.loads(json.dumps(identity)):
            raise ValueError('Acquisition directory belongs to another source or calendar')
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError('Use an empty acquisition directory, not an existing data collection')
        output.mkdir(parents=True, exist_ok=True)
        _write_json(marker, identity)
    (output / 'metadata').mkdir(exist_ok=True)
    (output / 'sessions').mkdir(exist_ok=True)
    return output


def _tree(output, reference, name, index_source=None):
    path = output / 'metadata' / f'{name}.json'
    if path.exists():
        tree = json.loads(path.read_text())
    else:
        supplied = Path(index_source) / f'{name}.json' if index_source is not None else None
        if supplied is not None and supplied.is_file():
            tree = json.loads(supplied.read_text())
        else:
            url = f'https://api.github.com/repos/{REPOSITORY}/git/trees/{reference}'
            tree = json.loads(_get(url, 16 * 1024 ** 2))
        if tree.get('truncated') or tree.get('sha') != reference:
            raise ValueError('Provider index is truncated or has an unexpected revision')
        _write_json(path, tree)
        print(json.dumps(dict(index=name, entries=len(tree['tree']))), flush=True)
    if tree.get('truncated') or tree.get('sha') != reference:
        raise ValueError('Cached provider index is incomplete or from another revision')
    return tree


def _child(tree, name):
    found = [row['sha'] for row in tree['tree']
             if row.get('type') == 'tree' and row.get('path') == name]
    if len(found) != 1:
        raise ValueError(f'Provider directory not uniquely found: {name}')
    return found[0]


def calendar_rows(trees):
    """Original two-site month blocks, including connections in each lead-in."""
    rows = []
    for site, site_dir, location in SITES:
        for item in trees[site]['tree']:
            filename = item['path']
            if not filename.endswith('.csv.gz'):
                continue
            match = PATTERN.fullmatch(filename)
            if item.get('type') != 'blob' or not match:
                raise ValueError('Unrecognized source session entry')
            size = item.get('size')
            if not isinstance(size, int) or not 2 <= size <= FILE_LIMIT:
                raise ValueError('Source session size is missing or exceeds the file limit')
            stamp = pd.Timestamp(dt.datetime.strptime(match[2], '%Y-%m-%dT%H-%M-%S-%f')
                                 .replace(tzinfo=dt.timezone.utc))
            day = str(stamp.tz_convert('America/Los_Angeles').date())
            for split, first, last in BLOCKS:
                lower = str((pd.Timestamp(first) - pd.Timedelta(days=LEAD_IN_DAYS)).date())
                if lower <= day <= last:
                    rows.append(dict(site=site, split=split, local_date=day,
                        station_id=match[1], connection_utc=str(stamp), filename=filename,
                        source_path=f'time series data/{site_dir}/{location}/{filename}',
                        source_bytes=size, query_start_local=first,
                        query_end_exclusive_local=str((pd.Timestamp(last) + pd.Timedelta(days=1)).date())))
    return sorted(rows, key=lambda row: (row['site'], row['connection_utc'], row['filename']))


def unique_files(rows):
    files = {}
    for row in rows:
        previous = files.get(row['filename'])
        if previous and (previous['source_path'], previous['source_bytes']) != (row['source_path'], row['source_bytes']):
            raise ValueError('Conflicting source paths for a session filename')
        files[row['filename']] = row
    return list(files.values())


def sample_files(rows):
    """First and last connection per site/block, deduplicated without reading labels."""
    selected = []
    for site, _, _ in SITES:
        for split, _, _ in BLOCKS:
            subset = sorted((row for row in rows if row['site'] == site and row['split'] == split),
                            key=lambda row: (row['connection_utc'], row['filename']))
            if subset:
                selected.extend((subset[0], subset[-1]))
    return unique_files(selected)


def build_plan(output, index_source=None):
    """Recover all calendar contexts from six pinned indexes, without sessions."""
    output = _workspace(output)
    if index_source is not None and not Path(index_source).is_dir():
        raise ValueError('The supplied provider-index directory does not exist')
    started = time.monotonic()
    root = _tree(output, REVISION, 'root', index_source)
    series = _tree(output, _child(root, 'time series data'), 'time_series', index_source)
    trees = {}
    for site, site_dir, location in SITES:
        directory = _tree(output, _child(series, site_dir), site_dir, index_source)
        trees[site] = _tree(output, _child(directory, location), f'{site_dir}_{location}', index_source)
    notice = output / 'SOURCE_README.md'
    if not notice.exists():
        content = _get(RAW_BASE + 'README.md', 1024 ** 2)
        temporary = notice.with_name(notice.name + '.partial')
        temporary.write_bytes(content)
        temporary.replace(notice)
    rows = calendar_rows(trees)
    files = unique_files(rows)
    total = sum(row['source_bytes'] for row in files)
    if not files or len(files) > 10000 or total > COLLECTION_LIMIT:
        raise ValueError('Full source collection exceeds its bound or is empty; do not silently subset')
    manifest = pd.DataFrame(rows)
    manifest_path = output / 'manifest.csv'
    if manifest_path.exists():
        pd.testing.assert_frame_equal(pd.read_csv(manifest_path), manifest, check_dtype=False)
    else:
        manifest.to_csv(manifest_path, index=False)
    record = dict(plan_completed=True, source_url=SOURCE_URL, revision=REVISION,
        manifest_contexts=len(rows), planned_files=len(files), listed_bytes=total,
        lead_in_days=LEAD_IN_DAYS, blocks=BLOCKS, session_downloads=0,
        optional_index_directory=str(Path(index_source).resolve()) if index_source is not None else None,
        seconds=round(time.monotonic() - started, 3))
    _write_json(output / 'plan_receipt.json', record)
    if not (output / 'receipt.json').exists():
        _write_json(output / 'receipt.json', dict(completed=False, phase='planned',
            planned=len(files), manifest_contexts=len(rows), revision=REVISION))
    return record


def _available(output, row):
    path = output / 'sessions' / row['filename']
    if not path.is_file() or path.stat().st_size != row['source_bytes']:
        return False
    with path.open('rb') as stream:
        return stream.read(2) == b'\x1f\x8b'


def acquire(output, sample=False, workers=4, index_source=None):
    """Download the full planned collection or only a bounded acquisition sample."""
    if not 1 <= workers <= 4:
        raise ValueError('Use one to four acquisition workers')
    build_plan(output, index_source=index_source)
    output = Path(output).resolve()
    rows = pd.read_csv(output / 'manifest.csv').to_dict('records')
    files = unique_files(rows)
    selected = sample_files(rows) if sample else files
    started = time.monotonic()
    stop = threading.Event()

    def fetch(row):
        name = row['filename']
        result = dict(filename=name, source_path=row['source_path'])
        if stop.is_set():
            return dict(result, status='stopped', bytes=0)
        target = output / 'sessions' / name
        try:
            if _available(output, row):
                return dict(result, status='cached', bytes=0)
            if target.exists():
                raise ValueError('Existing source file is incomplete or inconsistent; inspect it before reuse')
            content = _get(RAW_BASE + quote(row['source_path'], safe='/'), row['source_bytes'])
            if len(content) != row['source_bytes'] or content[:2] != b'\x1f\x8b':
                raise ValueError('Source download has unexpected size or is not gzip')
            temporary = target.with_name(target.name + '.partial')
            temporary.write_bytes(content)
            temporary.replace(target)
            return dict(result, status='downloaded', bytes=len(content))
        except (HTTPError, URLError, OSError, ValueError) as error:
            stop.set()
            return dict(result, status='error', bytes=0, error=str(error))

    outcomes = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(fetch, row) for row in selected]
        for job in as_completed(jobs):
            outcomes.append(job.result())
            if len(outcomes) % 25 == 0 or len(outcomes) == len(selected):
                progress = dict(processed=len(outcomes), selected=len(selected),
                    downloaded=sum(row['status'] == 'downloaded' for row in outcomes),
                    errors=sum(row['status'] == 'error' for row in outcomes),
                    seconds=round(time.monotonic() - started, 3))
                _write_json(output / 'progress.json', progress)
                print(json.dumps(progress), flush=True)
    pd.DataFrame(outcomes).sort_values('filename').to_csv(output / 'acquisition.csv', index=False)
    available = sum(_available(output, row) for row in files)
    selected_completed = all(row['status'] in ('cached', 'downloaded') for row in outcomes)
    record = dict(completed=available == len(files), terminal=True,
        phase='sample' if sample else 'full', selected_completed=selected_completed,
        selected=len(selected), planned=len(files), available_files=available,
        manifest_contexts=len(rows), downloaded=sum(row['status'] == 'downloaded' for row in outcomes),
        errors=sum(row['status'] == 'error' for row in outcomes), stopped=stop.is_set(),
        transferred_bytes=sum(row['bytes'] for row in outcomes), source_url=SOURCE_URL,
        revision=REVISION, workers=workers, seconds=round(time.monotonic() - started, 3),
        note='A partial download is not a benchmark subset and cannot be prepared for evaluation.')
    _write_json(output / 'receipt.json', record)
    return record
