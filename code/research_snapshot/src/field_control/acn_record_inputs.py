"""Recover only ACN query-known input rows absent from the labeled packet."""
from pathlib import Path
import time
import numpy as np
import pandas as pd
from .acn import read_static, minute_grid, history_features


def recover_query_inputs(population, known_meta, known_x, source, progress=None):
    source = Path(source)
    pop = population[['case_id','site','split','query_utc','station_id']].copy().reset_index(drop=True)
    if not pop.case_id.is_unique or not known_meta.case_id.is_unique:
        raise ValueError('Unique query IDs required')
    known = pd.Index(known_meta.case_id).get_indexer(pop.case_id)
    missing = np.flatnonzero(known < 0)
    x = np.empty((len(pop), known_x.shape[1]), dtype=np.float32)
    supported = known >= 0
    x[supported] = known_x[known[supported]]
    pop['filename'] = pop.case_id.str.rsplit('@', n=1).str[0]
    manifest = pd.read_csv(source/'manifest.csv')
    lookup = manifest.set_index(['site','split','filename'])
    if not lookup.index.is_unique:
        raise ValueError('Ambiguous source context')
    started = time.monotonic()
    files = 0
    recovered = 0
    for filename, positions in pop.iloc[missing].groupby('filename', sort=True).groups.items():
        if '/' in filename or '\\' in filename or not filename.endswith('.csv.gz'):
            raise ValueError('Invalid source filename')
        frame = read_static(source/'sessions'/filename)
        minutes, _, _ = minute_grid(frame)
        for pos in positions:
            row = pop.iloc[pos]
            context = lookup.loc[(row.site,row.split,filename)]
            feature = history_features(minutes, pd.Timestamp(row.query_utc),
                                       pd.Timestamp(context.connection_utc))
            vector = np.concatenate([feature[name] for name in ['power','control','state','calendar']]).astype(np.float32)
            if vector.shape != (x.shape[1],) or not np.isfinite(vector).all():
                raise ValueError('Invalid recovered query-only feature vector')
            x[pos] = vector
        files += 1
        recovered += len(positions)
        if progress and files % 100 == 0:
            progress(dict(stage='recover_acn_inputs', files=files,
                          recovered_cases=recovered,
                          seconds=time.monotonic()-started))
    if not np.isfinite(x).all():
        raise ValueError('Missing recovered inputs')
    return x, dict(query_known_cases=len(pop), existing_input_rows=int(supported.sum()),
                   recovered_input_rows=len(missing), source_files_read=files,
                   seconds=time.monotonic()-started)
