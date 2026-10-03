"""Matched feature and training-population helpers for the new-floor pilot."""
import numpy as np
import pandas as pd


def transfer_features(packet, condition, indices=None):
    if condition not in ('power', 'thermal'):
        raise ValueError('Use power or thermal')
    ix = np.arange(len(packet['meta'])) if indices is None else np.asarray(indices)
    pieces = [np.asarray(packet['power'][ix], dtype=np.float32),
              np.asarray(packet['calendar'][ix], dtype=np.float32)]
    if condition == 'thermal':
        if packet.get('thermal') is None:
            raise ValueError('Transfer thermal arm requires actual sensor fields')
        thermal = np.asarray(packet['thermal'][ix], dtype=np.float32)
        pieces += [np.nan_to_num(thermal, nan=0, posinf=0, neginf=0).reshape(len(ix), -1),
                   np.isfinite(thermal).reshape(len(ix), -1).astype(np.float32)]
    result = np.concatenate(pieces, axis=1)
    if result.shape != (len(ix), 304 if condition == 'thermal' else 64) or not np.isfinite(result).all():
        raise ValueError('Expected finite history/calendar and optional masked thermal features')
    return result


def training_block(packet):
    meta = packet['meta']
    ix = np.flatnonzero(meta.split.to_numpy() == 'train')
    if not len(ix):
        raise ValueError('No source training cases')
    query = pd.to_datetime(meta['query'].iloc[ix])
    if ((query < pd.Timestamp('2019-04-01')).any() or
            (query + pd.Timedelta(minutes=150) > pd.Timestamp('2019-07-01')).any()):
        raise ValueError('Training targets must stay inside the prescribed April-June block')
    return dict(x=transfer_features(packet, 'thermal', ix),
                y=np.asarray(packet['target'][ix], dtype=np.float32),
                cases=len(ix), dates=int(meta.date.iloc[ix].nunique()),
                first_query=str(query.min()), last_query=str(query.max()))


def training_floors(floors, target, protocol):
    if target not in floors or protocol not in ('local', 'other'):
        raise ValueError('Unknown floor or training protocol')
    selected = [target] if protocol == 'local' else [f for f in floors if f != target]
    if not selected:
        raise ValueError('At least one other training floor is required')
    return selected


def training_arrays(cache, target, protocol, condition):
    selected = training_floors(sorted(cache), target, protocol)
    if condition not in ('power', 'thermal'):
        raise ValueError('Unknown input condition')
    width = 64 if condition == 'power' else 304
    x = np.concatenate([cache[f]['x'][:, :width] for f in selected])
    y = np.concatenate([cache[f]['y'] for f in selected])
    if y.shape != (len(x), 10) or not np.isfinite(y).all():
        raise ValueError('Finite ten-bin training targets required')
    return x, y, selected


def context_interaction(local_power, local_thermal, other_power, other_thermal, weights):
    """Difference in thermal-input effects; negative favors context under transfer."""
    a, b, c, d = [np.asarray(x, dtype=float) for x in
                  (local_power, local_thermal, other_power, other_thermal)]
    if (any(x.shape != a.shape or not np.isfinite(x).all() for x in (b,c,d))
            or not np.isfinite(a).all() or weights.shape[1] != len(a)):
        raise ValueError('Aligned paired calendar dates required')
    effect = (d-c)-(b-a)
    low, high = np.quantile(weights @ effect, [.025,.975])
    result = dict(difference=float(effect.mean()), low=float(low), high=float(high))
    wa,wb,wc,wd = [weights @ x for x in (a,b,c,d)]
    if a.mean()>0 and c.mean()>0 and (wa>0).all() and (wc>0).all():
        low,high=np.quantile(100*(wd/wc-wb/wa),[.025,.975])
        result.update(change_effect_pp=float(100*(d.mean()/c.mean()-b.mean()/a.mean())),
                      change_effect_low=float(low),change_effect_high=float(high))
    return result
