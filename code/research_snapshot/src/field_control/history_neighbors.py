"""A train-library diagnostic for state information within load neighborhoods.

This is a descriptive analog forecast, not a causal estimator or a claim that
power history contains no information. Neighbor selection never receives query
targets. All methods draw the same number of analogs from history-rank bands.
"""
import numpy as np


def rank_band_choices(state_distance, bands=8, random_seed=2027):
    """Choose one near/far/random state analog per sorted history-rank band."""
    distance = np.asarray(state_distance, dtype=float)
    if distance.ndim != 2 or distance.shape[1] % bands:
        raise ValueError('Candidate count must be divisible by the number of bands')
    if not np.isfinite(distance).all():
        raise ValueError('State distances must be finite')
    block = distance.reshape(len(distance), bands, -1)
    offsets = np.arange(bands)[None, :] * block.shape[-1]
    rng = np.random.default_rng(random_seed)
    return dict(
        near_state=offsets + block.argmin(axis=2),
        far_state=offsets + block.argmax(axis=2),
        random=offsets + rng.integers(block.shape[-1], size=(len(distance), bands)),
    )


def analog_forecasts(candidate_targets, choices):
    target = np.asarray(candidate_targets)
    if target.ndim != 3:
        raise ValueError('Expected query, candidate, lead targets')
    return {
        name: target[np.arange(len(target))[:, None], index].mean(axis=1)
        for name, index in choices.items()
    }


def caliper_band_choices(state_distance, power_distance, context_distance,
                         bands=8, seed=2027):
    """Restrict each rank band around a random, outcome-independent anchor.

    Power RMS must be within 0.02 kW or 5% of the anchor; full matching distance
    must be within 0.02 units or 5%. Anchor inclusion makes every band nonempty.
    """
    state = np.asarray(state_distance).reshape(len(state_distance), bands, -1)
    power = np.asarray(power_distance).reshape(state.shape)
    context = np.asarray(context_distance).reshape(state.shape)
    if not all(np.isfinite(a).all() for a in [state,power,context]):
        raise ValueError('Caliper inputs must be finite')
    rng = np.random.default_rng(seed)
    anchor = rng.integers(state.shape[2], size=state.shape[:2])
    p0 = np.take_along_axis(power,anchor[:,:,None],axis=2)
    c0 = np.take_along_axis(context,anchor[:,:,None],axis=2)
    allowed = ((np.abs(power-p0)<=np.maximum(.02,.05*p0)) &
               (np.abs(context-c0)<=np.maximum(.02,.05*c0)))
    offsets = np.arange(bands)[None,:]*state.shape[2]
    choices = dict(near_state=offsets+np.where(allowed,state,np.inf).argmin(axis=2),
                   far_state=offsets+np.where(allowed,state,-np.inf).argmax(axis=2))
    for draw in [2027,2028,2029]:
        u = np.random.default_rng(draw+seed).uniform(size=state.shape)
        choices['random'+str(draw)] = offsets+np.where(allowed,u,-1).argmax(axis=2)
    return choices, allowed.sum(axis=2).mean(axis=1)


def date_block_interval(dates, baseline, alternative, seed=2027, repetitions=1000):
    """Percent error reduction with a seven-calendar-day circular bootstrap."""
    dates = np.asarray(dates, dtype='datetime64[D]')
    if len(np.unique(dates)) != len(dates):
        raise ValueError('Input must already be averaged within dates')
    slots = (dates - dates.min()).astype(int)
    values = np.full((int(slots.max()) + 1, 2), np.nan)
    values[slots] = np.column_stack([baseline, alternative])
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(repetitions):
        start = rng.integers(len(values), size=int(np.ceil(len(values) / 7)))
        ix = ((start[:, None] + np.arange(7)) % len(values)).ravel()[:len(values)]
        chosen = values[ix]
        valid = np.isfinite(chosen).all(axis=1)
        if valid.any():
            b, a = chosen[valid].mean(axis=0)
            if b > 0:
                draws.append(100 * (1 - a / b))
    if not draws:
        return (float('nan'), float('nan'))
    return tuple(float(x) for x in np.quantile(draws, [.025, .975]))
