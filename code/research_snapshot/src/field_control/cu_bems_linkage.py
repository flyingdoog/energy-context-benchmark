"""Thermal-record reassignment on a fixed source population, without targets."""
import numpy as np


def full_population_donors(query_minutes, device_count, *, floor, seed):
    """One random cyclic permutation per query, including all source devices."""
    queries = np.asarray(query_minutes, dtype=np.int64)
    if queries.ndim != 1 or len(np.unique(queries)) != len(queries):
        raise ValueError('Unique query minutes required')
    if device_count < 2 or (queries < 0).any():
        raise ValueError('Nonnegative clock and at least two devices required')
    mapping = np.empty((len(queries), device_count), dtype=np.int16)
    for i, minute in enumerate(queries):
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(floor), int(minute)]))
        order = rng.permutation(device_count)
        mapping[i, order] = np.roll(order, 1)
    return mapping


def thermal_bank(sensor_grid, query_minutes, *, history=60):
    """Return query x zone x past-minute x field; never read the query or future."""
    grid = np.asarray(sensor_grid)
    query = np.asarray(query_minutes, dtype=np.int64)
    if grid.ndim != 3 or grid.shape[2] != 2:
        raise ValueError('Expected minute x zone x temperature/RH')
    if (query < history).any() or (query > len(grid)).any():
        raise ValueError('History outside the recorded clock')
    ids = query[:, None] + np.arange(-history, 0)
    return grid[ids].transpose(0, 2, 1, 3).copy()


def receiver_histories(bank, query_index, donor_device, device_zone):
    """Values and their implicit finite masks move together; no label access."""
    q = np.asarray(query_index, dtype=np.int64)
    donor = np.asarray(donor_device, dtype=np.int64)
    zones = np.asarray(device_zone, dtype=np.int64)
    if q.shape != donor.shape or q.ndim != 1:
        raise ValueError('Aligned receiver queries and donors required')
    if (q < 0).any() or (q >= len(bank)).any() or (donor < 0).any() or (donor >= len(zones)).any():
        raise ValueError('Query or donor outside original population')
    return np.asarray(bank)[q, zones[donor]]


def changed_inputs(original, reassigned):
    """Distinguish a changed record identity from changed numeric/missing inputs."""
    a, b = np.asarray(original), np.asarray(reassigned)
    if a.shape != b.shape or a.ndim != 3:
        raise ValueError('Aligned thermal histories required')
    mask = (np.isfinite(a) != np.isfinite(b)).any(axis=(1, 2))
    values = (~np.isclose(a, b, rtol=1e-6, atol=1e-7, equal_nan=True)).any(axis=(1, 2))
    return values | mask, mask
