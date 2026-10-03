"""Task-aligned supervised examples for the existing Chronos input mapping."""
from pathlib import Path
import numpy as np
import pandas as pd


def epfl_training_minutes(source, meta, binned_targets):
    """Recover actual minute labels for the original training cases only."""
    source = Path(source)
    if len(meta) == 0 or not meta.split.eq("train").all():
        raise ValueError("This entry point accepts only original training cases")
    ids = meta.case_id.astype(str).to_numpy()
    if len(set(ids)) != len(ids):
        raise ValueError("Training case IDs must be unique")
    support = pd.read_csv(source / "session_support.csv").set_index("key")
    linked = pd.read_csv(source / "session_id_crosswalk.csv").set_index("key_measurement")
    if not support.index.is_unique or not linked.index.is_unique:
        raise ValueError("Nonunique source session linkage")
    offsets = support.loc[ids, ["start_offset_seconds", "end_offset_seconds"]].to_numpy()
    if not np.isfinite(offsets).all() or np.any(offsets != 0):
        raise ValueError("Unverified source endpoint")
    departure = pd.to_datetime(linked.loc[ids, "Departure"]).to_numpy()
    query = pd.to_datetime(meta["query"]).to_numpy()
    times = query[:, None] + np.arange(30)[None, :] * np.timedelta64(1, "m")
    if (times >= np.datetime64("2023-01-01")).any():
        raise ValueError("Training labels cross the original temporal boundary")
    frame = pd.read_csv(source / "source_active_rows.csv.gz")
    frame["key"] = frame.CCS.astype(str) + ":" + frame.Session.astype(int).astype(str)
    frame = frame[frame.key.isin(ids)].copy()
    frame["time"] = pd.to_datetime(frame["Date and time"])
    power = frame.set_index(["key", "time"])["P (W)"]
    if not power.index.is_unique:
        raise ValueError("Duplicate session-minute records")
    index = pd.MultiIndex.from_arrays([np.repeat(ids, 30), times.reshape(-1)],
                                     names=["key", "time"])
    labels = power.reindex(index).to_numpy().reshape(-1, 30) / 1000
    inside = times <= departure[:, None]
    if not np.isfinite(labels[inside]).all():
        raise ValueError("Missing within-session training label")
    labels[~inside] = 0
    if not np.allclose(labels.reshape(-1, 6, 5).mean(axis=2), binned_targets,
                       atol=1e-5, rtol=1e-7):
        raise ValueError("Native training labels change original scored bins")
    return labels


def append_training_targets(prepared, targets):
    """Append labels and query-known covariates, never future telemetry.

    With total length L+H and min_past=L, the official random-slice sampler
    has exactly one permissible origin. No model or sampler code is changed.
    """
    import torch
    if len(prepared) != len(targets):
        raise ValueError("Training target and input case counts differ")
    output = []
    for item, target in zip(prepared, targets):
        values = torch.as_tensor(np.asarray(target), dtype=torch.float32)
        if values.ndim == 1:
            values = values[None]
        tail = item["future_covariates"].clone()
        if values.shape != (item["n_targets"], tail.shape[1]):
            raise ValueError("Native label and model forecast shapes differ")
        if not torch.isfinite(values).any() or torch.isinf(values).any():
            raise ValueError("Invalid training labels")
        # The inference preprocessor already masks targets and unknown
        # future telemetry, retaining only query-known future covariates.
        n_unknown = item["n_targets"] + item["n_covariates"] - item["n_future_covariates"]
        if not torch.isnan(tail[:n_unknown]).all():
            raise ValueError("Unexpected future target or telemetry in inputs")
        tail[:item["n_targets"]] = values
        output.append({
            **item,
            "context": torch.cat([item["context"].clone(), tail], dim=1),
            "future_covariates": torch.full_like(tail, torch.nan),
        })
    return output
