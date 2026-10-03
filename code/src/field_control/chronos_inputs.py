"""Input-only mapping of native operational records to Chronos-2 items.

No model import, fitting or target access. Query-only quantities occupy one
final context slot; this is a side-channel packing convention, not a claim
that a whole history of that quantity was observed.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ForecastGrid:
    source: str
    context_length: int
    prediction_length: int
    output_bins: int
    minutes_per_output_bin: int
    model_steps_per_output_bin: int

    def aggregate(self, point):
        """Average point trajectories only; marginal quantiles do not aggregate."""
        values = np.asarray(point, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.prediction_length:
            raise ValueError("Point trajectory and source grid differ")
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite point forecast")
        return values.reshape(len(values), self.output_bins,
                              self.model_steps_per_output_bin).mean(axis=2)


def _numeric(value, observed):
    value = np.asarray(value, dtype=np.float32).copy()
    value[~np.asarray(observed, dtype=bool)] = np.nan
    return value


def _category(value, observed):
    value = np.asarray(value)
    observed = np.asarray(observed, dtype=bool)
    out = np.empty(value.shape, dtype=object)
    out[:] = np.nan
    for ix in np.ndindex(value.shape):
        if observed[ix] and pd.notna(value[ix]):
            out[ix] = "code_" + str(int(value[ix]))
    return out


def _query_slot(value, length, categorical=False):
    out = np.empty(length, dtype=object if categorical else np.float32)
    out[:] = np.nan
    out[-1] = value
    return out


def _calendar(query, offsets, source):
    query = pd.Timestamp(query)
    if source == "ACN":
        query = query.tz_convert("America/Los_Angeles")
    stamps = query + pd.to_timedelta(np.asarray(offsets), unit="min")
    hour = stamps.hour.to_numpy() + stamps.minute.to_numpy() / 60
    days = stamps.dayofweek.to_numpy()
    result = {"hour_sin": np.sin(2 * np.pi * hour / 24),
              "hour_cos": np.cos(2 * np.pi * hour / 24)}
    if source == "EPFL":
        result.update(weekday=days, month=stamps.month.to_numpy())
    else:
        result.update(weekday_sin=np.sin(2*np.pi*days/7),
                      weekday_cos=np.cos(2*np.pi*days/7))
    return {k: np.asarray(v, dtype=np.float32) for k, v in result.items()}


def make_chronos_items(view, *, query_times=None):
    """Map a shared input view; all arrays are copies, never label-derived.

    Non-BPA query_times are native metadata timestamps. ACN is timezone-aware;
    EPFL/CU retain their source clocks. There is no future operational telemetry.
    """
    source = view["contract"]["source"]
    p = view["native_inputs"]
    count = len(view["case_id"])
    if source == "BPA":
        length = p["past_power_kw"].shape[1]
        grid = ForecastGrid(source, length, 10, 10, 15, 1)
    else:
        length = p["power_kw"].shape[1]
        width, minutes = (6, 5) if source == "EPFL" else (10, 15)
        grid = ForecastGrid(source, length, width*minutes, width, minutes, minutes)
        if query_times is None or len(query_times) != count:
            raise ValueError("Native query timestamps required")
    items = []
    for i in range(count):
        past, future = {}, {}
        if source == "BPA":
            fraction = p["past_power_observed_fraction"][i]
            target = _numeric(p["past_power_kw"][i], fraction > 0)
            past["power_observed_fraction"] = np.asarray(fraction, dtype=np.float32).copy()
            # These scalar reports occur at query, not throughout the past.
            past["query_power_kw"] = _query_slot(
                p["query_power_kw"][i] if p["query_power_observed"][i] else np.nan, length)
            past["query_power_observed"] = _query_slot(float(p["query_power_observed"][i]), length)
            for j, name in enumerate(("hour_sin", "hour_cos", "weekday_sin", "weekday_cos")):
                past[name] = p["past_calendar"][i, :, j].astype(np.float32).copy()
                future[name] = p["future_calendar"][i, :, j].astype(np.float32).copy()
            # Native day/week profiles are already known at query. They are
            # lead-aligned past measurements, not future realized consumption.
            for label in ("day", "week"):
                values = p[f"previous_{label}_profile_kw"][i]
                masks = p[f"previous_{label}_profile_observed_fraction"][i]
                name = f"previous_{label}_profile_kw"
                past[name] = np.full(length, np.nan, dtype=np.float32)
                future[name] = _numeric(values[:10], masks[:10] > 0)
                past[name+"_observed_fraction"] = np.full(length, np.nan, dtype=np.float32)
                future[name+"_observed_fraction"] = masks[:10].astype(np.float32).copy()
                # Preserve the two supplied extra profile bins without
                # misrepresenting them as the ten prediction-time covariates.
                for j in range(10, len(values)):
                    past[f"{name}_extra_{j}"] = _query_slot(values[j] if masks[j] > 0 else np.nan, length)
                    past[f"{name}_extra_{j}_observed_fraction"] = _query_slot(masks[j], length)
            if "past_state" in p:
                names = ("present", "capacity", "oper_state", "running", "opt_out")
                for j, name in enumerate(names):
                    categorical = j >= 2
                    convert = _category if categorical else _numeric
                    past[name] = convert(p["past_state"][i, :, j], p["past_state_observed"][i, :, j])
                    past[name+"_observed"] = p["past_state_observed"][i, :, j].astype(np.float32)
                    q = convert(p["current_state"][i, j+1:j+2], p["current_state_observed"][i, j+1:j+2])[0]
                    past["query_"+name] = _query_slot(q, length, categorical)
                    past["query_"+name+"_observed"] = _query_slot(float(p["current_state_observed"][i, j+1]), length)
            if "past_command_type" in p:
                masks = p["past_command_observed"][i]
                past["command_type"] = _category(p["past_command_type"][i], masks)
                past["command_parameter"] = _numeric(p["past_command_parameter"][i], masks)
                past["command_observed"] = masks.astype(np.float32)
                qmask = bool(p["current_command_observed"][i])
                qtype = _category(p["current_command"][i, :1], [qmask])[0]
                past["query_command_type"] = _query_slot(qtype, length, True)
                past["query_command_parameter"] = _query_slot(p["current_command"][i, 1] if qmask else np.nan, length)
                past["query_command_observed"] = _query_slot(float(qmask), length)
        else:
            target = _numeric(p["power_kw"][i, :, 0], p["power_observed"][i, :, 0])
            past["power_observed"] = p["power_observed"][i, :, 0].astype(np.float32)
            past.update(_calendar(query_times[i], np.arange(-length, 0), source))
            future.update(_calendar(query_times[i], np.arange(grid.prediction_length), source))
            if source == "ACN":
                # Connected-at is known at query; elapsed time is a deterministic
                # clock, not a claim that the device remains connected.
                age = float(p["calendar"][i, -1])
                past["hours_since_connection"] = (age + np.arange(-length, 0)/60).astype(np.float32)
                future["hours_since_connection"] = (age + np.arange(grid.prediction_length)/60).astype(np.float32)
            if "numeric_state" in p:
                for j, name in enumerate(p["numeric_state_names"]):
                    past[name] = _numeric(p["numeric_state"][i, :, j], p["numeric_state_observed"][i, :, j])
                    past[name+"_observed"] = p["numeric_state_observed"][i, :, j].astype(np.float32)
            if "operating_state_fractions" in p:
                for j, name in enumerate(p["operating_state_names"]):
                    past["state_fraction_"+name] = _numeric(p["operating_state_fractions"][i, :, j],
                                                           p["operating_state_observed"][i, :, j])
                    past["state_fraction_"+name+"_observed"] = p["operating_state_observed"][i, :, j].astype(np.float32)
            for name, mask in (("pilot_a", "pilot_observed"), ("requested_kw", "requested_observed"),
                               ("setpoint_kw", "setpoint_observed"), ("controlled", "controlled_observed")):
                if name in p:
                    past[name] = _numeric(p[name][i, :, 0], p[mask][i, :, 0])
                    past[name+"_observed"] = p[mask][i, :, 0].astype(np.float32)
            # Fixed identity is constant metadata, not a measured state history.
            if source == "CU-BEMS":
                for name, value in (("device", int(p["device_index"][i])),
                                    ("zone", p["zone"][i]), ("floor", p["floor"])):
                    past[name] = np.full(length, str(value), dtype=object)
                    future[name] = np.full(grid.prediction_length, str(value), dtype=object)
        if target.shape != (length,):
            raise ValueError("Expected one target power trajectory")
        if any(np.shape(x) != (length,) for x in past.values()):
            raise ValueError("Past covariate grid differs")
        if any(np.shape(x) != (grid.prediction_length,) for x in future.values()):
            raise ValueError("Future covariate grid differs")
        items.append({"target": target, "past_covariates": past, "future_covariates": future})
    return items, grid
