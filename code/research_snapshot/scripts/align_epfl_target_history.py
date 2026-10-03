"""Align the EPFL requested target's own-history inputs; reuse the original pilot.

Run with the project's bundled Python. No source workbook reads, no HPO.
"""
import json
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor
import run_local
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
import pilot_epfl_operational_forecasting as pilot

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/epfl_operational_pilot_v1"
OUT = ROOT / "results/epfl_target_history_alignment_v1"
METRICS = ["mse", "mae", "energy_ae_kwh"]


def feature_schema():
    names = [f"actual_kw@{t}" for t in range(10)]
    names += ["hour_sin", "hour_cos", "weekday", "month"]
    names += [f"{field}@{t}" for t in range(10) for field in pilot.STATE]
    names += [f"{field}@{t}" for t in range(10) for field in pilot.CONTROL]
    return names


def feature_columns():
    names = feature_schema()
    own = [i for i, name in enumerate(names) if name.startswith("Preq (W)@")]
    return {
        "own_history": np.array(sorted(set(range(14)) | set(own))),
        "own_history_state": np.array(sorted(set(range(64)) | set(own))),
        "full": np.arange(len(names)),
    }


def aligned_prediction(path, case_ids):
    with np.load(path, allow_pickle=False) as archive:
        if not np.array_equal(archive["case_id"].astype(str), case_ids):
            raise ValueError(f"Prediction case alignment differs: {path.name}")
        value = archive["prediction_kw"]
    if value.shape != (len(case_ids), 6, 2) or not np.isfinite(value).all():
        raise ValueError(f"Invalid prediction shape or values: {path.name}")
    return value


def paired_contrast(base, candidate, block_length=7, repetitions=2000):
    """Positive differences mean smaller error; paired circular observed-day blocks."""
    base = np.asarray(base, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    if base.ndim != 1 or base.shape != candidate.shape or len(base) == 0:
        raise ValueError("Paired nonempty daily vectors required")
    if not (np.isfinite(base).all() and np.isfinite(candidate).all()):
        raise ValueError("Daily scores must be finite")
    n = len(base)
    block_length = min(block_length, n)
    rng = np.random.default_rng(20270907)
    starts = rng.integers(n, size=(repetitions, int(np.ceil(n / block_length))))
    indices = ((starts[:, :, None] + np.arange(block_length)) % n).reshape(repetitions, -1)[:, :n]
    differences = base - candidate
    draws = differences[indices].mean(axis=1)
    return dict(
        days=n,
        baseline_mean=float(base.mean()),
        candidate_mean=float(candidate.mean()),
        improvement=float(differences.mean()),
        improvement_percent=float(100 * differences.mean() / base.mean()) if base.mean() else None,
        ci95_low=float(np.quantile(draws, .025)),
        ci95_high=float(np.quantile(draws, .975)),
        observed_day_block_length=block_length,
        bootstrap_repetitions=repetitions,
    )


def build_contrasts(daily):
    rows = []
    pairs = [
        ("power", "own_history"),
        ("own_history", "own_history_state"),
        ("own_history_state", "full"),
        ("own_history", "full"),
        ("power_state", "own_history_state"),
    ]
    for split in ["validation", "later"]:
        part = daily[(daily.split == split) & (daily.target == "requested_power")]
        for span in [5, 15, 30, "average"]:
            view = part if span == "average" else part[part.span_minutes == span]
            for metric in METRICS:
                scores = view.groupby(["date", "method"])[metric].mean().unstack("method").sort_index()
                for before, after in pairs:
                    if scores[[before, after]].isna().any().any():
                        raise ValueError("Methods must cover exactly the same dates")
                    rows.append(dict(split=split, span_minutes=str(span), metric=metric,
                                     baseline=before, candidate=after,
                                     **paired_contrast(scores[before], scores[after])))
    return pd.DataFrame(rows)


def outputs(daily):
    summary = daily.groupby(["split", "target", "span_minutes", "method"])[METRICS].mean().reset_index()
    summary.to_csv(OUT / "summary.csv", index=False)
    averages = summary.groupby(["split", "target", "method"])[METRICS].mean().reset_index()
    averages.to_csv(OUT / "span_average.csv", index=False)
    contrasts = build_contrasts(daily)
    contrasts.to_csv(OUT / "paired_contrasts.csv", index=False)
    return summary, averages, contrasts


def main():
    start = time.monotonic()
    OUT.mkdir(exist_ok=True)
    receipt = json.loads((BASE / "receipt.json").read_text())
    if not receipt["completed"]:
        raise ValueError("Original pilot is incomplete")
    meta = pd.read_csv(BASE / "cases.csv")
    case_ids = meta.case_id.to_numpy(dtype=str)
    with np.load(BASE / "packet.npz", allow_pickle=False) as archive:
        if not np.array_equal(archive["case_id"].astype(str), case_ids):
            raise ValueError("Packet/case alignment differs")
        x, y = archive["x"], archive["y"]
    if x.shape != (len(meta), len(feature_schema())) or y.shape != (len(meta), 6, 2):
        raise ValueError("This runner expects the documented v1 ten-minute feature layout")
    train = meta.split.to_numpy() == "train"
    columns = feature_columns()
    old = {name: aligned_prediction(BASE / f"{name}.npz", case_ids)
           for name in ["power", "power_state", "full"]}
    model_params = {entry["condition"]: entry["parameters"] for entry in receipt["models"]}
    default = LGBMRegressor(n_jobs=2, random_state=2027, verbosity=-1)
    if default.get_params() != model_params["full"]:
        raise ValueError("Model recipe differs from the saved original pilot")

    def fit(condition):
        prediction_path = OUT / f"{condition}.npz"
        info_path = OUT / f"{condition}_receipt.json"
        if prediction_path.exists() and info_path.exists():
            info = json.loads(info_path.read_text())
            if info["columns"] != columns[condition].tolist() or info["parameters"] != default.get_params():
                raise ValueError("Cached input layout/recipe differs")
            return condition, aligned_prediction(prediction_path, case_ids), dict(info, new_fits_this_run=0)
        begin = time.monotonic()
        # Original actual-target predictions remain unchanged and are not scored
        # as newly fitted actual-target models in this requested-target comparison.
        prediction = old["power"].copy()
        folder = OUT / condition
        folder.mkdir(exist_ok=True)
        for lead in range(6):
            model = LGBMRegressor(n_jobs=2, random_state=2027, verbosity=-1)
            model.fit(x[train][:, columns[condition]], y[train, lead, 1])
            prediction[:, lead, 1] = np.maximum(model.predict(x[:, columns[condition]]), 0)
            model.booster_.save_model(str(folder / f"requested_lead{lead}.txt"))
        info = dict(condition=condition, columns=columns[condition].tolist(),
                    feature_names=[feature_schema()[i] for i in columns[condition]],
                    parameters=model.get_params(), lead_fits=6, new_fits_this_run=6,
                    fit_seconds=time.monotonic() - begin)
        np.savez_compressed(prediction_path, case_id=case_ids, prediction_kw=prediction)
        info_path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(dict(completed=condition, new_fits=6)), flush=True)
        return condition, prediction, info

    predictions = dict(old)
    fit_records = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(fit, name) for name in ["own_history", "own_history_state"]]
        for job in jobs:
            condition, prediction, info = job.result()
            predictions[condition] = prediction
            fit_records.append(info)
    predictions["own_target_persistence"] = np.repeat(
        meta[["last_actual_kw", "last_requested_kw"]].to_numpy()[:, None, :], 6, axis=1)
    scored = []
    for name, prediction in predictions.items():
        value = pilot.score(meta, y, prediction, name)
        scored.append(value[value.target == "requested_power"])
    daily = pd.concat(scored, ignore_index=True)
    daily.to_csv(OUT / "daily_scores.csv.gz", index=False)
    summary, averages, contrasts = outputs(daily)
    final = dict(
        completed=True, seconds=time.monotonic()-start,
        new_lead_fits_this_run=sum(r["new_fits_this_run"] for r in fit_records),
        total_alignment_lead_fits=12, fits=fit_records,
        reused=["power", "power_state", "full", "packet", "cases", "scorer"],
        support=meta.groupby("split").agg(sessions=("case_id", "size"), dates=("date", "nunique")).reset_index().to_dict(orient="records"),
        target="Logged EV-requested power (kW); actual-target predictions are not newly fitted or scored here.",
        history="Every aligned arm includes ten observed actual-power AND requested-power values and calendar.",
        uncertainty="Paired circular blocks of seven observed dates, 2000 replicates; fixed fitted models. Not seven consecutive calendar dates or model-seed uncertainty.",
        original_pilot="Unchanged; no raw workbook reads or source edits.",
        recipe="Same default LightGBM as original pilot; no HPO or selection.")
    (OUT / "receipt.json").write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
    print(averages[averages.split=="later"].round(3).to_string(index=False), flush=True)
    print(contrasts[(contrasts.split=="later") & (contrasts.span_minutes=="average") &
                    (contrasts.metric=="mse")].round(3).to_string(index=False), flush=True)
    print(json.dumps({k: v for k, v in final.items() if k != "fits"}), flush=True)


if __name__ == "__main__":
    main()

