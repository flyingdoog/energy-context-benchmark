"""Run a local Chronos-2 checkpoint on one complete native operational split."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import run_local
import numpy as np
from field_control import OperationalForecastTask
from field_control.chronos_inputs import make_chronos_items


def dump(path, data):
    temporary = path.with_suffix(".partial.json")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(path)


def native_items(task, indices, condition):
    column = "query_utc" if task.source == "ACN" else "query"
    times = None if task.source == "BPA" else task.meta.iloc[indices][column].tolist()
    return make_chronos_items(task.inputs(indices, condition=condition), query_times=times)


def point_forecasts(pipeline, items, grid, batch_variates):
    _, points = pipeline.predict_quantiles(
        items, prediction_length=grid.prediction_length,
        quantile_levels=[.1, .5, .9], batch_size=batch_variates, cross_learning=False)
    # The tested API returns the native-step median in its second output.
    native = np.stack([value.cpu().numpy()[0] for value in points])
    return grid.aggregate(native)


def saved_batch(path, case_ids, width):
    with np.load(path, allow_pickle=False) as saved:
        if saved["case_id"].tolist() != case_ids.tolist():
            raise ValueError("Saved batch is for different cases")
        values = saved["prediction_kw"].copy()
    if values.shape != (len(case_ids), width) or not np.isfinite(values).all():
        raise ValueError("Saved batch shape or values differ")
    return values


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["BPA", "ACN", "EPFL", "CU-BEMS"], required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--season")
    parser.add_argument("--site")
    parser.add_argument("--split", required=True)
    parser.add_argument("--condition", choices=["power", "power_state", "full"], default="full")
    parser.add_argument("--checkpoint", type=Path, help="Existing local checkpoint; no automatic download")
    parser.add_argument("--model-label", default="Chronos-2")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--case-batch", type=int, default=32)
    parser.add_argument("--variate-batch", type=int, default=256)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--describe-inputs", action="store_true", help="Inspect at most two inputs; no model or score")
    args = parser.parse_args(argv)
    if min(args.case_batch, args.variate_batch, args.threads) < 1:
        parser.error("Batch sizes and threads must be positive")
    if not args.describe_inputs:
        if args.checkpoint is None or not (args.checkpoint/"config.json").is_file():
            parser.error("Provide an existing local checkpoint directory with config.json")
        if args.output is None:
            parser.error("--output is required for a forecast run")
    task = OperationalForecastTask(args.data, args.source, season=args.season, site=args.site)
    try:
        indices = task.indices(args.split)
        if args.condition not in task.contract["available_conditions"]:
            raise ValueError("Requested input condition unavailable at this site")
        items, grid = native_items(task, indices[:min(2, len(indices))], args.condition)
        if args.describe_inputs:
            print(json.dumps(dict(source=args.source, site=task.site, split=args.split,
                condition=args.condition, split_cases=len(indices), preview_cases=len(items),
                grid=asdict(grid), past_fields=list(items[0]["past_covariates"]),
                future_fields=list(items[0]["future_covariates"]),
                new_predictions=0, scores_produced=False)), flush=True)
            return
        protocol = dict(source=args.source, site=task.site, season=task.season,
            split=args.split, condition=args.condition, data=str(args.data.resolve()),
            checkpoint=str(args.checkpoint.resolve()), model=args.model_label, grid=asdict(grid),
            case_batch=args.case_batch, variate_batch=args.variate_batch, device=args.device,
            cross_learning=False, point="native-step median averaged within output bins",
            hpo=False, new_fits=0, forecast_targets_passed_to_model=False)
        args.output.mkdir(parents=True, exist_ok=True)
        protocol_path = args.output/"protocol.json"
        if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("Output folder belongs to a different run; use a new folder")
        dump(protocol_path, protocol)
        if (args.output/"complete.json").exists():
            print("Complete run exists; reuse its submission and scores.", flush=True)
            return
        os.environ.update(HF_HUB_OFFLINE="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
                          HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false")
        import torch
        from chronos import Chronos2Pipeline
        torch.set_num_threads(args.threads)
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA device unavailable; no silent CPU fallback")
        began = time.perf_counter()
        pipeline = Chronos2Pipeline.from_pretrained(str(args.checkpoint), device_map=args.device, local_files_only=True)
        pipeline.model.eval()
        predictions, new_queries = [], 0
        for start in range(0, len(indices), args.case_batch):
            ix = indices[start:start+args.case_batch]
            ids = task.meta.case_id.iloc[ix].to_numpy(dtype=str)
            path = args.output/f"batch_{start:07d}.npz"
            if path.exists():
                values = saved_batch(path, ids, task.width)
            else:
                items, batch_grid = native_items(task, ix, args.condition)
                with torch.inference_mode():
                    values = point_forecasts(pipeline, items, batch_grid, args.variate_batch)
                temp = path.with_suffix(".partial.npz")
                np.savez_compressed(temp, case_id=ids, prediction_kw=values)
                temp.replace(path)
                new_queries += len(ix)
            predictions.append(values)
            progress = dict(pid=os.getpid(), stage="inference", done=start+len(ix), total=len(indices),
                            updated_at_utc=datetime.now(timezone.utc).isoformat())
            dump(args.output/"worker.json", progress)
            print(json.dumps(progress), flush=True)
        prediction = np.concatenate(predictions)
        task.submission(args.output/"submission.npz", split=args.split, prediction_kw=prediction)
        summary, daily = task.score(split=args.split, prediction_kw=prediction,
                                   method=args.model_label, require_groups=task.supports_groups)["power"]
        summary.to_json(args.output/"summary.json", orient="records", indent=2)
        daily.to_json(args.output/"daily_scores.json", orient="records", indent=2)
        receipt = dict(complete=True, cases=len(indices), new_predictions=new_queries,
            new_fits=0, summary_rows=len(summary), seconds=time.perf_counter()-began,
            completed_at_utc=datetime.now(timezone.utc).isoformat())
        dump(args.output/"complete.json", receipt)
        dump(args.output/"worker.json", dict(stage="complete", **receipt))
        print(json.dumps(receipt), flush=True)
    finally:
        if hasattr(task.base, "close"):
            task.base.close()


if __name__ == "__main__":
    main()
