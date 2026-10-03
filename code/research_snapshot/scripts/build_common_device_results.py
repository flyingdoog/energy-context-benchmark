"""Combine saved device-scale BPA/ACN/EPFL scores without any new model fit."""
import argparse
import json
from pathlib import Path
import run_local
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from build_operational_results import load_results, CONTEXTS, MODELS
from align_epfl_target_history import paired_contrast

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/common_device_comparison_v1"
EPFL = ROOT / "results/epfl_actual_power_comparison_v1"
RETRIEVAL = ROOT / "results/epfl_prefix_retrieval_v1"
CONDITIONS = ("power", "power_state", "full")
METRICS = ["mse", "mae", "energy_ae_kwh"]
CONTEXT_LIST = CONTEXTS + [("EPFL", "Fast charging", "later")]
KEYS = ["source", "site", "period", "model", "condition"]


def summarize_common(data):
    """Equal-span averages within each source/site/period/model/input view."""
    if data.duplicated(KEYS + ["span_minutes"]).any():
        raise ValueError("Duplicate device comparison cell")
    if not np.isfinite(data[METRICS]).all().all():
        raise ValueError("Nonfinite scores")
    for (source, site, period), group in data.groupby(["source", "site", "period"]):
        expected = {5, 15, 30} if source == "EPFL" else {30, 90, 150}
        if set(group.model) != set(MODELS) or set(group.condition) != set(CONDITIONS):
            raise ValueError("Incomplete model/input matrix")
        if group.dates.nunique() != 1 or group.units.nunique() != 1:
            raise ValueError("Different scored case/date support within a context")
        for _, part in group.groupby(["model", "condition"]):
            if set(part.span_minutes) != expected or len(part) != 3:
                raise ValueError("Incomplete source-native forecast spans")
    return data.groupby(KEYS).agg(**{m: (m, "mean") for m in METRICS},
                                  dates=("dates", "first"), units=("units", "first")).reset_index()


def load_common():
    original = load_results()
    data = original[(original.scale == "device") & original.condition.isin(CONDITIONS)].copy()
    epfl = pd.read_csv(EPFL / "summary.csv")
    epfl = epfl[(epfl.split == "later") & (epfl.target == "actual_power") & epfl.family.isin(MODELS)]
    epfl = epfl.rename(columns={"family": "model", "method": "condition"})
    meta = pd.read_csv(ROOT / "results/epfl_dataset_v1/cases.csv")
    support = meta[meta.split == "later"]
    epfl = epfl.assign(source="EPFL", site="Fast charging", period="later", scale="device",
                       dates=support.date.nunique(), units=len(support))
    result = pd.concat([data, epfl[data.columns]], ignore_index=True)
    if len(result) != 189: raise ValueError("Expected 189 common score cells")
    return result


def write_common(data, average):
    names = ["BPA spring", "BPA summer", "Caltech autumn", "Caltech winter", "Office autumn", "Office winter", "EPFL later"]
    lines = ["# Device-scale operational forecasting", "",
        "Date-balanced MSE (kW squared) / MAE (kW), averaged equally over each source's three forecast spans. BPA and ACN: 30, 90, 150 minutes. EPFL: 5, 15, 30 minutes. Contexts retain their own query schedules and observed devices/sessions; the seven columns are periods/settings from three source collections, not seven datasets. History includes calendar inputs. N-HiTS averages three seed errors, not predictions. All baseline recipes are fixed.", "",
        "| Model | Inputs | " + " | ".join(names) + " |",
        "|---|---|" + "|".join(["---:"] * len(names)) + "|"]
    labels = dict(power="History", power_state="History + state", full="All inputs")
    for model in MODELS:
        for condition in CONDITIONS:
            cells = []
            for source, site, period in CONTEXT_LIST:
                row = average[(average.source == source) & (average.site == site) &
                              (average.period == period) & (average.model == model) &
                              (average.condition == condition)]
                if len(row) != 1: raise ValueError("Missing or ambiguous main-table cell")
                r = row.iloc[0]; cells.append(f"{r.mse:.3f} / {r.mae:.3f}")
            lines.append("| " + " | ".join([model, labels[condition], *cells]) + " |")
    lines += ["", "No raw-score average is computed across sources. Scored units are device-query cases, not unique devices.", "",
              "| Context | Dates | Device-query cases |", "|---|---:|---:|"]
    for name, (source, site, period) in zip(names, CONTEXT_LIST):
        r = average[(average.source == source) & (average.site == site) & (average.period == period)].iloc[0]
        lines.append(f"| {name} | {int(r.dates)} | {int(r.units):,} |")
    dest = ROOT / "paper/generated/common_device_results.md"
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Unlike the span-average table, this figure displays the error trajectory.
    fig, axes = plt.subplots(1, 4, figsize=(11.8, 3.4))
    colors = {"Ridge": "#7567a2", "LightGBM": "#167c94", "N-HiTS": "#c37c24"}
    settings = [("BPA", "Field study", "BPA"), ("ACN", "Caltech", "ACN Caltech"),
                ("ACN", "Office", "ACN Office"), ("EPFL", "Fast charging", "EPFL")]
    for ax, (source, site, label) in zip(axes, settings):
        part = data[(data.source == source) & (data.site == site) & (data.condition == "full")]
        curve = part.groupby(["model", "span_minutes"])["mse"].mean()
        for model in MODELS:
            row = curve.loc[model].sort_index()
            ax.plot(row.index, row, marker="o", ms=4, lw=1.6, color=colors[model], label=model)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Forecast span (minutes)", fontsize=9)
        ax.set_ylabel("MSE (kW²)", fontsize=9)
        ax.set_xticks([5, 15, 30] if source == "EPFL" else [30, 90, 150])
        ax.set_ylim(bottom=0); ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2); ax.tick_params(labelsize=9)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=[0, .12, 1, 1])
    fig.savefig(OUT / "full_input_horizon_profiles.png", dpi=160)
    plt.close(fig)
    (ROOT / "paper/generated/common_device_figure_caption.md").write_text(
        "All-input device forecasting across source-native forecast spans. MSE uses equal-date scoring within a period. BPA and each ACN site average their two evaluated periods equally; EPFL has one later period. Each panel has its own vertical scale. All three model recipes are fixed; N-HiTS curves average three independent seed errors. The span-average table retains separate periods.\n", encoding="utf-8")
    return lines


def write_retrieval():
    main = pd.read_csv(EPFL / "summary.csv")
    new = pd.read_csv(RETRIEVAL / "summary.csv").assign(family="Prefix retrieval")
    combined = pd.concat([main, new], ignore_index=True)
    combined.to_csv(OUT / "epfl_extended_scores.csv", index=False)
    later = combined[combined.split == "later"]
    lines = ["# EPFL charging-prefix retrieval comparison", "",
        "Actual-power MSE (kW squared) / MAE (kW) on the same 739 later sessions and 100 dates. Five uniformly averaged training analogs, no HPO. Prefix retrieval is a project-implemented time-aligned reference, not an exact reproduction of a published physical or SoC-profile method. N-HiTS entries average three seed errors.", "",
        "| Predictor | Inputs | 5 min | 15 min | 30 min | Average |", "|---|---|---:|---:|---:|---:|"]
    for family in ["Persistence", "Prefix retrieval", *MODELS]:
        conditions = ["own_target_persistence"] if family == "Persistence" else CONDITIONS
        for condition in conditions:
            p = later[(later.family == family) & (later.method == condition)].set_index("span_minutes")
            if set(p.index) != {5, 15, 30}: raise ValueError("Missing EPFL span")
            cells = [f"{p.loc[h, 'mse']:.3f} / {p.loc[h, 'mae']:.3f}" for h in [5, 15, 30]]
            cells += [f"{p.mse.mean():.3f} / {p.mae.mean():.3f}"]
            label = dict(power="History", power_state="History + state", full="All inputs", own_target_persistence="Last actual power")[condition]
            lines.append("| " + " | ".join([family, label, *cells]) + " |")
    (ROOT / "paper/generated/epfl_retrieval_comparison.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    daily = pd.concat([pd.read_csv(EPFL / "daily_scores.csv.gz"),
                       pd.read_csv(RETRIEVAL / "daily_scores.csv.gz").assign(family="Prefix retrieval")])
    contrasts = []
    for condition in CONDITIONS:
        p = daily[(daily.split == "later") & (daily.method == condition)]
        for metric in METRICS:
            wide = p.groupby(["date", "family"])[metric].mean().unstack().sort_index()
            for candidate in MODELS:
                pair = wide[["Prefix retrieval", candidate]]
                if pair.isna().any().any(): raise ValueError("Unpaired reference dates")
                contrasts.append(dict(condition=condition, metric=metric, baseline="Prefix retrieval",
                                      candidate=candidate, **paired_contrast(pair.iloc[:, 0], pair.iloc[:, 1])))
    pd.DataFrame(contrasts).to_csv(OUT / "epfl_retrieval_contrasts.csv", index=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main-only', action='store_true',
                        help='Write the three-family main table without requiring prefix-retrieval results.')
    args = parser.parse_args(argv)
    OUT.mkdir(exist_ok=True)
    data = load_common(); average = summarize_common(data)
    data.to_csv(OUT / "common_device_scores.csv", index=False)
    average.to_csv(OUT / "common_device_span_average.csv", index=False)
    lines = write_common(data, average)
    if not args.main_only:
        write_retrieval()
    receipt = dict(completed=True, new_model_fits=0, shared_cells=len(data),
        summary_cells=len(average), source_collections=3, device_families=2,
        context_periods=len(CONTEXT_LIST), source_native_spans={"BPA_ACN": [30,90,150], "EPFL": [5,15,30]},
        no_cross_source_raw_score_average=True, original_results_modified=False,
        retrieval_included=not args.main_only,
        figure="Full-input horizon profiles; two-period means only within each BPA/ACN setting")
    (OUT / "receipt.json").write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
    print("\n".join(lines)); print(json.dumps(receipt))


if __name__ == "__main__": main()
