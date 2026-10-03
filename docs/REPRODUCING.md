# Reproducing the paper

This guide accompanies *When Does Context Help Forecasting? An Empirical Study
of Energy Systems*. Commands in this document run from the repository root.

## 1. Inspect the reported results

The aggregate measurements are included as CSV and JSON. No source data,
trained model or GPU is needed to read them. Manuscript PDF, LaTeX sources and
typesetting scripts are not distributed in this code/data repository.

| Paper item | Supplied measurements / definitions |
| --- | --- |
| Table I: source/task inventory | Source definitions and chronological splits in the source guides and `data/catalog.json` |
| Table II: device errors | `results/device_average.csv`, `results/cu_common_scores.json`, `results/paper_device_cells.json` |
| Table III: demand declarations | `results/declarations_tree.csv`, `results/declarations_nhits.csv`, `results/paper_declaration_cells.json` |
| Figure 2: record reassignment | `results/reassignment_effects.json`, including saved intervals |

The stored intervals are from the completed study. No bootstrap inference or
model training was rerun to prepare this release.

## 2. Load data and evaluate forecasts

```sh
python -m pip install -r code/requirements-operational.txt
python code/inspect_data.py --source epfl
```

EPFL's prepared packet is in the repository. Extract the separate
`context-energy-forecasting-data-v1.0.0.zip` into the repository root to add all
seven CU-BEMS floors, then run `python code/inspect_data.py --source cu_bems`.
The [data guide](../data/README.md) gives input/target formats, split definitions,
source versions and download commands. No personal credentials are needed for
these two collections.

BPA and ACN records are not redistributed. Use the [BPA](../code/docs/BPA_QUICKSTART.md)
and [ACN](../code/docs/ACN_QUICKSTART.md) guides to acquire and prepare them.
The ACN static telemetry release does **not** include the separate demand
declarations used in Table III; that experiment also needs authorized access
to the corresponding ACN session metadata containing `userInputs`.

For your model, the [runtime guide](../code/README.md) and
[custom-model example](../code/docs/RECORD_CUSTOM_MODEL.md) expose inputs,
targets, named splits and prediction-submission scoring. Source-specific guides
run from `code/`. Use the chronological validation split for model choices.
Evaluation preserves native intervals and observation masks. Main-table MSE is
in kW squared and MAE in kW, with equal date and native-span weights; CU-BEMS
additionally weights its five common thermal floors equally.

## 3. Repeat record-association comparisons

The portable [four-arm comparison](../code/docs/RECORD_COMPARISON_RUNTIME.md)
supports power history, paired records, fixed-model reassignment and retraining
with reassigned records on prepared BPA/ACN data. It has a preview mode before
training. Install `code/requirements-record.txt` only when running the tree
models. The guide lists dataset paths, donor records, mapping seeds and outputs.

The paper's LightGBM uses 100 trees and 31 leaves. Ridge standardizes inputs
using training statistics, fits an intercept and uses unit L2 regularization.
The N-HiTS runs use 1,000 squared-error updates and three seeds. Source/floor
models are fitted separately. These are the recorded experimental settings,
not newly tuned configurations for this release.

## 4. Original training implementations

`code/research_snapshot/` preserves the implementations used during the study:

| Experiment | Files under `code/research_snapshot/scripts/` |
| --- | --- |
| BPA typed telemetry and neural repeats | `bpa_typed_inputs.py`, `run_bpa_typed_repeats.py`, `aggregate_bpa_typed_repeats.py` |
| ACN telemetry | `run_acn_fleet_reference.py`, `run_acn_native_neural.py`, `run_acn_neural_repeats.py` |
| EPFL telemetry | `pilot_epfl_operational_forecasting.py`, `run_epfl_native_neural.py` |
| CU-BEMS common-floor comparison | `run_cu_bems_full_reference.py`, `analyze_cu_bems_full_reference.py`, `run_cu_bems_native_neural.py` |
| ACN declarations | `run_acn_declared_demand_reference.py`, `run_acn_native_neural.py --declarations`, `analyze_acn_declared_neural.py` |
| Record reassignment | `run_record_linkage.py`, `run_frozen_reassignment.py`, `analyze_frozen_reassignment.py` |

These entries retain their historical `results/...` input/intermediate layout.
They are implementation references, not a one-command retraining pipeline for
the public prepared packets. Read each entry's `DATA`/`BASE`/`OUT` constants and
CLI before adapting paths. Neural entries additionally need their model
libraries. The recorded EPFL seed-1 full-input run used `neuralforecast 3.2.1`,
`torch 2.14.0+cpu` and `numpy 2.5.2`; it used 1,000 steps and learning rate 0.001.
These are versions from that run's receipt, not a newly tested environment lock
for every source. Use either the portable runtime's `src/` or the snapshot's `src/`
on the import path, not both. No weights or per-example prediction caches are
included. Table III additionally requires the declarations source noted above.

## 5. Release checks

```sh
python -m unittest discover -s code/tests -v
python code/scripts/check_public_artifact.py
```

Tests use synthetic fixtures and mocked downloads. They exercise preparation,
input views, scoring and command interfaces; they do not retrain the paper's
models. The artifact check scans required files, Python syntax, private paths,
credential-like strings and datasets excluded from redistribution. Original
code is MIT licensed; data licenses are listed in [the data guide](../data/README.md).
