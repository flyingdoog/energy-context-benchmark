# Operational forecasting runtime

## Scope

This directory contains the source-preparation and evaluation runtime,
and a separate `research_snapshot/` archive. Saved aggregate
results are in the repository root's `results/`. Prepared EPFL data are included;
the separate data asset supplies CU-BEMS. Obtain BPA and ACN from their providers.
The runtime and the research snapshot each contain a `field_control` package;
use only one of their `src/` directories on the Python import path at a time.

The commands below and in `docs/` assume the current directory is `code/`.
Choose local data and output directories when following the source guides.
Start with [the release instructions](../README.md): `inspect_data.py` opens
locally prepared packets without a model library. The root `results/` directory
contains the saved experimental measurements; manuscript files are not included.

The research snapshot preserves original experiment paths and additional model
dependencies; it is not a portable one-command retraining bundle. Historical
reconstruction/test claims below describe the original runtime, not new tests
performed for this manuscript revision.

Read original device histories, build a source-faithful forecasting task, and
score portable power/occurrence submissions without model-library dependencies.

## Install

    python -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements-operational.txt

On Linux/macOS use .venv/bin/python. Python 3.12 is the locally tested version.
For every subsequent `python` command in these guides, use this environment's
interpreter: `.\.venv\Scripts\python.exe` on Windows or `.venv/bin/python` on
Linux/macOS. Creating a virtual environment does not activate it. You do not
need to change your system Python or PowerShell execution policy.
NumPy, pandas, SciPy, openpyxl and inflate64 cover the four sources; no PyTorch
or LightGBM is needed for preparation and the simple submission examples.

## Choose the source

- [EPFL quickstart](docs/EPFL_QUICKSTART.md): public workbook download through
  preparation and scoring. Independently installed local reconstruction is complete.
- [ACN quickstart](docs/ACN_QUICKSTART.md): fixed public-source acquisition,
  calendar manifest, preparation and scoring. Acquisition defaults to metadata
  planning; `--sample` is an access check, while `--download` explicitly fetches
  the full collection. Samples cannot be used for evaluation. Both sites and
  all four examples passed isolated local reconstruction using existing files.
- [BPA quickstart](docs/BPA_QUICKSTART.md): existing original winter, spring and
  summer archives through preparation and scoring. Two CPU workers by default;
  no download. All 174,436 cases and 36 example score rows were recovered by the
  author-run isolated reconstruction; that runtime passed 48 tests.
- [CU-BEMS quickstart](docs/CU_BEMS_QUICKSTART.md): an original 2019 floor CSV
  through air-conditioner, floor and zone scoring. Thermal measurements retain
  their source units and missingness masks. No internal experiment paths or
  trained models are required.

Examples use training-only state frequencies and power persistence, not new
trained forecasting models. Scorers retain each source's physical units,
native time bins, missingness and supported group definitions.

## Use your own model

The [record-association comparison](docs/RECORD_COMPARISON_RUNTIME.md) trains
default Trees on prepared ACN/BPA inputs and compares power-only, paired,
fixed-model reassignment and retrained reassignment. Its preview is model-free;
execution requires the optional tree dependency. No saved research models or
prediction caches are needed.

The [user-model comparison guide](docs/RECORD_CUSTOM_MODEL.md) shows how to
provide your own fit/predict model, reuse the prepared input views, and obtain
the same four-arm result tables. A small synthetic example runs without source
data or optional model dependencies.

The [group-level diagnostic](docs/RECORD_LEVEL_DIAGNOSTIC.md) reads a completed
record-comparison run and evaluates a validation-only offset control, with
original/corrected MSE/MAE at every native span. It makes no model calls and
includes a separate source-free example.

Save your forecasting code as a Python file, then run it through the local
launcher. It makes `from field_control import OperationalForecastTask` work
without installing the project globally or editing your Python path:

    python scripts/run_model.py my_forecaster.py --your-option value

The launcher passes the remaining arguments to your script. The source guides
show how to load the prepared task and write a portable submission. Relative
data/output paths are resolved from your current directory. The runtime does
not install a modeling library; install your model's dependencies in your own
environment when needed.

The [optional auxiliary-report blackout](docs/OPERATIONAL_TELEMETRY_RECENCY.md)
holds the query and targets fixed while making recent state/control records
unavailable. The default inputs and power-only view are unchanged. Its small
example runs without model libraries; the guide explains matched training
and how to read unknown boolean/categorical values with their masks.

## Optional pretrained reference

The [Chronos-2 quickstart](docs/CHRONOS_REFERENCE_QUICKSTART.md) provides a
shared entry for an existing local checkpoint on any of the four sources.
It includes a model-free input preview, resumable inference, a portable
submission and native device/group scoring. Model dependencies are optional;
the base preparation and evaluation environment remains model-library-free.

See [source notes](docs/SOURCE_NOTES.md) for original data access and citations,
and [dependency notices](THIRD_PARTY_NOTICES.md) for the separately installed
runtime dependencies. Third-party packages are installed separately. Included
EPFL and CU-BEMS packets retain the providers' licenses and attribution.

Run the packaged tests with:

    python -m unittest discover -s tests -v

The repository includes saved aggregate scores, not model weights or prediction
caches. Original project code uses the root [MIT license](../LICENSE);
provider terms govern source data. See the [data guide](../data/README.md).
