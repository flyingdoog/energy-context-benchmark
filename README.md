# Energy Context Benchmark

**When Does Context Help Forecasting? An Empirical Study of Energy Systems**

Xu Zheng, Jingchao Ni, Mo Sha, and Dongsheng Luo

Accepted at the **IEEE International Conference on Computer & Applications 2026**.

[Download data](https://github.com/flyingdoog/energy-context-benchmark/releases/download/v1.0.0/context-energy-forecasting-data-v1.0.0.zip) · [Data guide](data/README.md) · [Reproduction guide](docs/REPRODUCING.md) · [MIT license](LICENSE)

This study asks when operational records improve forecasts beyond power history.
It covers water heaters, workplace and fast EV charging, and building air
conditioning, using four source collections: BPA, ACN-Data, EPFL, and CU-BEMS.
Ridge, LightGBM, and N-HiTS compare matched information conditions. A separate
record-reassignment experiment distinguishes a fitted predictor's sensitivity
from the effect of retraining with reassigned records.

## Quick start

Use Python 3.12 in your own environment. From the repository root:

```sh
python -m pip install -r code/requirements-operational.txt
python code/inspect_data.py --source epfl
```

The EPFL prepared packet is included: 1,771 sessions with inputs, targets and
chronological splits. This command reads data without fitting a model.
For all seven CU-BEMS floors, extract the separate
[`context-energy-forecasting-data-v1.0.0.zip`](https://github.com/flyingdoog/energy-context-benchmark/releases/download/v1.0.0/context-energy-forecasting-data-v1.0.0.zip)
at the repository root, then run
`python code/inspect_data.py`. See the [data guide](data/README.md).

## Contents

| Directory | Contents |
| --- | --- |
| `results/` | Saved aggregate scores used in the paper |
| `data/` | Prepared EPFL data, public data catalog and provider notices |
| `docs/` | Reproduction and public release instructions |
| `code/src/`, `code/scripts/` | Source preparation, forecasting interfaces, scoring, and record-comparison tools |
| `code/tests/`, `code/docs/` | Runtime tests and source-specific instructions |
| `code/research_snapshot/` | Implementations from the completed research experiments |

Prepared EPFL data are included; prepared CU-BEMS data are supplied in the
separate data asset. BPA and ACN raw/prepared records are obtained from the
providers. No model weights, environments, credentials or prediction caches
are bundled. The reported aggregate results are available without downloading
data or training models. Manuscript PDF, LaTeX sources and typesetting tools are
not part of this repository or its release assets.

## Reported results

The [reproduction guide](docs/REPRODUCING.md) maps each experimental table and
figure to the saved CSV/JSON files in `results/`. The files include device-level
errors, the declaration study, and record-reassignment estimates and intervals.
These are the existing experimental results; no additional research experiments
were run to prepare this public release.

See [v1.0.0](https://github.com/flyingdoog/energy-context-benchmark/releases/tag/v1.0.0)
for the code snapshot and separate prepared-data asset.

## Prepare data and evaluate forecasts

Install `code/requirements-operational.txt`, then follow the
[runtime guide](code/README.md). Source guides assume the working directory is
`code/`:

- [BPA water heaters](code/docs/BPA_QUICKSTART.md)
- [ACN workplace charging](code/docs/ACN_QUICKSTART.md)
- [EPFL fast charging](code/docs/EPFL_QUICKSTART.md)
- [CU-BEMS air conditioning](code/docs/CU_BEMS_QUICKSTART.md)

The runtime preserves observation masks, native forecast intervals, device
identity, and complete-group membership. It supports
[custom forecasting models](code/docs/RECORD_CUSTOM_MODEL.md) and a
[four-arm record-association comparison](code/docs/RECORD_COMPARISON_RUNTIME.md).
Model dependencies are installed separately from preparation and scoring tools.

The research snapshot retains the original experimental directory conventions
and additional model dependencies. The [reproduction guide](docs/REPRODUCING.md)
maps the paper's results to data and implementations, distinguishing saved
measurements, portable evaluation, and the original training entries.

Run the local tests (synthetic fixtures and mocked acquisition):

```sh
python -m unittest discover -s code/tests -v
python code/scripts/check_public_artifact.py
```

## Data attribution and licensing

Original project code is [MIT licensed](LICENSE). This does not relicense the
paper or third-party data. EPFL packets retain the
[DESL/EPFL MIT notice](data/epfl/source_notices/LICENSE); CU-BEMS retains
[CC BY 4.0 attribution](data/CU_BEMS_ATTRIBUTION.md). BPA and ACN redistribution
permissions have not been established, so this repository supplies access and
preparation instructions rather than their records. See
[source notes](code/docs/SOURCE_NOTES.md) and
[dependency notices](code/THIRD_PARTY_NOTICES.md).

## Citation

```bibtex
@inproceedings{zheng2026contextenergy,
  title = {When Does Context Help Forecasting? An Empirical Study of Energy Systems},
  author = {Zheng, Xu and Ni, Jingchao and Sha, Mo and Luo, Dongsheng},
  year = {2026},
  booktitle = {IEEE International Conference on Computer \& Applications},
  note = {Accepted for publication},
  url = {https://github.com/flyingdoog/energy-context-benchmark}
}
```
