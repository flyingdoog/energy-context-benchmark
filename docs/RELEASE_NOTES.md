# IEEE ICCA 2026 - Code and Data v1.0.0

Code and data accompanying **When Does Context Help Forecasting? An Empirical
Study of Energy Systems**, accepted at IEEE ICCA 2026.

Xu Zheng, Jingchao Ni, Mo Sha, and Dongsheng Luo.

## Included

- MIT-licensed preparation, forecasting interfaces, evaluation and comparison code.
- The original experimental implementation snapshot and saved aggregate results.
- Prepared EPFL data (1,771 sessions), with the original DESL/EPFL MIT notice.
- Prepared CU-BEMS data for seven floors in one building, with CC BY 4.0 attribution.
- Source guides, chronological split metadata and reproduction instructions.

## Downloads

- `context-energy-forecasting-code-v1.0.0.zip`: the repository snapshot.
- `context-energy-forecasting-data-v1.0.0.zip`: EPFL and CU-BEMS prepared packets.
  Extract this asset into the root of the checked-out repository or code archive.

BPA and ACN records are not redistributed; source acquisition and preparation
instructions are included. The separate ACN declaration study requires access
to the corresponding session metadata. No model weights, credentials, manuscript
PDF or LaTeX source files are included.

## Checks and scope

The offline suite has 141 tests: 140 passed and one optional LightGBM test was
skipped because that dependency was not installed in the release-check
environment. EPFL and all seven CU-BEMS prepared-data loaders were checked.
Release preparation did not run new research experiments. Historical training
entries retain their intermediate directory conventions; see `docs/REPRODUCING.md`
for the requirements when adapting them to a new machine.
