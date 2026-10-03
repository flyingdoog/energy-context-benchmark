# Runtime dependencies and source notices

Original project code uses the root [MIT license](../LICENSE).
This release does not include third-party wheels, binaries or model weights.
`requirements-operational.txt` installs dependencies separately. Prepared EPFL
data retain the source MIT notice; the separate CU-BEMS data asset retains
CC BY 4.0 attribution. Dependency licenses below do not relicense source data.

## Tested direct dependencies

The following versions and license fields were read from the installed
distributions' metadata and notice files in the separately installed Windows
Python 3.12 reconstruction environment on September 8, 2026. Requirement ranges
are compatibility bounds, not a promise that every version was tested.

| Distribution | Tested version | Observed license information |
| --- | --- | --- |
| NumPy | 2.5.3 | Package license expression: BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0; the wheel includes further component notices |
| pandas | 3.0.5 | BSD 3-Clause License; full text in its installed LICENSE |
| SciPy | 1.18.1 | Main license: three-clause BSD; the wheel includes further component notices |
| openpyxl | 3.1.5 | MIT; full text in its installed LICENCE.rst |
| inflate64 | 1.0.4 | LGPL-2.1-or-later; full text in its installed COPYING |

Both inspected NumPy and SciPy Windows wheels also identify OpenBLAS
(BSD-3-Clause), LAPACK (BSD-3-Clause-Open-MPI), and GCC runtime libraries
(GPL-3.0-or-later WITH GCC-exception-3.1). This summary is not a replacement for
their complete notices, nor a complete inventory of another platform's wheels.
If later distributing an environment or vendored dependency, retain the full
license and component notices from the actual versions being distributed.
Do not infer one combined benchmark license from this table.

Inspect the versions in your environment with:

    python -m pip show numpy pandas scipy openpyxl inflate64

Installed distribution metadata retains each package's original authorship
and license files. The runtime's dependencies install their own transitive
requirements; those packages are not copied into this code archive.

## Data is separate

The optional Chronos entry adds a separately installed model-library
requirement. Chronos, Transformers and their PyTorch dependencies, caches and
weights are not copied into this artifact. Their distribution notices remain
with the installed packages; the checkpoint's model card governs its use.
The tested model-library versions are listed in requirements-chronos.txt.

See [source notes](docs/SOURCE_NOTES.md) for data providers, source revisions,
scholarly citations and access instructions. BPA and ACN data redistribution
permissions have not been established here, so these records are not bundled.
EPFL downloads and prepared packets retain the provider's original MIT notice.
See [CU-BEMS attribution](../data/CU_BEMS_ATTRIBUTION.md) for the source DOI,
creators, license and preprocessing description.
