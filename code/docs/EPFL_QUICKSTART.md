# EPFL source-to-score quickstart

This small runtime reconstructs the operational forecasting task directly from
the two original EPFL workbooks, then exports and scores a simple forecast.
It does not need historical research caches, fitted models or neural libraries.
The common task API also supports the [ACN](ACN_QUICKSTART.md) and
[BPA](BPA_QUICKSTART.md) preparation paths included in the same runtime.

## Install locally

From the extracted runtime directory, use a separate environment. Windows:

    python -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements-operational.txt
    .\.venv\Scripts\python.exe scripts/quickstart_epfl.py --download --workdir run

On Linux/macOS, use .venv/bin/python for the final two commands. Python 3.12
is the environment used for the local reconstruction. The runtime requires
NumPy, pandas, SciPy and openpyxl for EPFL; the common requirements also include
inflate64 for BPA. No LightGBM, PyTorch or scikit-learn is needed.
Exact installed versions appear in the run receipt.
Below, `python` means the environment's interpreter, not an unchanged system
Python. Use `.\.venv\Scripts\python.exe` or `.venv/bin/python` explicitly.

The download option reads approximately 24 MB from the original
[DESL/EPFL repository](https://github.com/DESL-EPFL/Level-3-EV-charging-dataset)
at revision 680bc3bb751f86a47db32a5641f09dcfa079caae. It retains the original
README and MIT license with both workbooks. No data is sent to another system.
To use already acquired, unmodified source workbooks:

    python scripts/quickstart_epfl.py --raw path/to/epfl-workbooks --workdir run-local

The source directory must contain Measurement_data.xlsx and Session_data.xlsx
and must be separate from the new run directory. The pipeline does not edit
the workbooks. Choose a fresh work directory or reuse a completed quickstart;
it refuses unrelated nonempty directories.

## What the command does

1. Acquires the fixed public files when requested.
2. Streams the measurement sheets once, retaining active-session rows and
   recorded zeros while dropping empty calendar placeholders.
3. Joins measurement-local and session-global IDs by plug and arrival, with
   the recorded departure confirming the endpoint.
4. Constructs the original ten-minute-history, thirty-minute-forecast task.
5. Persists the last observed actual power and estimates a training-only
   state-frequency probability, then exports and reloads a submission.
6. Scores device power/energy and positive-power probability through
   OperationalForecastTask.

First-run workbook reading is the main preparation cost and may take several
minutes. Repeating the same command returns the completed run without
redownloading or rebuilding. An interrupted download can resume its recorded
files. An incomplete extraction directory requires inspection or a new run
directory; existing outputs are not silently replaced.

The source has 1,878 sessions and 61,816 active measurement rows in the
acquired workbooks. The task has 946 training, 86 validation and 739 later
sessions. Later evaluation is on 100 dates. These counts describe the
specific source release and eligibility, not a large multi-site charging fleet.

## Outputs

- source/: downloaded workbooks and original notices, only with --download.
- cache/: compact source rows, confirmed endpoint support and ID crosswalk.
- dataset/: named splits, case metadata and the two logged target arrays.
- example/: the model-neutral submission and power/probability score files.
- quickstart_receipt.json: case support, scores, dependency versions and timing.

The primary target is actual delivered power. Positive probability concerns a
recorded power bin greater than zero. A zero after confirmed session departure
is a known same-session outcome; an interior gap is not filled with zero.
Future setpoints, actual departure, final energy and departure SoC are not
inputs. Timestamps stay on the original recorded clock.

## Submit another forecaster

Save the following API usage in your forecasting script (replacing the model
placeholder with your implementation), and run it with
`python scripts/run_model.py my_forecaster.py` from the runtime directory:

    from field_control import OperationalForecastTask
    task = OperationalForecastTask("run/dataset", "EPFL")
    train = task.indices("train")
    later = task.indices("later")
    history = task.inputs(train, condition="full")
    targets = task.targets(train)
    # Fit a model on history/targets; predict task.inputs(later).
    # prediction_kw must have shape (len(later), 6).
    task.submission("forecast.npz", split="later", prediction_kw=prediction_kw)

The launcher supplies the project import path. Evaluate that submission with:

    python scripts/example_operational_submission.py --source EPFL --data run/dataset --split later --submission forecast.npz --output evaluation

The evaluator accepts power-only, probability-only or both outputs. It does
not fabricate a probability from a point forecast. EPFL has device-only
evaluation; no group is formed from its two plugs.

## Scope and release

On September 8, 2026, this workflow was run from a separately extracted
25-file runtime in a new Python 3.12 environment, with the source files
downloaded again from the fixed author revision. It reconstructed all 1,771
cases and the same inputs, targets and chronological metadata within normal
numeric tolerance, and recovered all six example score rows. The 24 packaged
tests passed. Acquisition through scoring took about 29 seconds on that run;
network and hardware differences change this time. The completed-run reuse
path also worked without repeating extraction.

This was an author-run isolated local reconstruction. The public package now
includes the prepared packet under `data/epfl/`; source acquisition above is
available for rebuilding it. The packet retains the DESL/EPFL MIT notice.
Original project code is covered separately by the repository's MIT license.

Tests can be run with:

    python -m unittest discover -s tests -v
