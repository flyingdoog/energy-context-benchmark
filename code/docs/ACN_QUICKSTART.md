# ACN original-session-to-score quickstart

Acquire the fixed original ACN collection and its calendar manifest, then
construct the hourly, shared-clock forecasting task. Preparation needs no
prepared packet, trained model or historical analysis output.

## Run

Use the environment described in the runtime README. First generate the source
plan and complete calendar manifest:

    python scripts/acquire_acn_source.py --output acn-source

The default reads the fixed GitHub revision's directory metadata and README;
it does not download session recordings. The public source is
[Tongxin Li's ACN-Data-Static release at abb7cf15cc7108913e965d4375cc7270d532b65f](https://github.com/tongxin-li/ACN-Data-Static/tree/abb7cf15cc7108913e965d4375cc7270d532b65f).
No ACN API token is needed for this author-linked static release.

An optional bounded access check downloads each site/split's first and last
session file (at most sixteen files):

    python scripts/acquire_acn_source.py --output acn-source --sample --workers 2

This sample is only an acquisition check. It is incomplete and cannot be used
to prepare or evaluate the benchmark. Fetch the full collection explicitly
before running the forecasting quickstart:

    python scripts/acquire_acn_source.py --output acn-source --download --workers 4
    python scripts/quickstart_acn.py --source acn-source --workdir run-acn --workers 4

`--sample` and `--download` are mutually exclusive. Download workers range from
one to four, with four by default. Repeated acquisitions resume existing work,
so a sample can be followed by the full download in the same source directory.
Unfinished selected downloads return a nonzero exit code. `plan_receipt.json`
records the plan; `receipt.json` and `acquisition.csv` record acquisition status.

If a temporary GitHub API failure interrupts directory indexing, optionally
add `--index-source path/to/github-tree-json` to any acquisition command to
reuse existing provider tree responses. Only indexes matching the fixed
revision's directory chain are accepted. This option is not required: the
default obtains indexes from the public API. Index reuse changes neither the
complete manifest nor the selected session population.

If the original files and complete manifest are already available, start with:

    python scripts/quickstart_acn.py --source path/to/acn-source --workdir run-acn --workers 4

The source directory contains:

    manifest.csv
    sessions/
        <original session>.csv.gz

The manifest needs these columns: site, split, station_id, connection_utc,
filename, query_start_local, query_end_exclusive_local. The last two specify
inclusive/exclusive calendar bounds in America/Los_Angeles; connection_utc
carries its original timezone. Session contents remain unmodified.

The supported sites are caltech and office001. The original comparison uses
April–June 2019 for training, July 2019 for validation, October 2019 and January
2020 for later evaluation. Acquisition includes a 14-day lead-in before each
block so ongoing sessions can contribute queries. The full source collection
has 7,761 unique files and 8,246 session/calendar contexts; one source session
may be relevant to adjacent blocks.

Read the author-linked [static source](https://github.com/tongxin-li/ACN-Data-Static)
and the [official ACN Data access information](https://ev.caltech.edu/dataset)
before acquiring data. The static repository has no standalone data license;
it directs readers to official terms before redistribution and requests the
original dataset citation. A fixed public revision and local acquisition do
not authorize redistribution of source recordings or derived evaluation data.

## Task and examples

Four CPU preparation workers are the default; one to four are supported.
No model library or GPU is used. Workers have owned logs and a 15-minute
preparation bound. A completed run is reused; unrelated nonempty directories
and missing source sessions are rejected. Existing source files are not edited.

The original task rules are unchanged: sixty closed history minutes, hourly
queries, ten future 15-minute power means, source age at most ten seconds,
and at least 24 observed minutes among the latest thirty.

A future gap remains missing. Only an explicit terminal UNPLUGGED record
supports later zero power for that same departed session. Group membership
is defined from query-available history; incomplete future labels exclude
the entire group, not just the missing member. Groups need at least two
distinct stations. Future arrivals and unobserved station demand are outside
this task.

The example persists current power and uses a training-only state-frequency
probability. It fits no ML model. Both sites' autumn and winter outputs are
scored on device power/energy, positive-power probability and complete-cohort
power/expected positive-device count.

Expected supported case counts for the original collection:

| Site | Train | Validation | Autumn | Winter |
| --- | ---: | ---: | ---: | ---: |
| Caltech | 16,167 | 4,288 | 4,858 | 3,983 |
| Office | 807 | 344 | 749 | 796 |

Outputs are dataset/, example/<site>_<split>/, and quickstart_receipt.json.
The packet contains query_population.csv.gz as well as eligible cases so
evaluation can preserve the complete query-known group requirement.

## Another model

Put this API usage in your model script and run it with
`python scripts/run_model.py my_forecaster.py` from the runtime directory.
Use the environment's Python described in the README. The launcher supplies
the local `field_control` import; no global installation is needed.

    from field_control import OperationalForecastTask
    task = OperationalForecastTask("run-acn/dataset", "ACN", site="caltech")
    train = task.indices("train")
    winter = task.indices("winter")
    inputs, targets = task.inputs(train), task.targets(train)
    # Fit using training inputs and targets; choose settings using validation.
    # Forecast shape: (len(winter), 10), in kW.
    task.submission("forecast.npz", split="winter", prediction_kw=forecast)

Evaluate an existing submission:

    python scripts/example_operational_submission.py --source ACN --data run-acn/dataset --site caltech --split winter --submission forecast.npz --output evaluation

User declarations from the optional ACN API are a separate supplementary
input track and are not fabricated from these static session files.

## Reconstruction status

The fixed-source acquisition entry was tested from a separately extracted
41-file runtime on September 8. All 8,246 manifest contexts match the original
collection. Sixteen newly downloaded files (578,924 bytes) parsed successfully;
repeating the command reused them without downloads. All 70 packaged tests pass.
The large provider directory returned HTTP 502 during the fresh index attempt,
so the full manifest check reused original same-revision provider indexes via
the explicit optional index input. This was not a new full-source acquisition.
This runtime includes the [source attribution and access notes](SOURCE_NOTES.md).

The extracted 32-file runtime was run in a separately installed Python 3.12
environment on September 8, 2026. All 31,992 labeled cases and 33,231
query-known records were recovered from the existing original files, with the
same inputs, targets and metadata. Four examples recovered all 72 power,
probability and group score rows within ordinary numeric tolerance.
Preparation and scoring took about 116 seconds with four CPU workers.
All 37 packaged tests passed, as did nine original ACN protocol regression
tests. A repeated command reused its completed outputs.

This is an author-run isolated local reconstruction. It reused previously
acquired source files and does not establish a full download through the new
acquisition command or an independent third-party reproduction.
