# Device-record association: runnable four-arm comparison

This entry trains the default references from prepared ACN or BPA data. It
does not load historical prediction files or privately fitted models. It
compares power-only, correctly paired records, evaluation-only reassignment
of the paired model, and models retrained on the same reassignment.

## Dependencies and inputs

Use the private virtual environment from the main quickstart. Install the
optional tree dependency using that environment's Python:

    python -m pip install -r requirements-record.txt

The reported recipe uses LightGBM 4.7.0, 100 trees, 31 leaves, squared-error
loss and model seed 2027. Preparation and workload preview do not import
LightGBM. No GPU, checkpoint download or parameter search is used.

Prepare the original files using the ACN or BPA quickstart first. For ACN,
keep the original source directory: its manifest and sessions reconstruct
query-known histories whose future targets are incomplete. They remain
eligible donors even though they cannot supply a supervised target. The
runner refuses to silently drop them if the source is absent. BPA retains
its original label-eligible population and needs only the prepared root.

## Preview before execution

    python scripts/run_record_comparison.py --data run-acn/dataset --source ACN --site caltech
    python scripts/run_record_comparison.py --data run-bpa/dataset --source BPA

By default the entry only prints case counts, feature-cache size and the
planned 50 lead-model fits. It does not create outputs, recover inputs or
train models. Each complete run includes training, validation and both later
periods, with all three native prediction spans.

## Run the comparison

    python scripts/run_record_comparison.py --data run-acn/dataset --source ACN --site caltech --original-source acn-source --output runs/caltech_records --run
    python scripts/run_record_comparison.py --data run-acn/dataset --source ACN --site office001 --original-source acn-source --output runs/office_records --run
    python scripts/run_record_comparison.py --data run-bpa/dataset --source BPA --output runs/bpa_records --run

Two lead workers with two CPU threads each are the defaults. `--workers` and
`--threads` control execution, not accuracy. The original BPA feature cache
occupies about 2.8 GB; check free memory and the printed workload before
starting it. Sources/prepared datasets must be outside the output directory.

For a pipeline check, add `--smoke-train-cases 256` and choose a separate output
folder. This uses a smaller training subset but retains complete evaluation
splits. Its receipt explicitly marks the result as a smoke, ineligible for
paper accuracy comparisons. It is not parameter tuning.

`--prepare-only` builds query inputs and mappings without importing LightGBM.
`--inputs-cache path` lets a later full run reuse the input cache from that
step or a smoke. It only accepts caches from this entry with matching source
and recipe settings. Repeating a run resumes saved heads and submissions;
use a new output directory for different settings or data.

## What changes in each arm

- Power only: state/control columns and their masks are constant zero.
- Paired: the same default model recipe with original operational records.
- Fixed: the fitted paired model receives reassigned auxiliary blocks at
  validation/evaluation only. No new fit is used for this arm.
- Retrained: separate default fits receive reassigned records in training and
  the corresponding reassignment in validation/evaluation.

Three deterministic mappings use seeds 2027, 2028 and 2029. Values and masks
move together within the original query group; the receiving device retains
its power history, calendar and BPA historical day/week profiles. The same
mapping is used for the fixed and retrained arms. Complete groups and target
intervals do not change. Mapping construction does not read future labels.

The runner saves eight error streams: power-only, paired, three fixed-model
mappings and three retrained mappings. It averages mapping **errors**, not
predictions, to make the four-arm summary. Incomplete future labels remove
the original ACN group from group scoring, never just its missing member.

## Outputs

Generate a readable comparison after a full run, without importing a model
library or doing any further training:

    python scripts/summarize_record_comparison.py --run runs/caltech_records --output runs/caltech_records/comparison.md

The report keeps every period and native span, puts MSE / MAE explicitly in
each four-arm error cell, and includes average-span uncertainty for both
metrics. It labels training-subset smokes. Omit `--output` to print the report;
an existing output file is not overwritten.

- `protocol.json`, `progress.jsonl`, `complete.json`: actual settings,
  progress, versions and whether the run is full or a training-subset smoke.
- `inputs/`: local derived input cache, unless a separate cache was supplied.
- `models/`: 50 default lead models, without an accuracy search.
- `predictions/<period>/`: portable complete-split submissions for all arms.
- `raw_daily.json`: mapping-specific device/group MSE, MAE and energy errors.
- `daily.json`, `summary.json`: four-arm native-span errors.
- `effects.json`: fixed/retrained penalties and recovery, at native spans and
  span average, separately for MSE and MAE. Intervals resample 1,500 paired
  local calendar weeks and condition on the fitted models. Fewer than two
  weeks or a zero reference error yields a missing interval, not an invented
  confidence bound.

These are controlled predictor comparisons under specified reassignment,
not estimates of the natural prevalence or financial cost of record errors.
They establish neither a new feature-importance estimator nor an automatic
model-selection rule.

The local runtime archive contains code only. Source-data rights, model
outputs, author code licensing and external publication are separate matters.
