# Check level offsets in a four-arm prediction run

Use this entry after the default or user-model record comparison. It reads
the saved power predictions and prepared native targets, then separates
average group-level error from variation across groups and dates. It also
applies one validation-fitted offset per lead to each arm and mapping.
No model is loaded, trained or queried again.

The question is whether a reassignment penalty remains after the same simple
group-level correction is available to every arm. The output retains original
and corrected absolute MSE/MAE, so changes in the paired reference stay visible.

## Try a source-free example

    python scripts/example_record_level.py --output examples/group_level

This creates a small synthetic example, explicitly labeled as such. One date
has one query group and another has three; group sizes range from two to four.
The validation residuals have a nonzero mean. Later periods use the same
offsets without refitting. The example needs only the base runtime dependencies,
not source data, model weights, an ML library or a GPU.

## Preview a completed native run

Prepare the source and finish the comparison using the
[default-model guide](RECORD_COMPARISON_RUNTIME.md) or the
[user-model guide](RECORD_CUSTOM_MODEL.md). Then:

    python scripts/run_record_level_diagnostic.py --run-directory runs/my_caltech --data run-acn/dataset --source ACN --site caltech

Preview checks the run description and expected packet filenames without
reading their prediction arrays, loading model code or creating output.
For Office use `--site office001`. For BPA use `--source BPA` and omit `--site`.

The supplied run needs `protocol.json`, a successful `complete.json`, and the
24 `predictions/<period>/<arm>.npz` packets produced by the comparison runner.
It does not need its training input cache, model files, historical score JSON,
original download directory or the authors' project workspace. `--data`
explicitly locates the prepared dataset, allowing an existing run to be moved.
The original model script is not executed, even for a user-model run.

## Compute the control

    python scripts/run_record_level_diagnostic.py --run-directory runs/my_caltech --data run-acn/dataset --source ACN --site caltech --output analyses/my_caltech_level --run

Use a new output directory, separate from the prepared data and prediction
run. Existing outputs and device submissions are left unchanged. The entry
uses validation and both later periods: spring/summer on BPA, autumn/winter
on ACN. It retains both native group scales and the 30/90/150-minute spans.

Packet reading checks case identities, target times and finite ten-lead
power predictions against the native dataset. Group scoring keeps complete
original memberships; an incomplete group is excluded as a whole. A period
without supported groups cannot supply this diagnostic. Mismatched or missing
packets fail with an error rather than creating a partial four-arm comparison.

## Read the results

`comparison.md` contains the group-total summary, every native span and group
scale, the level/variation decomposition, and paired later-period intervals.
All error cells explicitly show MSE / MAE in squared kW / kW.

- Original: the supplied predictions, scored with the original date weights.
- Offset: subtract the signed validation mean residual for each lead, group
  scale and arm/mapping. Apply it unchanged later, without clipping or search.
- The MSE decomposition squares the mean separately for each lead and mapping
  before averaging. It is a decomposition of group error, not an estimate of
  centered cross-device covariance.

Offset validation rows are calibration in-sample results. Later rows evaluate
transfer. The control adjusts group outputs only: it does not construct a
coherent corrected set of device forecasts or replace the original baseline.
A smaller penalty need not mean every forecast improved; its paired reference
also receives its own validation-fitted correction.

The structured files retain daily mapping-specific and averaged errors,
per-lead and span decompositions, absolute errors and paired contrasts.
`validation_offsets.json` records the actual intercepts. Calendar-week
intervals are pointwise, conditional on the trained predictors and the fitted
offsets, and omit offset-estimation uncertainty. Fewer than two weeks yields
an unavailable interval. No later-period labels choose a correction.

## Use the numerical API

`field_control.record_level_diagnostic.analyze_groups` accepts records of
`(period, raw_arm, groups)`, where `groups` is the return value of
`field_control.group_error_level.group_residuals`. Supply every mapping stream
and an explicit period list starting with validation. The implementation
matches group identities across arms and applies the existing date-weighted
scoring. `example_record_level.py` is a complete small usage example.

The returned tables are ordinary pandas frames; `write_result` writes the
same report and JSON files as the native entry. Synthetic inputs are examples,
not benchmark evidence. A real user's run reports that user's predictor and
does not automatically reproduce the paper's default Tree results.
