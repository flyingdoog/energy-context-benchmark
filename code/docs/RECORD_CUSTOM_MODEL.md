# Compare record information using your own predictor

The record comparison accepts a model with `fit(X, Y)` and `predict(X)`.
The runner constructs power-only, paired and reassigned inputs, fits five
model instances, reuses the paired model for all three fixed-model
reassignments, and scores complete native splits. Your model does not need
to implement mappings, group aggregation or uncertainty estimation.

## Try the interface without source data

    python scripts/example_record_model.py --output examples/synthetic_record_comparison.json

This small NumPy example creates synthetic query groups, includes records
without supervised targets as input-only donors, and fits a linear predictor.
It produces eight prediction streams for each of three evaluation periods.
These synthetic scores demonstrate API use; they are not benchmark results.
No downloads, model checkpoints, optional ML libraries or GPU are required.

## Supply a model

Place a function with this signature in your Python file:

```python
def make_model(seed=2027):
    return MyForecaster(seed=seed)
```

Each call must return a fresh object. It needs:

- `fit(X, Y)`: fit this object on the supplied training data; the return value
  is ignored. `Y` has shape `(training_cases, 10)` and is measured in kW.
- `predict(X)`: return a finite NumPy-compatible `(cases, 10)` array in kW.
  Prediction must leave the fitted state unchanged. For a neural model, enter
  inference/evaluation mode yourself and return CPU arrays.

The constructor receives the same model seed for all five fits. Mapping
seeds change device-record correspondence, not initialization. The runner
does no hyperparameter search or checkpoint selection. Choose a recipe using
training/validation data before reporting the later-period comparison.
`--clip-nonnegative` is an explicit option, disabled by default; use the same
output transformation for all arms.

Inputs use the existing **tabular** default-reference encoding: 1,445 columns
for ACN and 4,064 for BPA. NaNs and observation masks are retained. Your model
must handle these inputs; the runner does not silently normalize, impute or
reshape them into sequences. This entry therefore supports compatible
predictors and explicit wrappers, not every forecasting architecture unchanged.

## Prepare the input views once

First use the source quickstart to prepare the dataset. Then use the
[record-comparison preparation](RECORD_COMPARISON_RUNTIME.md):

    python scripts/run_record_comparison.py --data run-acn/dataset --source ACN --site caltech --original-source acn-source --output records/caltech_inputs --prepare-only

The input cache is `records/caltech_inputs/inputs`. A completed full Tree run
also supplies this cache. No saved prediction or fitted model is required.
ACN's original source files recover query-known histories with incomplete
future targets. Their records remain donors; labeled recipients and scoring
groups retain the native task support.

## Preview and run

    python scripts/run_custom_record_comparison.py --data run-acn/dataset --source ACN --site caltech --inputs-cache records/caltech_inputs/inputs

Preview reads support metadata only: it neither imports your model script,
creates output files nor fits a model. To execute:

    python scripts/run_custom_record_comparison.py --data run-acn/dataset --source ACN --site caltech --inputs-cache records/caltech_inputs/inputs --model-script my_forecaster.py --model-name "My forecaster" --output runs/my_caltech --run

For BPA, use its prepared dataset and input cache, `--source BPA`, and omit
`--site`. The named validation and both later periods are always evaluated;
the runner does not provide an option to remove an unfavorable period.

Run your own trusted model code only. `--model-script` executes that file to
load its factory. Install its dependencies in your project environment and
manage its device choice yourself. Five fitted instances remain alive for
comparison; model memory and fit costs are model-dependent. Prediction is
batched (`--batch-size 4096` by default). An arbitrary model's weights are
not automatically serialized or resumed; choose a new output directory for
a new run. The separate default Tree entry retains its existing resume path.

## Results

After this run completes, the [group-level diagnostic](RECORD_LEVEL_DIAGNOSTIC.md)
can read its prediction packets and compare original errors with a fixed
validation-only group offset. It requires no further model calls or training.

The output contains the model recipe, progress, 24 portable prediction packets,
native device/group scores, mapping-averaged errors, paired calendar-week
uncertainty and `comparison.md`. The report identifies the supplied model
and shows MSE / MAE for every period and span. Reassignment averages errors,
not forecasts. Evaluation views provide input and case identities, not labels
to the fit/predict callbacks; the native scorer reads labels separately.

The complete-run flag denotes full training support. A user model's run is
not automatically a reproduction of the paper's default Tree or a new paper
result. The synthetic demo is labeled separately and has no native-data claim.

## Use the API directly

```python
from field_control.record_comparison import (
    RecordInputView, fit_record_models, predict_record_models,
)

train = RecordInputView.from_cache(cache, 'train')
models = fit_record_models(train, make_model)
later = RecordInputView.from_cache(cache, 'autumn')
for method, prediction in predict_record_models(models, later):
    # prediction rows follow later.case_ids; pass them to the native task.
    task.submission(output_for(method), split='autumn', prediction_kw=prediction)
```

Run standalone scripts through `scripts/run_model.py` to enable local imports,
as shown in the main quickstart. Original dataset licenses, source acquisition
and public code distribution remain separate from running the comparison.
