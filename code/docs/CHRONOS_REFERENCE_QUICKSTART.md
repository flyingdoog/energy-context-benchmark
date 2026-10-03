# Local Chronos-2 reference

This optional entry runs an existing local checkpoint on a complete native
split and writes a portable submission plus device/group scores. The model
adapter is shared by the four source collections. It does not download
weights, train, search hyperparameters or choose test-performing inputs.

## Setup

Use the same private virtual environment as the main runtime. Install the
optional model dependencies there if you need inference:

    python -m pip install -r requirements-chronos.txt

Use a PyTorch build appropriate for your CPU/GPU. The runner checks a requested
CUDA device and does not silently start a long CPU run if CUDA is absent.
The source preparation, input preview and scoring interfaces do not require
Chronos or PyTorch.

Obtain the standard checkpoint from the official
[Chronos project](https://github.com/amazon-science/chronos-forecasting)
separately and supply its local directory, containing config.json and weights.
The reported zero-shot baseline is amazon/chronos-2, not Chronos-2-Synth.
Model weights and their redistribution are not part of this code artifact.

## Inspect the input mapping first

Given the prepared EPFL task from the main quickstart:

    python scripts/run_chronos_reference.py --source EPFL --data prepared/epfl --split train --condition full --describe-inputs

This inspects at most two input cases, prints the native time grid and field
names, and does not load the model or produce evaluation scores.

## Predict and score a complete split

    python scripts/run_chronos_reference.py --source EPFL --data prepared/epfl --split later --condition full --checkpoint models/chronos-2 --device cuda --output runs/epfl_full_later

CPU inference is available with --device cpu. --case-batch controls native
queries in a saved packet; --variate-batch controls the model's variate budget
(default 256). These are execution controls, not accuracy tuning.

For the other sources use the corresponding prepared roots and selectors:

| Source | Selectors | Native splits |
|---|---|---|
| BPA | --source BPA --season winter | train, validation |
| BPA | --source BPA --season spring (or summer) | spring (or summer) |
| ACN | --source ACN --site caltech (or office001) | train, validation, autumn, winter |
| CU-BEMS | --source CU-BEMS --site floor1 through floor7 | train, validation, later_1; later_2 where supported |

Choose --condition power, power_state or full explicitly. CU floor 1 supports
power only. On the other CU floors, full and power_state are the same input:
reuse the state prediction instead of recomputing an extra method. Native
floor 6 has no eligible later_2 data. The runner does not invent a missing split.

Each output folder belongs to one source/site/split/condition/checkpoint.
Completed batches resume automatically when called with the same settings.
Use a new output folder for a different configuration. Outputs are:

- submission.npz: case IDs, original target offsets and point predictions.
- summary.json and daily_scores.json: native device and supported group scores.
- protocol.json and complete.json: actual run settings and completion.

The generic runner is inference-only. The reported EPFL supervised comparison
used the standard full-fine-tuning recipe (1,000 steps, learning rate 1e-6,
variate batch 256, seed 42) on its training split. An already fine-tuned local
checkpoint can use this inference entry with --model-label "Chronos-2 fine-tuned".
This guide does not claim that the generic entry reproduces that training.

## Meaning of the prediction

The tested Chronos API returns the native-step median in the point slot.
For ACN/CU, fifteen minute-level points are averaged per output bin; for EPFL,
five are averaged; BPA directly predicts the ten original 15-minute bins.
These are not necessarily conditional means or medians of aggregate loads.
Marginal quantiles must not be summed into group prediction intervals.

All future operational states and departures remain unknown to the model.
Only query-known calendar/profile/identity information is supplied at future
positions. See [the complete input mapping](CHRONOS_INPUTS.md) for masks,
query-only side channels and the one-report BPA boundary convention.

Full support is scored at once. Floor/cohort errors are computed from summed
residuals, not averages of per-device errors. Use model and input labels from
the run; do not turn the results into a test-selected input recipe.
