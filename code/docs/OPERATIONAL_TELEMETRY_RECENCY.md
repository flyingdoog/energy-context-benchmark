# Optional recent auxiliary-report blackout

Use the same device queries, power histories and targets while withholding
recent auxiliary state/control. The common task supports:

```python
batch = task.inputs(indices, condition='full', auxiliary_blackout_minutes=5)
native = batch['native_inputs']
```

For a positive value D, auxiliary records timestamped at or after query minus
D minutes are unavailable. Older records retain their original values and
masks. Zero is the unchanged default. This is a synthetic availability test,
not an observed communications delay; queries and target times do not move.
Five minutes is the completed cross-source reference condition. Supported
integer settings are 0–60 for BPA/ACN and 0–10 for EPFL; other settings have not
been evaluated by the reference comparison.

## What changes

| Source | Withheld auxiliary values | Clock | Preserved |
|---|---|---|---|
| BPA | Past/current device state and command code/parameter | State at each 15-minute bin's final minute; snapshot at query | All power, including snapshot power; masks; day/week power profiles; calendar |
| ACN | Pilot, current/voltage/cumulative energy, operating-state fractions | Left-labelled closed minute intervals | Power history and its masks; calendar and connected-time input |
| EPFL | SoC/capacity/energy, requested power, setpoint and control flag | Native one-minute history records | Actual power history, its target-history alias and calendar |

Unknown floating values are NaN and their observed masks are false. BPA
unknown historical command codes use -1 with a false mask. EPFL's boolean
control flag uses false with a false observed mask: read both fields together;
that placeholder is not an observed lack of control. Do not impute from a
later record. Power-only calls are unchanged. With `condition='power_state'`,
the cutoff does not reintroduce control fields excluded by that condition.

Positive auxiliary views include `batch['availability']`, recording the
synthetic cutoff. The submission/scoring target schema stays unchanged. Store
the condition, cutoff and training regime with your model's run metadata.

## Inspect a real input batch without training

After the source quickstart has prepared the data, run with that environment's
Python (use `.venv/bin/python` on Linux/macOS):

```powershell
.\.venv\Scripts\python.exe scripts/example_auxiliary_blackout.py --source EPFL --data path/to/prepared/epfl --split later --blackout-minutes 5 --output runs/epfl_availability
```

For ACN, supply `--source ACN --site caltech --split winter`; for BPA, supply
`--source BPA --season spring --split spring`, with the respective prepared
data roots. By default the example inspects 16 evenly spaced cases from the
specified split and writes observed/withheld value counts. It does not load a
model library, fit a predictor, use labels or produce a forecast score.

## Train and evaluate with the intended information

For a model intended to operate with this cutoff, apply it in both training
and evaluation. The following is an adapter sketch for your own model:

```python
D = 5
train_ix = training_task.indices('train')
train_input = training_task.inputs(train_ix, auxiliary_blackout_minutes=D)
model.fit(train_input['native_inputs'], training_task.targets(train_ix))

eval_ix = task.indices(eval_split)
eval_input = task.inputs(eval_ix, auxiliary_blackout_minutes=D)
prediction = model.predict(eval_input['native_inputs'])
task.submission('prediction.npz', split=eval_split, prediction_kw=prediction)
scores = task.score(split=eval_split, prediction_kw=prediction, method='my_model')
```

For BPA, `training_task` is the winter task while `task` can be spring or
summer. A separately labelled stress test can hold a timely-trained model
fixed and withhold reports only at evaluation. Comparing that stress test with
matched training distinguishes unfamiliar missing inputs from information
that is unavailable even after adaptation. Keep the original power-only
reference, all native spans and original supported periods.
