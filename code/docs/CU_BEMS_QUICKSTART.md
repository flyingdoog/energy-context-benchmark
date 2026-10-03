# CU-BEMS original-CSV-to-score quickstart

Prepare air-conditioner forecasts directly from one original 2019 floor CSV.
The same runtime supports BPA, ACN and EPFL. Preparation and these examples
need NumPy and pandas, without model libraries or earlier experiment outputs.

## Source and run

Obtain the original `2019FloorN.csv` from the
[CU-BEMS publisher release, version 6](https://doi.org/10.6084/m9.figshare.11726517.v6).
Keep its CC BY 4.0 attribution. This runtime does not download or bundle the
source. See [source notes](SOURCE_NOTES.md) for credit and transformations.

Use the Python environment described in the runtime [README](../README.md).
On Windows, `python` below means `.\.venv\Scripts\python.exe`; on Linux/macOS,
use `.venv/bin/python`. Replace the CSV path with your acquired original:

```text
python scripts/prepare_cu_bems.py --csv raw/2019Floor3.csv --floor 3 --output prepared/floor3
python scripts/example_operational_submission.py --source CU-BEMS --site floor3 --data prepared/floor3 --split later_1 --output example/floor3/later_1
python scripts/example_operational_submission.py --source CU-BEMS --site floor3 --data prepared/floor3 --split later_2 --output example/floor3/later_2
```

Choose a new or empty preparation directory. The original CSV is not modified.
A publisher filename must agree with `--floor`; for a renamed CSV, supply its
correct floor. Changing floors requires changing the input, output and `--site`.
Floor 6 supports only the first later period. Floor 1 has no thermal input.

## Inputs, targets and scoring

Each input contains the last 60 one-minute power observations, calendar and
device/zone identity. Where supported, zone temperature and relative humidity
have explicit missingness masks. No future sensor value is exposed. Targets
are ten 15-minute power means starting at the query. The scored spans are 30,
90 and 150 minutes. Power stays in kW and integrated energy in kWh. Source
clock labels are retained without an assumed timezone conversion.

The named splits use April-June for training, July-August for validation,
September-October for `later_1` and November-December for `later_2`. The example
persists current power and estimates positive-power probabilities using
training-only state frequencies. It performs no ML forecasting fit.

Device scores include all supported cases. Floor/zone totals require every
member of the original group to have a supported target. Missing members do
not redefine a smaller group. Temperature is in degC and humidity in RH%.
Unavailable thermal observations are missing, not zero.

Each example exports a portable NPZ submission, a run receipt, and JSON scores
for power, activity probability and supported group activity. The preparation
receipt records device definitions, original groups and split support.

## Use another model

Save the following API usage in your model script and run it through
`python scripts/run_model.py my_forecaster.py`. Replace the model placeholder
with your implementation and choose settings on validation.

```python
from field_control import OperationalForecastTask
task = OperationalForecastTask('prepared/floor3', 'CU-BEMS', site='floor3')
train, later = task.indices('train'), task.indices('later_1')
model.fit(task.inputs(train, condition='full'), task.targets(train))
prediction = model.predict(task.inputs(later, condition='full'))
task.submission('forecast.npz', split='later_1', prediction_kw=prediction)
task.base.close()
```

Use `condition='power'` for Floor 1. On other supported floors, `power_state`
and `full` are aliases for the same thermal-input condition because the source
does not supply control commands. `task.contract['available_conditions']`
reports the available choices. Forecast arrays contain ten columns and every
case in the selected named split.

```text
python scripts/example_operational_submission.py --source CU-BEMS --site floor3 --data prepared/floor3 --split later_1 --submission forecast.npz --output evaluation/floor3
```

## Local reconstruction reference

The original Floor 3 file yields 28,835 training, 19,711 validation, 19,753
first-period and 18,597 second-period cases. Each later period has 61 dates.
The earlier author-run source reconstruction and shared scorer agree on 60
native score rows in the author-run check. Prepared data for all seven floors
are now supplied in the separate public data asset under CC BY 4.0. Seven
floors belong to one Bangkok building, not seven datasets.
