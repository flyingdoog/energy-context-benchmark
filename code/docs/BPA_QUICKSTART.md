# BPA original-archive-to-score quickstart

Prepare the common BPA forecasting task from the original seasonal archives,
then score spring and summer examples. No previous training output, prepared
packet or source-dimension metadata is required.

## Run

Install `requirements-operational.txt` in the project-local environment
described in the runtime README, then run:

    python scripts/quickstart_bpa.py --source path/to/bpa-source --workdir run-bpa --workers 2

The source directory needs only these unmodified files:

    winter2018.zip
    spring2018.zip
    summer2018.zip

Acquire them from the [original BPA project page](https://www.bpa.gov/energy-and-services/conservation/emerging-technologies/portfolio/cta-2045-water-heaters).
This command starts with existing archives; it does not download data. BPA's
data redistribution terms remain unresolved. Credit BPA and the original
CTA-2045 Water Heater Demonstration Report when using the records.

Two CPU preparation workers are the default; use `--workers 1` for lower peak
memory. The `inflate64` dependency reads the provider's summer ZIP compression.
Preparation derives the source dimensions itself. Keep `--workdir` separate
from the source directory: neither may contain the other. An unrelated
nonempty work directory is rejected. Repeating the same command reuses a
completed run; keep the same source and worker setting.

## Task and outputs

The query is the original changed-command report plus 30 minutes. Its ten
15-minute target bins use reports at query+1 through query+150 minutes
(original origin+31 through origin+180). Targets remain reported device power
in kW with the source observation/eligibility rules. Winter supplies training
and validation; spring and summer retain their native later-period splits.

The examples persist query-time power and estimate activity probabilities
from winter training-only state frequencies. They fit no ML forecasting model
and use no GPU. Each season is scored for device power/energy, positive-power
probability, group power and expected positive-device counts. BPA groups contain
the original label-eligible synchronous devices and require at least ten
members; they do not represent the complete utility fleet.

Outputs are `dataset/`, `example/spring/`, `example/summer/`,
`workflow.json` and `quickstart_receipt.json`. Each example includes a portable
`example_submission.npz`, a receipt and the power, probability and group-activity
score JSON files. The quickstart receipt records preparation receipts, supported
case counts, scores, runtime versions, timing and CPU/no-ML-fit status.

Evaluate another model's spring submission with:

    python scripts/example_operational_submission.py --source BPA --data run-bpa/dataset --season spring --split spring --submission forecast.npz --output evaluation-spring

The corresponding task is
`OperationalForecastTask(Path('run-bpa/dataset'), 'BPA', season='spring')`.
In a custom script import `Path` from `pathlib` and `OperationalForecastTask`
from `field_control`, then launch it with
`python scripts/run_model.py my_forecaster.py`. Use the environment's Python
from the runtime README; the launcher supplies the local API import.
Submit `(number_of_spring_cases, 10)` power predictions and/or positive-power
probabilities through its common submission interface. Future commands,
states and outcomes are not query inputs.

## Validation scope

The September 8 author-run reconstruction used the three original archives in
a separate local installation. It recovered 174,436 cases, the original split
and 36 example score rows in 149.015 s with two CPU workers. All 48 packaged tests
passed, and a repeated command reused the completed run. This is local
reconstruction, not external reproduction. Detailed records are maintained
with the research run; no prior result files are needed to use this command.
