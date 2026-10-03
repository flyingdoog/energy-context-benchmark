# Native operational inputs for Chronos-2

September 9, 2026. Executable adapter: src/field_control/chronos_inputs.py.
It accepts the shared task input view plus existing query timestamps, without
fitting, scoring or accessing future target arrays.

## Time and output mapping

| Source | Model context | Model prediction | Shared output |
| --- | --- | --- | --- |
| BPA | 96 historical 15-minute report means | 10 ordered 15-minute bins | 10 bins, unchanged |
| ACN | 60 one-minute means | 150 one-minute points | 10 means of 15 predicted points |
| EPFL | 10 one-minute records | 30 one-minute points | 6 means of 5 predicted points |
| CU-BEMS | 60 one-minute records | 150 one-minute points | 10 means of 15 predicted points |

BPA's native last historical block covers query-15 through query-1; the first
future block covers query+1 through query+15. The one-report boundary gap is
retained. Ordered bin steps are used by the model, with the original calendar
features; no label is shifted to pretend the blocks are contiguous. Query
power and state at offset zero enter as separate query-only side channels.

The installed predict_quantiles API returns the median in its point/mean slot.
For the minute-resolution sources, the submitted forecast is the arithmetic
mean of these minute-level median points. This is not necessarily the median
or mean of the aggregate predictive distribution. Saved marginal quantiles
must not be summed or averaged and called group/bin prediction intervals.

## Field mapping

| Source | Past-only operational fields | Known prediction-time inputs |
| --- | --- | --- |
| BPA | Storage state, categorical operating/running/opt-out codes, observed control code/parameter, query reports, masks | Calendar; lead-aligned day/week profiles already observed before query |
| ACN | Current, voltage, cumulative delivered energy, operating-state fractions, pilot and masks | Calendar and elapsed time from the known connection |
| EPFL | SoC, capacity fields, energy, requested/setpoint power, controlled indicator | Native calendar |
| CU-BEMS | Linked zone temperature/humidity and masks | Calendar and fixed device/zone/floor identity |

All conditions keep power and calendar information. Power-state adds the
source's state fields; full adds its observed control fields. CU-BEMS full and
power-state are identical and share their smoke prediction file content.
ACN declarations are a separate optional task interface and are not in this
common full-input comparison, matching its existing main reference conditions.

Missing numeric reports become NaN and retain explicit masks/fractions.
BPA categorical codes become category strings; reserved code 255 is retained
when observed, not treated as a large numeric state or automatically missing.
No future realized device state, control or departure is supplied.

Query-only values occupy one last context slot in separate channels, with
earlier slots missing; they are not broadcast as observed historical series.
This is a packing convention for an API without a separate query-vector input.
The extra two supplied day/week profile bins are retained as query-only inputs.
The first ten are known future-position covariates with unavailable past
covariate slots left missing. Fixed identity strings describe metadata, not
measured state trajectories. Calendar trajectories are deterministically
reconstructed from native query clocks for ACN/EPFL/CU-BEMS; this preserves
query-available information but is not identical to repeating the tabular
reference's four/five query-time calendar columns.

## Important model boundary

Preserving a value in the input item does not imply that the pretrained
encoder retains it. Installed Chronos 2.3.1 uses per-variate InstanceNorm:
subtract the observed-history mean, divide by its standard deviation, then
optionally apply arcsinh. Target output units are restored using the target's
statistics; a past-only covariate's mean is not an extra target-side feature.
A singleton numeric query channel therefore loses its numerical level.
Constant categorical metadata can likewise be uninformative after the
default per-item target encoding and normalization.

The adapter preserves the source fields; the default encoder may discard
some of their numerical levels. Its normalization and category handling
remain unchanged in this reference. Model/input comparisons should retain
this distinction between available information and its internal encoding.
