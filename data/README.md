# Data for context-aware energy forecasting

Download the [v1.0.0 data asset](https://github.com/flyingdoog/energy-context-benchmark/releases/download/v1.0.0/context-energy-forecasting-data-v1.0.0.zip)
and extract `context-energy-forecasting-data-v1.0.0.zip` at the repository root.
It supplies the following derived forecasting packets:

| Packet | Contents | License |
| --- | --- | --- |
| `data/epfl/` | 1,771 sessions: 946 train, 86 validation, 739 later; inputs, targets, cases | DESL/EPFL MIT |
| `data/cu_bems/floor1` through `floor7` | AC history, observed thermal fields, calendar, targets, cases, preparation records | CC BY 4.0 |

The small EPFL packet is also included in the code repository. CU-BEMS arrays
belong in the separate data asset, not Git history. The paper's matched thermal
comparison uses Floors 2-5 and 7: Floor 1 has no thermal fields and Floor 6 has
no November-December evaluation. The seven floors form one building. Release
packaging does not change measurements, targets, masks or split assignments.

```sh
python code/inspect_data.py --source epfl
python code/inspect_data.py --source cu_bems
```

## Obtain or reconstruct the original sources

| Source | Access and preparation | Public artifact |
| --- | --- | --- |
| BPA | [BPA guide](../code/docs/BPA_QUICKSTART.md), seasonal ZIPs | Download/preparation code and aggregate scores; no raw/prepared records |
| ACN | [ACN guide](../code/docs/ACN_QUICKSTART.md), pinned static release | Acquisition/preparation code and aggregate scores; no raw/prepared records |
| EPFL | [EPFL guide](../code/docs/EPFL_QUICKSTART.md), pinned workbooks | Prepared packet and original MIT notice |
| CU-BEMS | [CU-BEMS guide](../code/docs/CU_BEMS_QUICKSTART.md), Figshare v6 | Seven prepared floors in the separate data asset |

The static ACN release does not contain the query-time declarations used in
Table III; see [reproduction scope](../docs/REPRODUCING.md).

List BPA or CU-BEMS download URLs without downloading:

```sh
python code/scripts/download_sources.py --source bpa --output tmp/raw-bpa
python code/scripts/download_sources.py --source cu_bems --output tmp/raw-cu
```

Add `--download` to fetch source files. CU-BEMS defaults to the five paper floors;
`--floors 1 2 3 4 5 6 7` selects all floors. Keep BPA/ACN records local and follow
provider terms. Source versions and citations are in
[SOURCE_NOTES](../code/docs/SOURCE_NOTES.md).

## Packet dictionary

EPFL: `cases.csv` contains session IDs, plugs, splits and query times;
`packet.npz` contains `case_id`, `x` (1,771 by 94) and `y` (1,771 by 6 by 2).
The two logged targets are actual delivered and requested power. The paper
uses actual power. Histories cover ten minutes and targets six five-minute bins.
`receipt.json` records support and interval definitions. Original provider
notices are in `epfl/source_notices/`.

Each CU-BEMS floor contains `cases.jsonl.gz`, `power.npy`, `calendar.npy`,
`target.npy`, and `preparation.json`; `thermal.npy` exists where the provider
supplies thermal fields. Histories cover sixty minutes and targets ten
fifteen-minute power means. Named splits are `train`, `validation`, `later_1`
and `later_2`. Missing observations remain masked, not zero-filled outcomes.
Temperature is in degrees Celsius, humidity in RH%, and power in kW.

Read named inputs, masks, units and intervals through the API:

```python
from pathlib import Path
import sys
sys.path.insert(0, str(Path('code/src').resolve()))
from field_control import OperationalForecastTask
task = OperationalForecastTask('data/epfl', 'EPFL')
rows = task.indices('train')[:2]
inputs = task.inputs(rows, condition='full')
targets_kw = task.targets(rows)
print(task.contract)
```

For CU-BEMS, use `OperationalForecastTask('data/cu_bems/floor3', 'CU-BEMS',
site='floor3')`. Conditions are `power`, `power_state`, and `full`; state/full
coincide for CU-BEMS. Use training/validation for development, and retain the
later splits for evaluation. `results/` holds aggregate scores, not input data.
