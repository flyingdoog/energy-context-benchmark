"""Prepare a CU-BEMS publisher CSV without internal experiment directories."""
import argparse
import json
from pathlib import Path
import sys

# A source checkout works from any cwd; numpy and pandas come from the caller's environment.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from field_control.cu_bems_preparation import prepare_cu_bems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, required=True, help='Original 2019FloorN.csv')
    parser.add_argument('--floor', type=int, choices=range(1, 8), required=True)
    parser.add_argument('--output', type=Path, required=True, help='New or empty prepared-packet directory')
    args = parser.parse_args(argv)
    receipt = prepare_cu_bems(args.csv, args.output, floor=args.floor)
    print(json.dumps(dict(completed=True, floor=args.floor, cases=receipt['cases'],
        seconds=receipt['seconds'], support=receipt['support'], output=str(args.output),
        model_libraries_loaded=[x for x in ('torch', 'lightgbm', 'sklearn') if x in sys.modules],
        new_ml_fits=0, gpu_used=False)), flush=True)


if __name__ == '__main__':
    main()
