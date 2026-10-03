"""Compatibility entry for the original pilot and expanded ACN preparation."""
import argparse
import json
from pathlib import Path
import run_local
import pandas as pd
from field_control.acn_preparation import build_acn_packet, GROUPS

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/acn_static_pilot_v1'
DEST = ROOT / 'results/acn_fleet_pilot_v1'


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--expanded', action='store_true')
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    args = parser.parse_args(argv)
    source = ROOT / ('data/acn_static_months_v1' if args.expanded else 'data/acn_static_pilot_v1')
    destination = ROOT / ('results/acn_fleet_months_v1' if args.expanded else 'results/acn_fleet_pilot_v1')
    if args.shards > 1:
        destination = destination / 'shards' / f'{args.shard_index:02d}'
    acquisition = json.loads((source / 'receipt.json').read_text())
    if not acquisition.get('completed'):
        raise ValueError('Source acquisition is incomplete')
    # Historical pilot intervals predate the full-month protocol. Preserve
    # them only in this compatibility entry; the new runtime uses explicit bounds.
    manifest = pd.read_csv(source / 'manifest.csv')
    return build_acn_packet(source, destination, args.shards, args.shard_index, manifest)


if __name__ == '__main__':
    main()
