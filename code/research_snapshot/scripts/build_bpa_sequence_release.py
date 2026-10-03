"""Build model-neutral histories and standard future windows from public BPA raw.

Separate output from the earlier diagnostic; no prior labels/scores are copied.
"""
import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import time
import run_local
import numpy as np
import pandas as pd
from build_bpa_command_pilot import read_season, command_onsets, segment_mean, FIELDS, ROOT

OUT = ROOT/'results/benchmark_bpa_sequences_v1'
EXTRA_FIELDS = dict(FIELDS, oper_state=('oper_state', np.float32, np.nan),
                    running=('running', np.float32, np.nan))


def build(season):
    """Compatibility entry for the original research output path."""
    from field_control.bpa_preparation import build_season
    source = ROOT/'results/benchmark_source_feasibility_v1/bpa_cta2045'
    return build_season(source, OUT, season)


def main():
    parser = argparse.ArgumentParser();parser.add_argument('--season', choices=['winter','spring','summer'])
    args = parser.parse_args()
    receipts = [build(s) for s in ([args.season] if args.season else ['winter','spring','summer'])]
    print(json.dumps({'completed_seasons': [r['season'] for r in receipts]}), flush=True)


if __name__ == '__main__':
    main()
