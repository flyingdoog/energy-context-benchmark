"""Source-free group-offset demonstration; synthetic, not benchmark evidence."""
import argparse
import json
from pathlib import Path
import run_local
import numpy as np
import pandas as pd
from field_control.group_error_level import group_residuals
from field_control.record_level_diagnostic import analyze_groups, write_result, render_report
from field_control.record_reference_scores import RAW_ARMS

PERIODS = ['validation','later_1','later_2']


def synthetic_groups():
    for index, period in enumerate(PERIODS):
        month = index+1
        # First date has one query, second has three; group sizes also differ.
        sizes = [2,3,4,2]
        dates = [f'2020-{month:02d}-01']+[f'2020-{month:02d}-09']*3
        rows = [dict(site='example',split=period,query_utc=f'{period}-q{group}',
                     local_date=dates[group],station_id=f'd{member}')
                for group,size in enumerate(sizes) for member in range(size)]
        meta = pd.DataFrame(rows)
        for method in RAW_ARMS:
            family = method.split()[0]
            base = {'Power':3.,'Paired':2.,'Fixed':4.,'Retrained':2.5}[family]
            mapping = (int(method.split()[-1])-2028)*.4 if family in ('Fixed','Retrained') else 0.
            lead = np.linspace(.8,1.2,10)
            group_errors = np.asarray([base+mapping+index*.6+change for change in (1.,-1.,0.,1.)])[:,None]*lead
            residual = np.concatenate([np.repeat(group_errors[g:g+1]/size,size,axis=0)
                                       for g,size in enumerate(sizes)])
            yield period, method, group_residuals('ACN',meta,residual,population=meta)


def run_example():
    return analyze_groups(synthetic_groups(),source='Synthetic',site='Example',periods=PERIODS)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='New results directory; otherwise print the report')
    args = parser.parse_args(argv)
    result = run_example()
    metadata = dict(purpose='Synthetic usage demonstration, not benchmark evidence',
        groups_per_date=[1,3],members_per_group=[2,3,4,2])
    if args.output:
        print(json.dumps(write_result(args.output,result,metadata)),flush=True)
    else:
        print(render_report(result,metadata))


if __name__ == '__main__':
    main()
