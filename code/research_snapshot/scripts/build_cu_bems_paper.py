"""Build CU-BEMS manuscript tables from completed JSON results; no fits."""
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'paper/generated'
BASE = ROOT/'results/cu_bems_neural_reference_v1'
METHODS = ('Persistence', 'Ridge: power', 'Ridge: thermal', 'Tree: power',
           'Tree: thermal', 'N-HiTS: power', 'N-HiTS: thermal')
PERIODS = {'later_1': 'Sep-Oct', 'later_2': 'Nov-Dec'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(name, title, caption, lines):
    (OUT/name).write_text(f'# {title}\n\n{caption}\n\n'+'\n'.join(lines)+'\n', encoding='utf-8')


def main():
    values = {(r['split'], r['method'], r['metric']): r['value']
        for r in read(BASE/'analysis/common_cohort_scores.json')
        if r['cohort']=='thermal_five_floors' and r['scale']=='device'
        and r['span_minutes']==0 and r['metric'] in ('mse','mae')}
    original = (OUT/'common_device_results.md').read_text(encoding='utf-8')
    table = re.findall(r'(?m)^\|[^\n]*\n(?:\|[^\n]*(?:\n|$))+', original)[0]
    lines = []
    for index, line in enumerate(table.splitlines()):
        fields = [s.strip() for s in line.strip('|').split('|')]
        if index==0:
            extra = ['CU-BEMS Sep-Oct', 'CU-BEMS Nov-Dec']
        elif index==1:
            extra = ['---:', '---:']
        else:
            model = {'LightGBM':'Tree'}.get(fields[0], fields[0])
            condition = 'power' if fields[1]=='History' else 'thermal'
            method = f'{model}: {condition}'
            extra = [f"{values[(period,method,'mse')]:.3f} / {values[(period,method,'mae')]:.3f}"
                     for period in PERIODS]
        lines.append('| '+' | '.join(fields+extra)+' |')
    write('common_device_results_with_cu_bems.md', 'Four-source device forecasting',
        'MSE (kW squared) / MAE (kW). CU-BEMS uses Floors 2-5 and 7 on common dates; '
        'equal floor/date/span weights. Its state and all-input conditions are the same thermal arm, '
        'not separate fits. Other source results are reused unchanged.', lines)

    native = read(ROOT/'results/cu_bems_full_reference_v1/summary.json')+read(BASE/'scores/summary.json')
    keys = ('floor','split','scale','span_minutes','method')
    lookup = {tuple(r[k] for k in keys):r for r in native}
    if len(lookup)!=len(native): raise ValueError('Duplicate native score cells')
    for scale, name in [('device','cu_bems_device_full.md'), ('floor_total','cu_bems_floor_full.md')]:
        lines = ['| Floor | Period | Span (min) | Persistence | Ridge history | Ridge + thermal | Tree history | Tree + thermal | N-HiTS history | N-HiTS + thermal |',
                 '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        count = 0
        for floor in range(1,8):
            for period, label in PERIODS.items():
                if floor==6 and period=='later_2': continue
                for span in (30,90,150):
                    cells = [str(floor), label, str(span)]
                    for method in METHODS:
                        key = (floor,period,scale,span,method)
                        if floor==1 and method.endswith('thermal'):
                            if key in lookup: raise ValueError('Unexpected sensor-absent result')
                            cells.append('n.a.')
                        else:
                            row = lookup[key]
                            cells.append(f"{row['mse']:.3f} / {row['mae']:.3f}")
                    lines.append('| '+' | '.join(cells)+' |'); count += 1
        if count!=39: raise ValueError('All 13 available floor-periods and three spans required')
        write(name, f'CU-BEMS {scale} results', 'MSE (kW squared) / MAE (kW); date-balanced within each floor. '
            'N-HiTS averages three seed errors. Floor 6 has no Nov-Dec targets; Floor 1 has no thermal arm.', lines)

    support = read(ROOT/'results/benchmark_source_feasibility_v1/cu_bems/full_2019/support.json')
    lookup = {(r['floor'],r['split']):r for r in support}
    lines = ['| Floor | ACs (thermal) | Training cases | Validation cases | Sep-Oct cases | Nov-Dec cases | Sep-Oct floor groups / dates | Nov-Dec floor groups / dates |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for floor in range(1,8):
        row = lookup[(floor,'train')]
        cells = [str(floor), f"{row['devices']} ({row['thermal_devices']})"]
        cells += [f"{lookup[(floor,s)]['device_cases']:,}" for s in ('train','validation','later_1','later_2')]
        cells += [f"{lookup[(floor,s)]['floor_windows']:,} / {lookup[(floor,s)]['floor_dates']}" for s in PERIODS]
        lines.append('| '+' | '.join(cells)+' |')
    write('cu_bems_support.md','CU-BEMS floor support',
        'Complete power history and future define device cases. Thermal missingness does not remove cases. '
        'Floor groups require every original member.',lines)
    receipt = dict(completed=True, new_fits=0, tables=4, detailed_rows_per_scale=39,
        main_common_floors=[2,3,4,5,7], main_thermal_devices=44,
        main_note='State and full reuse the same CU-BEMS arm; original source results unchanged')
    (BASE/'paper_tables.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__': main()
