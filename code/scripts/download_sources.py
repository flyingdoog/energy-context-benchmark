"""Plan or explicitly download original BPA/CU-BEMS files; no model fitting."""
import argparse
import json
from pathlib import Path
import shutil
import urllib.request

BPA = 'https://www.bpa.gov/-/media/Aep/energy-efficiency/emerging-technologies/'
CU_FILES = {1: (21350208, 40487855), 2: (21350226, 95782259),
            3: (21350193, 77311428), 4: (21350172, 76454851),
            5: (21350205, 75491946), 6: (21350169, 62710879),
            7: (21350217, 78867512)}


def source_plan(source, floors):
    if source == 'bpa':
        files = [dict(name=f'{s}2018.zip', url=BPA+f'{s}2018.zip')
                 for s in ['winter', 'spring', 'summer']]
        return dict(source=source, license='Provider terms; redistribution not established',
                    source_url='https://www.bpa.gov/energy-and-services/conservation/emerging-technologies/portfolio/cta-2045-water-heaters', files=files)
    if source != 'cu_bems' or not floors or any(f not in CU_FILES for f in floors):
        raise ValueError('Select bpa or cu_bems floors 1 through 7')
    return dict(source=source, license='CC-BY-4.0',
                source_url='https://doi.org/10.6084/m9.figshare.11726517.v6',
                files=[dict(name=f'2019Floor{f}.csv',
                            url=f'https://ndownloader.figshare.com/files/{CU_FILES[f][0]}',
                            size=CU_FILES[f][1]) for f in sorted(set(floors))])


def download_files(plan, output):
    output.mkdir(parents=True, exist_ok=True)
    for item in plan['files']:
        target = output/item['name']
        partial = target.with_name(target.name+'.part')
        if target.exists():
            raise FileExistsError(f'{target} already exists; choose another output or use your existing source files')
        if partial.exists():
            raise FileExistsError(f'{partial} is incomplete; inspect it or choose another output')
        request = urllib.request.Request(item['url'], headers={'User-Agent': 'context-energy-forecasting/1.0'})
        with urllib.request.urlopen(request, timeout=90) as response, partial.open('xb') as stream:
            shutil.copyfileobj(response, stream, length=1024*1024)
        if item.get('size') and partial.stat().st_size != item['size']:
            raise ValueError(f'Incomplete source file: {partial}')
        if partial.stat().st_size == 0:
            raise ValueError(f'Empty source file: {partial}')
        partial.rename(target)
        print(json.dumps(dict(downloaded=target.name, bytes=target.stat().st_size)), flush=True)
    (output/'source_receipt.json').write_text(json.dumps(plan, indent=2)+'\n', encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=['bpa', 'cu_bems'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--floors', nargs='+', type=int, choices=range(1, 8), default=[2, 3, 4, 5, 7])
    parser.add_argument('--download', action='store_true', help='Actually download; default prints the plan only')
    args = parser.parse_args(argv)
    plan = source_plan(args.source, args.floors)
    if not args.download:
        print(json.dumps(dict(preview=True, **plan), indent=2))
        return plan
    download_files(plan, args.output.resolve())
    return plan


if __name__ == '__main__':
    main()
