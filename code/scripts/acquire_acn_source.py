"""Plan fixed-revision ACN source acquisition; download only when requested."""
import argparse
import json
from pathlib import Path
import run_local


def build_plan(output, index_source=None):
    from field_control.acn_source import build_plan as source_plan
    options = {'index_source': index_source} if index_source is not None else {}
    return source_plan(output, **options)


def acquire(output, sample=False, workers=4, index_source=None):
    from field_control.acn_source import acquire as source_acquire
    options = {'index_source': index_source} if index_source is not None else {}
    return source_acquire(output, sample=sample, workers=workers, **options)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True,
                        help='Source plan, manifest and downloaded-session directory')
    parser.add_argument('--index-source', type=Path,
                        help='Optional directory of existing provider GitHub tree JSON responses')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--sample', action='store_true',
                      help='Fetch only each site/split first and last file; not an evaluation dataset')
    mode.add_argument('--download', action='store_true',
                      help='Explicitly fetch the full fixed-source collection; resume existing files')
    parser.add_argument('--workers', type=int, choices=range(1, 5), default=4,
                        help='Download workers, one to four (default: four)')
    args = parser.parse_args(argv)
    output = args.output.resolve()
    index_options = {'index_source': args.index_source.resolve()} if args.index_source is not None else {}
    if args.sample or args.download:
        receipt = acquire(output, sample=args.sample, workers=args.workers, **index_options)
        selected_mode = 'sample' if args.sample else 'download'
    else:
        receipt = build_plan(output, **index_options)
        selected_mode = 'plan'
    summary = {'mode': selected_mode, 'output': str(output)}
    for key in ('plan_completed', 'completed', 'selected_completed', 'revision', 'phase',
                'manifest_contexts', 'planned_files', 'listed_bytes', 'session_downloads',
                'selected', 'planned', 'available_files', 'downloaded', 'errors', 'stopped',
                'transferred_bytes', 'workers', 'seconds'):
        value = receipt.get(key)
        if isinstance(value, (str, int, float, bool)):
            summary[key] = value
    print(json.dumps(summary), flush=True)
    if selected_mode != 'plan' and not receipt.get('selected_completed'):
        raise SystemExit(1)
    return receipt


if __name__ == '__main__':
    main()
