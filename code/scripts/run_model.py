"""Run a user model script with this runtime's local API available for import."""
import argparse
from pathlib import Path
import runpy
import sys
import run_local


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('script', type=Path, help='Your Python forecasting script')
    parser.add_argument('arguments', nargs=argparse.REMAINDER,
                        help='Arguments passed unchanged to your script')
    args = parser.parse_args(argv)
    script = args.script.resolve()
    if not script.is_file():
        parser.error(f'Model script does not exist: {script}')
    old_argv, old_path = sys.argv, sys.path[:]
    try:
        sys.argv = [str(script), *args.arguments]
        sys.path.insert(0, str(script.parent))
        runpy.run_path(str(script), run_name='__main__')
    finally:
        sys.argv, sys.path[:] = old_argv, old_path


if __name__ == '__main__':
    main()
