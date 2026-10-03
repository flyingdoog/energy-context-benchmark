"""Offline public-artifact checks; prints paths/counts, never secret values."""
import argparse
import ast
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
SKIP = {'.git', '__pycache__', '.pytest_cache', '.cache', 'tmp'}
REQUIRED = ['LICENSE', 'CITATION.cff', 'README.md', 'docs/REPRODUCING.md',
            'data/README.md', 'data/CU_BEMS_ATTRIBUTION.md', 'data/epfl/packet.npz',
            'data/epfl/source_notices/LICENSE',
            'results/paper_device_cells.json', 'results/paper_declaration_cells.json']


def check(root):
    errors = [f'Missing required file: {name}' for name in REQUIRED if not (root/name).is_file()]
    counts = dict(files=0, python_files=0)
    secret = re.compile(r'(?:sk-(?:proj-)?[A-Za-z0-9_-]{24,}|gh[pousr]_[A-Za-z0-9]{25,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----)')
    private = re.compile(r'(?:[CE]:[\\/]+Users[\\/]+luods|E:[\\/]+chatgpt|connect\.nmb\d+\.seetacloud\.com)', re.I)
    for path in sorted(root.rglob('*')):
        rel = path.relative_to(root)
        if not path.is_file() or any(p in SKIP or p.startswith('.venv') for p in rel.parts):
            continue
        counts['files'] += 1
        if path.suffix.lower() in {'.pdf', '.tex', '.bib', '.bbl', '.sty', '.cls'}:
            errors.append(f'Manuscript/typesetting file excluded from this release: {rel.as_posix()}')
        if any(p.lower() in {'bpa', 'acn', 'acn_original', 'acn_declarations'} for p in rel.parts[1:]) and rel.parts[0] == 'data':
            errors.append(f'Restricted dataset present: {rel.as_posix()}')
        if path.name.startswith('.env') or path.suffix.lower() in {'.pem', '.key', '.pt', '.pth', '.ckpt'}:
            errors.append(f'Unexpected private/model file: {rel.as_posix()}')
        separate_data = rel.parts[:2] == ('data', 'cu_bems')
        if path.stat().st_size > 95*1024*1024 and not separate_data:
            errors.append(f'Too large for Git artifact: {rel.as_posix()}')
        if path.suffix.lower() in {'.py', '.md', '.json', '.csv', '.txt', '.tex', '.cff', '.toml', '.yml'}:
            content = path.read_text(encoding='utf-8-sig')
            if secret.search(content):
                errors.append(f'Credential-like value: {rel.as_posix()}')
            if private.search(content):
                errors.append(f'Private workspace path: {rel.as_posix()}')
            if path.suffix == '.py':
                try:
                    ast.parse(content)
                    counts['python_files'] += 1
                except SyntaxError:
                    errors.append(f'Invalid Python syntax: {rel.as_posix()}')
    return dict(ok=not errors, **counts, errors=errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    result = check(args.root.resolve())
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['ok'] else 1)


if __name__ == '__main__':
    main()
