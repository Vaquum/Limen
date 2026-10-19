import ast
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / 'docs/examples'


def main():
    subprocess.run([sys.executable, '-m', 'ruff', 'check', '--select', 'F,E9', str(EXAMPLES)], check=True, timeout=30)
    subprocess.run([sys.executable, '-m', 'pyright', '--pythonpath', sys.executable, '--project', str(EXAMPLES / 'pyrightconfig.json')], check=True, timeout=60)
    sys.path.insert(0, str(ROOT / 'governance'))
    from fail_loud_gate import CATEGORIES, count_violations
    files = list(EXAMPLES.rglob('*.py'))
    counts = count_violations(files)
    if any(counts[category] for category in CATEGORIES):
        raise ValueError('Documentation source contains a silent fallback: ' + str(counts))
    for path in files:
        tree = ast.parse(path.read_text(), filename=str(path))
        if any(isinstance(node, ast.Name) and node.id == 'Any' or isinstance(node, ast.Attribute) and node.attr == 'Any' for node in ast.walk(tree)):
            raise ValueError('Documentation source introduces an explicit untyped escape: ' + str(path))
    print('Documentation Python passes lint, type checking and the zero-fallback scan.')


if __name__ == '__main__':
    main()
