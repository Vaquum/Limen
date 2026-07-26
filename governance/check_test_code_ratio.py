#!/usr/bin/env python3
"""Test/code SLOC ratio gate."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

from _common import REPO_ROOT, resolve_package_dir, significant_lines

TEST_DIR = REPO_ROOT / 'tests'

MIN_RATIO: Final[float] = 0.60
MAX_RATIO: Final[float] = 2.00
MIN_SOURCE_SLOC_FOR_GATE: Final[int] = 50


def count_py_sloc(root: Path) -> int:
    if not root.is_dir():
        return 0
    return sum(significant_lines(path) for path in sorted(root.rglob('*.py')))


def main() -> int:
    source_dir = resolve_package_dir('TEST/CODE RATIO GATE')
    source = count_py_sloc(source_dir)
    test = count_py_sloc(TEST_DIR)
    if source < MIN_SOURCE_SLOC_FOR_GATE:
        # The scan target exists (resolved and checked above); there is
        # just not yet enough source for a ratio to be meaningful.
        print(f'TEST/CODE RATIO GATE -- PASS (source {source} SLOC < {MIN_SOURCE_SLOC_FOR_GATE}, ratio not yet meaningful)')
        return 0
    ratio = test / source if source > 0 else 0.0
    if ratio < MIN_RATIO or ratio > MAX_RATIO:
        print('TEST/CODE RATIO GATE -- FAIL', file=sys.stderr)
        print('', file=sys.stderr)
        print(f'  source SLOC: {source} ({source_dir.name}/)', file=sys.stderr)
        print(f'  test   SLOC: {test} (tests/)', file=sys.stderr)
        print(
            f'  ratio:       {ratio:.2f} '
            f'(required: [{MIN_RATIO:.2f}, {MAX_RATIO:.2f}])',
            file=sys.stderr,
        )
        print('', file=sys.stderr)
        print('Merge blocked.', file=sys.stderr)
        return 1
    print(f'TEST/CODE RATIO GATE -- PASS (source={source}, test={test}, ratio={ratio:.2f})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
