#!/usr/bin/env python3
"""Test fallback gate: tests must not use try/except (use pytest.raises)."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

from _common import REPO_ROOT

TEST_DIRS = (REPO_ROOT / 'tests', REPO_ROOT / 'governance' / 'tests')


def find_try_statements(source: str) -> list[int]:
    # A test proves exception behavior with `pytest.raises`, never with a
    # try/except that can swallow a failing assertion. Flag every try (and
    # the 3.11+ try/except* form) by line number.
    tree = ast.parse(source)
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, (ast.Try, ast.TryStar))
    ]


def main() -> int:
    violations: list[tuple[Path, int]] = []
    for test_dir in TEST_DIRS:
        if not test_dir.is_dir():
            continue
        for path in sorted(test_dir.rglob('*.py')):
            if '__pycache__' in path.parts:
                continue
            for lineno in find_try_statements(path.read_text(encoding='utf-8')):
                violations.append((path.relative_to(REPO_ROOT), lineno))
    if violations:
        print('TEST FALLBACK GATE -- FAIL', file=sys.stderr)
        print('', file=sys.stderr)
        for rel, lineno in violations:
            print(
                f'  - {rel}:{lineno}: try/except in a test; use pytest.raises instead',
                file=sys.stderr,
            )
        print('', file=sys.stderr)
        print(f'{len(violations)} violation(s). Merge blocked.', file=sys.stderr)
        return 1
    print('TEST FALLBACK GATE -- PASS')
    return 0


if __name__ == '__main__':
    sys.exit(main())
