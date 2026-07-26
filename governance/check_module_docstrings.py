#!/usr/bin/env python3
"""Module-level docstring gate: one line, required on every non-empty module."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

from _common import REPO_ROOT, resolve_package_dir


def first_statement_docstring(source: str) -> ast.Constant | None:
    # Returns the first-statement docstring node if one exists; else None.
    # Caller is responsible for catching SyntaxError from ast.parse.
    tree = ast.parse(source)
    if not tree.body:
        return None
    first = tree.body[0]
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
        return first.value
    return None


def check_file(path: Path) -> str | None:
    # Returns a human-readable violation message, or None on pass.
    source = path.read_text(encoding='utf-8')
    stripped = source.strip()
    if not stripped:
        return None
    try:
        docstring = first_statement_docstring(source)
    except SyntaxError as exc:
        return f'cannot parse as Python (SyntaxError: {exc.msg})'
    if docstring is None:
        return 'first stmt is not a string literal'
    value = docstring.value
    if not isinstance(value, str):
        return 'first stmt is not a string literal'
    if '\n' in value:
        line_count = value.count('\n') + 1
        return f'module docstring spans {line_count} lines'
    return None


def main() -> int:
    source_dir = resolve_package_dir('MODULE DOCSTRING GATE')
    violations: list[tuple[Path, str]] = []
    for path in sorted(source_dir.rglob('*.py')):
        msg = check_file(path)
        if msg is not None:
            violations.append((path, msg))
    if violations:
        print('MODULE DOCSTRING GATE -- FAIL', file=sys.stderr)
        print('', file=sys.stderr)
        for path, msg in violations:
            rel = path.relative_to(REPO_ROOT)
            print(f'  - {rel}: {msg}', file=sys.stderr)
        print('', file=sys.stderr)
        print(f'{len(violations)} violation(s). Merge blocked.', file=sys.stderr)
        return 1
    print('MODULE DOCSTRING GATE -- PASS')
    return 0


if __name__ == '__main__':
    sys.exit(main())
