#!/usr/bin/env python3
"""Shared helpers for the governance gates.

One place for the things several gates would otherwise each re-implement:
the repo root, setup-failure reporting, package-root resolution, the
significant-line counter, and the Conventional-Commits patterns. This is
infrastructure shared by the gates, not a gate itself — it has no banner and
is never a required check. Sharing a constant here (e.g. `CC_RE`) is not the
same as coupling two gates' logic; the gates remain independently invocable.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Final, NoReturn

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
TYPING_BUDGET: Final[Path] = REPO_ROOT / '.github' / 'typing_budget.json'

# The Conventional Commits subject regex: type, optional scope (without
# parens), optional breaking marker, description. Shared by cc_gate and
# version_gate so the two cannot silently drift apart.
CC_RE: Final[re.Pattern[str]] = re.compile(
    r'^(?P<type>[a-z]+)'
    r'(?:\((?P<scope>[a-z0-9._/\-]+)\))?'
    r'(?P<breaking>!)?'
    r': (?P<description>.+)$'
)

# Issue-closing keyword regex, shared by cc_gate and slice_gate.
CLOSING_KEYWORD_RE: Final[re.Pattern[str]] = re.compile(
    r'\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s+#(\d+)\b',
    re.IGNORECASE,
)


def fail_setup(banner: str, message: str) -> NoReturn:
    """Report a gate setup failure under the gate's banner and exit 2.

    Setup failures (a missing config, an unresolvable package root) are
    distinct from gate violations: they mean the gate could not run, so it
    fails closed rather than passing over an empty tree.
    """
    print(f'{banner} -- FAIL', file=sys.stderr)
    print(f'  {message}', file=sys.stderr)
    sys.exit(2)


def resolve_package_dir(banner: str) -> Path:
    """Resolve the package directory from `typing_budget.json`'s single
    `package_root`, failing closed under `banner` if it cannot be resolved.

    A gate that cannot find its scan target must block the merge instead of
    passing over an empty tree, so a half-finished package rename cannot
    silently disable it.
    """
    if not TYPING_BUDGET.is_file():
        fail_setup(banner, f'missing {TYPING_BUDGET.relative_to(REPO_ROOT)} (cannot resolve package_root)')
    data = json.loads(TYPING_BUDGET.read_text(encoding='utf-8'))
    root = data.get('package_root') if isinstance(data, dict) else None
    path = REPO_ROOT / root if isinstance(root, str) and root else None
    if path is None or not path.is_dir():
        fail_setup(banner, f'package_root {root!r} is not a directory under the repo root')
    return path


def significant_lines(path: Path) -> int:
    """Count non-blank, non-comment-only lines — the unit budgets and the
    test/code ratio are measured in."""
    count = 0
    for line in path.read_text(encoding='utf-8').splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith('#'):
            count += 1
    return count


def _is_path_excluded(rel: Path, excludes: list[str]) -> bool:
    # Path-part match, not substring: an exclude entry matches only if its
    # parts appear as a contiguous slice of rel's parts, so 'dist' does not
    # spuriously match 'new_repository_template/distance.py'.
    parts = rel.parts
    for ex in excludes:
        ex_parts = Path(ex).parts
        if not ex_parts:
            continue
        width = len(ex_parts)
        for i in range(max(0, len(parts) - width + 1)):
            if parts[i:i + width] == ex_parts:
                return True
    return False


def find_python_files(root: Path, excludes: list[str]) -> list[Path]:
    """Every `*.py` under root whose path is not excluded, sorted."""
    return [
        path for path in sorted(root.rglob('*.py'))
        if not _is_path_excluded(path.relative_to(REPO_ROOT), excludes)
    ]
