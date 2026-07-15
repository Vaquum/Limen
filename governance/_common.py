#!/usr/bin/env python3
"""Shared helpers for the governance gates.

Adopted from Vaquum/new-repository-template@c4a7a05aa2c7ee487a3e6f66c20ea49a1fc5844b
and deliberately trimmed to the surface Limen's gates import: the slice
gate uses ``CLOSING_KEYWORD_RE``; the Conventional Commits gate uses
``CC_RE`` and ``CLOSING_KEYWORD_RE``; the dependency vulnerability gate
uses ``REPO_ROOT`` and ``fail_setup``. The upstream template ships
further helpers for gates Limen never adopted; carrying them here meant
documenting law that does not exist in this repo, so the divergence
from upstream is intentional.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Final, NoReturn

__all__ = ['CC_RE', 'CLOSING_KEYWORD_RE', 'REPO_ROOT', 'fail_setup']

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]

# The Conventional Commits subject regex: type, optional scope (without
# parens), optional breaking marker, description. Consumed by cc_gate.
CC_RE: Final[re.Pattern[str]] = re.compile(
    r'^(?P<type>[a-z]+)'
    r'(?:\((?P<scope>[a-z0-9._/\-]+)\))?'
    r'(?P<breaking>!)?'
    r': (?P<description>.+)$'
)

# Issue-closing keyword regex, shared by the slice gate (rule 1) and
# the CC gate's linked-issue rule, and mirrored by the on-issue and
# sweep reruns' PR-body scans.
CLOSING_KEYWORD_RE: Final[re.Pattern[str]] = re.compile(
    r'\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s+#(\d+)\b',
    re.IGNORECASE,
)


def fail_setup(banner: str, message: str) -> NoReturn:
    """Report a gate setup failure under the gate's banner and exit 2.

    Setup failures (a missing config, an unreadable pyproject) are
    distinct from gate violations: they mean the gate could not run, so
    it fails closed rather than passing over an empty tree.
    """
    print(f'{banner} -- FAIL', file=sys.stderr)
    print(f'  {message}', file=sys.stderr)
    sys.exit(2)
