#!/usr/bin/env python3
"""Shared helpers for Limen's governance gate.

Adopted from Vaquum/new-repository-template@c4a7a05aa2c7ee487a3e6f66c20ea49a1fc5844b,
then trimmed to the surface Limen's only gate imports.

Limen runs exactly one governance gate: `slice_gate.py`. The single thing it
shares from here is `CLOSING_KEYWORD_RE` (the issue-closing keyword pattern,
rule 1). The upstream template also ships a Conventional-Commits gate and a
typing-budget gate, whose helpers (`CC_RE`, `fail_setup`, `resolve_package_dir`,
`significant_lines`, `find_python_files`, `REPO_ROOT`, `TYPING_BUDGET`) Limen
never adopted; carrying them here would be dead surface that a reader would
mistake for live infrastructure. Divergence from the upstream template is
therefore intentional: this module contains only what the slice gate imports.
"""
from __future__ import annotations

import re
from typing import Final

__all__ = ['CLOSING_KEYWORD_RE']

# Issue-closing keyword regex, used by slice_gate (rule 1).
CLOSING_KEYWORD_RE: Final[re.Pattern[str]] = re.compile(
    r'\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s+#(\d+)\b',
    re.IGNORECASE,
)
