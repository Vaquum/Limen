#!/usr/bin/env python3
"""Shared helpers for the governance gates.

Adopted from Vaquum/new-repository-template@c4a7a05aa2c7ee487a3e6f66c20ea49a1fc5844b
and deliberately trimmed to the surface Limen's gates import: the slice
gate uses ``CLOSING_KEYWORD_RE`` and nothing else. The upstream template
ships further helpers for gates Limen never adopted; carrying them here
documented law that does not exist in this repo, so the divergence from
upstream is intentional.
"""
from __future__ import annotations

import re
from typing import Final

__all__ = ['CLOSING_KEYWORD_RE']

# Issue-closing keyword regex, shared by the slice gate (rule 1) and
# mirrored by the on-issue and sweep reruns' PR-body scans.
CLOSING_KEYWORD_RE: Final[re.Pattern[str]] = re.compile(
    r'\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s+#(\d+)\b',
    re.IGNORECASE,
)
