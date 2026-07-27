"""Honesty gate: the written laws and the enforced ruleset agree exactly.

The constitution (CLAUDE.md) lists the laws; each workflow-gate law carries
its required-status-check context as a trailing `*(context)*` annotation, and
exactly one law is the server-side branch-protection law. This gate asserts a
bijection: the contexts named in the laws equal the contexts the ruleset
actually requires. A gate added to the ruleset without a law, or a law whose
gate was dropped from the ruleset, fails here -- the written laws cannot drift
from what is enforced.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LAWS_DOC = REPO_ROOT / 'CLAUDE.md'
RULESET = REPO_ROOT / '.github/rulesets/main.json'

# The one law that maps to server-side branch protection, not a status check.
BRANCH_PROTECTION_ANNOTATION = 'branch protection, server-side'

_LAW_LINE = re.compile(r'^\d+\.\s')
_ANNOTATION = re.compile(r'\*\((?P<annotation>.+)\)\*\s*$')


def _laws_section() -> list[str]:
    # Only the "## The laws" section is parsed, so a numbered list elsewhere
    # in the document cannot be mistaken for a law.
    out: list[str] = []
    in_section = False
    for line in LAWS_DOC.read_text(encoding='utf-8').splitlines():
        if line.startswith('## '):
            in_section = line.strip() == '## The laws'
            continue
        if in_section:
            out.append(line)
    return out


def _law_annotations() -> list[str]:
    annotations: list[str] = []
    for line in _laws_section():
        if not _LAW_LINE.match(line):
            continue
        m = _ANNOTATION.search(line)
        assert m is not None, f'law without a trailing *(annotation)*: {line!r}'
        annotations.append(m.group('annotation').strip())
    return annotations


def _ruleset_contexts() -> set[str]:
    payload = json.loads(RULESET.read_text(encoding='utf-8'))
    checks = next(r for r in payload['rules'] if r['type'] == 'required_status_checks')
    return {entry['context'] for entry in checks['parameters']['required_status_checks']}


def test_exactly_one_branch_protection_law() -> None:
    branch = [a for a in _law_annotations() if a == BRANCH_PROTECTION_ANNOTATION]
    assert len(branch) == 1, f'expected exactly one branch-protection law, got {len(branch)}'


def test_laws_and_ruleset_are_in_bijection() -> None:
    law_contexts = {a for a in _law_annotations() if a != BRANCH_PROTECTION_ANNOTATION}
    ruleset_contexts = _ruleset_contexts()
    assert law_contexts == ruleset_contexts, (
        'laws <-> ruleset drift -- '
        f'in laws only: {sorted(law_contexts - ruleset_contexts)}; '
        f'in ruleset only: {sorted(ruleset_contexts - law_contexts)}'
    )
