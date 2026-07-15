#!/usr/bin/env python3
"""Conventional Commits gate -- hard-fail on any deviation.

Adopted from Vaquum/new-repository-template@405b84d341333993fb12593f999e55576cb38d1a
(governance/cc_gate.py); adapted for Limen: the linked-issue rule checks
every closing-referenced issue that carries the ``slice`` label instead
of upstream's single-reference short-circuit, because under slice-gate
rule 9 the last slice's PR closes {slice, parent PRD} and the
short-circuit would skip slice-title validation exactly there. PRD
titles stay exempt. The divergence is filed upstream
(Vaquum/new-repository-template#61).

This gate enforces the Conventional Commits v1.0.0 specification on:

  1. The PR title.
  2. The title of every slice-labelled issue the PR closes (resolved
     by the same Closes/Fixes/Resolves #N rule the slice gate uses;
     reference-count discipline and unresolvable references are the
     slice gate's concern, not duplicated here).
  3. Every non-merge commit message in the PR's commit range
     (``$BASE..$HEAD``).

It additionally fails if the PR title or any non-merge commit message
names an AI/LLM assistant -- commit metadata in this org never carries
AI attribution.

Specification reference: https://www.conventionalcommits.org/en/v1.0.0/

Accepted format:

    <type>[optional scope]!?: <description>

where ``<type>`` is one of::

    feat | fix | docs | style | refactor | perf | test | build |
    ci | chore | revert

all lowercase. Optional ``scope`` is parenthesized alphanumeric
(hyphens, underscores, slashes, dots allowed). Optional ``!`` before
the colon marks a breaking change. ``<description>`` must be
non-empty after the ``:`` plus exactly one space.

Exit codes:
  0 -- every checked subject matches CC.
  1 -- at least one deviation.
  2 -- gate setup failure (bad args, git/gh failure, etc.).

Usage:

  python governance/cc_gate.py \\
    --pr-title "<pr title>" \\
    --pr-body-file <path> \\
    --base-ref <rev> \\
    --head-ref <rev> \\
    --repo <owner>/<name>
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Final

from _common import CC_RE, CLOSING_KEYWORD_RE

# git log --format=%H%x09%P%x09%s yields exactly three tab-separated fields.
GIT_LOG_FIELD_COUNT: Final[int] = 3

# Allowed types per the widely-followed set (Angular convention + CC v1.0.0).
CC_TYPES: Final[frozenset[str]] = frozenset({
    'feat', 'fix', 'docs', 'style', 'refactor', 'perf',
    'test', 'build', 'ci', 'chore', 'revert',
})

# AI/LLM attribution scan. Commit metadata in this org never names an
# AI/LLM assistant, so the PR title and every non-merge commit message
# must be free of these markers. Legitimate topical references are scrubbed
# first (ATTRIBUTION_EXEMPT_RE): the repo's own *.md governance files
# (CLAUDE.md, AGENTS.md, copilot-instructions.md) and the required GitHub
# Copilot review -- those are filenames and a feature, not authorship.
ATTRIBUTION_EXEMPT_RE: Final[re.Pattern[str]] = re.compile(
    r'\b[\w-]+\.md\b|\bcopilot(?:[ -]code)?[ -]review\b',
    re.IGNORECASE,
)
ATTRIBUTION_RE: Final[re.Pattern[str]] = re.compile(
    r'\bclaude\b|\bcodex\b|\bchatgpt\b|\bgpt-?\d\b|\bcopilot\b|\bcursor\b'
    r'|\bgemini\b|\banthropic\b|\bopenai\b|\bllm\b|\bai[ -]?assistant\b'
    r'|generated[ -]with'
    r'|co-authored-by:\s*.*(?:claude|openai|anthropic|chatgpt|codex|copilot|cursor|gemini)',
    re.IGNORECASE,
)


def check_cc(subject: str) -> str | None:
    """Return ``None`` if ``subject`` matches CC; otherwise return a
    human-readable reason for the failure."""
    first_line = subject.split('\n', 1)[0]
    if not first_line:
        return 'empty subject'
    match = CC_RE.match(first_line)
    if match is None:
        return (
            'does not match Conventional Commits v1.0.0 format '
            "'<type>[(scope)][!]: <description>'"
        )
    cc_type = match.group('type')
    if cc_type not in CC_TYPES:
        return (
            f"type {cc_type!r} is not one of the allowed CC types "
            f"(allowed: {sorted(CC_TYPES)!r})"
        )
    description = match.group('description')
    if not description or not description.strip():
        return 'description after the colon is empty or whitespace-only'
    return None


def run_git(args: list[str]) -> str:
    try:
        result = subprocess.run(
            ['git', *args],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        print(f'cc_gate: git not found: {exc}', file=sys.stderr)
        raise SystemExit(2) from exc
    if result.returncode != 0:
        print(
            f'cc_gate: git {" ".join(args)} failed (exit {result.returncode}): '
            f'{result.stderr.strip()}',
            file=sys.stderr,
        )
        raise SystemExit(2)
    return result.stdout


def list_commits(base_ref: str, head_ref: str) -> list[dict[str, object]]:
    """List every commit in ``base_ref..head_ref`` with metadata for
    deciding whether to check it. ``is_merge`` is derived from the
    parent count (more than one = merge commit)."""
    out = run_git([
        'log', f'{base_ref}..{head_ref}',
        '--format=%H%x09%P%x09%s',
    ])
    commits: list[dict[str, object]] = []
    for line in out.splitlines():
        if not line.strip():
            continue
        parts = line.split('\t', 2)
        if len(parts) != GIT_LOG_FIELD_COUNT:
            continue
        sha, parents, subject = parts
        is_merge = len(parents.split()) > 1
        commits.append({
            'sha': sha,
            'subject': subject,
            'is_merge': is_merge,
        })
    return commits


def fetch_issue(repo: str, number: int) -> dict[str, object]:
    """Return ``{'title': ..., 'labels': [...]}`` for the issue.

    Any gh failure here is a gate setup failure because linked-issue
    title validation is part of cc_gate's own contract.
    """
    try:
        result = subprocess.run(
            [
                'gh', 'api', f'repos/{repo}/issues/{number}',
                '--jq', '{title: .title, labels: [.labels[].name]}',
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        print(f'cc_gate: gh not found: {exc}', file=sys.stderr)
        raise SystemExit(2) from exc
    if result.returncode != 0:
        error_output = (
            result.stderr.strip()
            or result.stdout.strip()
            or 'no error output'
        )
        print(
            f'cc_gate: gh api repos/{repo}/issues/{number} failed '
            f'(exit {result.returncode}): {error_output}',
            file=sys.stderr,
        )
        raise SystemExit(2)
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        print(
            f'cc_gate: linked issue #{number} payload is not JSON: {exc}',
            file=sys.stderr,
        )
        raise SystemExit(2) from exc
    if not isinstance(payload, dict) or not str(payload.get('title', '')).strip():
        print(
            f'cc_gate: linked issue #{number} has an empty title payload',
            file=sys.stderr,
        )
        raise SystemExit(2)
    return payload


def find_closing_references(body: str) -> list[int]:
    return [int(m.group(1)) for m in CLOSING_KEYWORD_RE.finditer(body)]


def attribution_hit(text: str) -> str | None:
    """Return the first AI/LLM-attribution substring in ``text``, or None.

    Legitimate topical references (``*.md`` filenames, the Copilot review
    feature) are scrubbed before the scan so they cannot register.
    """
    scrubbed = ATTRIBUTION_EXEMPT_RE.sub(' ', text)
    match = ATTRIBUTION_RE.search(scrubbed)
    if match is None:
        return None
    return match.group(0)


def list_commit_messages(base_ref: str, head_ref: str) -> list[tuple[str, str]]:
    """Return ``(sha, full message)`` for every non-merge commit in range.

    Merge commits are excluded (``--no-merges``) so an auto-generated merge
    subject that happens to name a branch like ``copilot/...`` does not
    register as authored attribution.
    """
    out = run_git([
        'log', f'{base_ref}..{head_ref}', '--no-merges', '--format=%H%x1f%B%x1e',
    ])
    messages: list[tuple[str, str]] = []
    for record in out.split('\x1e'):
        stripped = record.strip()
        if not stripped:
            continue
        sha, _, body = stripped.partition('\x1f')
        messages.append((sha.strip(), body))
    return messages


def gate(
    pr_title: str,
    pr_body: str,
    base_ref: str,
    head_ref: str,
    repo: str,
) -> list[str]:
    failures: list[str] = []

    # Rule 1: PR title.
    err = check_cc(pr_title)
    if err is not None:
        failures.append(f'PR title {pr_title!r} {err}.')

    # Rule 2: every non-merge commit in range.
    commits = list_commits(base_ref, head_ref)
    for c in commits:
        if c['is_merge']:
            continue
        sha = str(c['sha'])
        subject = str(c['subject'])
        err = check_cc(subject)
        if err is not None:
            failures.append(f'commit {sha[:8]} subject {subject!r} {err}.')

    # Rule 3: linked slice-issue titles. Reference-count discipline is
    # a slice_gate concern; cc_gate checks CC on every closing-referenced
    # issue that carries the ``slice`` label and exempts the rest (the
    # parent PRD on rule-9 last-slice PRs, most notably).
    refs = find_closing_references(pr_body)
    for number in refs:
        issue = fetch_issue(repo, number)
        labels = issue.get('labels')
        if not isinstance(labels, list) or 'slice' not in labels:
            continue
        issue_title = str(issue.get('title', ''))
        err = check_cc(issue_title)
        if err is not None:
            failures.append(
                f'linked slice issue #{number} title {issue_title!r} {err}.'
            )

    # Rule 4: no AI/LLM attribution in the PR title or any commit message.
    title_hit = attribution_hit(pr_title)
    if title_hit is not None:
        failures.append(
            f'PR title names an AI/LLM assistant ({title_hit!r}); strip the attribution.'
        )
    for sha, message in list_commit_messages(base_ref, head_ref):
        hit = attribution_hit(message)
        if hit is not None:
            failures.append(
                f'commit {sha[:8]} message names an AI/LLM assistant '
                f'({hit!r}); strip the attribution.'
            )

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description='Conventional Commits gate')
    parser.add_argument('--pr-title', required=True)
    parser.add_argument('--pr-body-file', required=True)
    parser.add_argument('--base-ref', required=True)
    parser.add_argument('--head-ref', required=True)
    parser.add_argument('--repo', required=True)
    args = parser.parse_args()

    try:
        pr_body = Path(args.pr_body_file).read_text(encoding='utf-8')
    except OSError as exc:
        print(
            f'cc_gate: cannot read --pr-body-file {args.pr_body_file}: {exc}',
            file=sys.stderr,
        )
        return 2

    failures = gate(
        args.pr_title,
        pr_body,
        args.base_ref,
        args.head_ref,
        args.repo,
    )

    if failures:
        print('CC GATE -- FAIL')
        print()
        for msg in failures:
            print(f'  - {msg}')
        print()
        print(f'{len(failures)} failure(s). Merge blocked.')
        return 1

    print('CC GATE -- PASS')
    return 0


if __name__ == '__main__':
    sys.exit(main())
