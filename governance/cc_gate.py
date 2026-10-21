#!/usr/bin/env python3
"""Conventional Commits gate -- hard-fail on any deviation.

The PRD gate owns closing-reference counts and issue existence.

This gate enforces the Conventional Commits v1.0.0 specification on:

  1. The PR title.
  2. Every non-merge commit subject in ``$BASE..$HEAD``.

Linked issues carry no format requirement: PRDs and ordinary issues
alike use free-form titles.

It also rejects AI/LLM attribution in PR titles and non-merge commit
messages; commit metadata in this org never carries AI attribution.
Every commit closing reference, including merge messages, must also be
declared in the PR body so PRD admission validates the full set, and
cross-repository commit closures are forbidden.

Specification reference: https://www.conventionalcommits.org/en/v1.0.0/

Accepted format:

    <type>[optional scope][!]: <description>

Types are listed in ``CC_TYPES``. Types/scopes are lowercase; scopes allow
alphanumeric, hyphen, underscore, slash and dot. ``!`` marks breaking changes.
Require a nonblank description after the colon and exactly one space.

Exit codes:
  0 -- every rule passes.
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
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit

from _common import CC_RE, CLOSING_KEYWORD_RE, QUALIFIED_CLOSING_RE

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
# The bare tokens ``gemini``, ``cursor``, ``llm``, and unqualified
# ``generated with`` collide with legitimate domain vocabulary (the
# Gemini crypto exchange, database cursors, "generated with <tool>")
# and would hard-fail a required gate on text that names no AI
# assistant. They are narrowed to AI-qualified forms; every unambiguous
# marker stays bare. ``api`` and ``code`` are deliberately not Gemini
# qualifiers — "Gemini API client" is exchange vocabulary — and the
# co-author alternation is word-bounded so a surname like Hillman
# cannot match ``llm`` as a substring.
ATTRIBUTION_RE: Final[re.Pattern[str]] = re.compile(
    r'\bclaude\b|\bcodex\b|\bchatgpt\b|\bgpt-?\d\b|\bcopilot\b'
    r'|\bgoogle[ -]gemini\b|\bgemini[ -](?:pro|ultra|flash|cli)\b'
    r'|\bcursor[ -](?:ai|ide|agent|editor)\b'
    r'|\banthropic\b|\bopenai\b|\bai[ -]?assistant\b'
    r'|\bllm[ -](?:assist(?:ant|ed)|generated|written|authored)\b'
    r'|generated[ -]with[ -](?:an?[ -])?(?:claude|chatgpt|codex|copilot|cursor|gemini|gpt|llm|ai)\b'
    r'|co-authored-by:\s*.*\b(?:claude|openai|anthropic|chatgpt|codex|copilot|cursor|gemini|gpt|llm)\b',
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
            # Fail closed: a line this gate cannot parse means commits
            # could go unchecked, which is a setup failure, not a skip.
            print(
                f'cc_gate: unparseable git log line: {line!r}',
                file=sys.stderr,
            )
            raise SystemExit(2)
        sha, parents, subject = parts
        is_merge = len(parents.split()) > 1
        commits.append({
            'sha': sha,
            'subject': subject,
            'is_merge': is_merge,
        })
    return commits


def find_closing_references(body: str) -> list[int]:
    # Deduplicated, first-seen order: a body repeating a reference
    # resolves to one declared reference, not two.
    return list(dict.fromkeys(
        int(m.group(1)) for m in CLOSING_KEYWORD_RE.finditer(body)
    ))


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
    """Return every full message, including merges that can close issues."""
    out = run_git([
        'log', f'{base_ref}..{head_ref}', '--format=%H%x1f%B%x1e',
    ])
    messages: list[tuple[str, str]] = []
    for record in out.split('\x1e'):
        stripped = record.strip()
        if not stripped:
            continue
        sha, _, body = stripped.partition('\x1f')
        messages.append((sha.strip(), body))
    return messages


def _commit_closure_failures(message: str, declared: set[int], repo: str) -> list[str]:
    refs = set(find_closing_references(message))
    for match in QUALIFIED_CLOSING_RE.finditer(message):
        target = match.group('target')
        if target.lower().startswith(('https://', 'http://')):
            parsed = urlsplit(target)
            parts = parsed.path.strip('/').split('/')
            owner_repo = '/'.join(parts[:2])
            if parsed.netloc.lower() != os.environ.get('GH_HOST', 'github.com').lower() or len(parts) != 4:
                owner_repo = ''
            number = parts[-1]
        else:
            owner_repo, number = target.rsplit('#', 1)
        if owner_repo.lower() != repo.lower():
            return [f'cross-repository commit closure {target!r} is forbidden; use a non-closing reference.']
        refs.add(int(number))
    return [f'commit closure #{number} is absent from the validated PR body; add `Closes #{number}` to the PR body.'
            for number in sorted(refs - declared)]


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

    # Rule 2: every non-merge commit subject in range.
    commits = list_commits(base_ref, head_ref)
    for c in commits:
        if c['is_merge']:
            continue
        sha = str(c['sha'])
        subject = str(c['subject'])
        err = check_cc(subject)
        if err is not None:
            failures.append(f'commit {sha[:8]} subject {subject!r} {err}.')

    # The PR body's declared references: every commit closure below must
    # be a member of this set (rule 3). Reference-count discipline and
    # issue existence are the PRD gate's concern, not duplicated here.
    refs = find_closing_references(pr_body)

    # Rule 3: every commit closing reference is declared in the PR body;
    # rule 4: no AI/LLM attribution in the PR title or any non-merge
    # commit message. Merge messages are exempt from attribution (an
    # auto-generated merge subject may name a branch like ``copilot/...``)
    # but not from closure declaration.
    title_hit = attribution_hit(pr_title)
    if title_hit is not None:
        failures.append(
            f'PR title names an AI/LLM assistant ({title_hit!r}); strip the attribution.'
        )
    merge_shas = {str(c['sha']) for c in commits if c['is_merge']}
    for sha, message in list_commit_messages(base_ref, head_ref):
        failures.extend(_commit_closure_failures(message, set(refs), repo))
        hit = attribution_hit(message) if sha not in merge_shas else None
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
