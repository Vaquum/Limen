#!/usr/bin/env python3
"""Block publishing until a commit's validation check-runs are green.

The publish workflow resolves a release tag, then runs this script against
the tagged commit: it polls the commit's check-runs until every
non-excluded run completes, and fails loudly unless every one of them
succeeded (or was skipped). Excluded runs are the release machinery
itself, matched by case-insensitive name substring, which can neither
validate the commit nor be waited on without self-deadlock.

A green snapshot passes only after it stays stable: the observed run set
must be nonempty, fully complete, and unchanged across ``settle_polls``
consecutive polls. A single fast check going green while the rest of the
suite is still registering must not unblock publication.

Zero observed validation runs is a failure, never a pass: a commit with
nothing watching it must not ship.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass


PASS_CONCLUSIONS = frozenset({'success', 'skipped', 'neutral'})


@dataclass(frozen=True)
class Verdict:
    pending: tuple[str, ...]
    failed: tuple[str, ...]
    passed: tuple[str, ...]
    excluded: tuple[str, ...]


def partition_runs(
    runs: list[dict[str, object]], excluded_substrings: list[str]
) -> Verdict:
    """Split raw check-run payloads into pending/failed/passed/excluded."""
    lowered = [s.lower() for s in excluded_substrings]
    pending: list[str] = []
    failed: list[str] = []
    passed: list[str] = []
    excluded: list[str] = []
    for run in runs:
        name = str(run.get('name', ''))
        if any(sub in name.lower() for sub in lowered):
            excluded.append(name)
            continue
        if run.get('status') != 'completed':
            pending.append(name)
            continue
        if str(run.get('conclusion', '')) in PASS_CONCLUSIONS:
            passed.append(name)
        else:
            failed.append(f'{name} ({run.get("conclusion")})')
    return Verdict(tuple(pending), tuple(failed), tuple(passed), tuple(excluded))


def fetch_check_runs(repo: str, sha: str) -> list[dict[str, object]]:
    """Return every check-run on a commit, failing loudly on API trouble.

    ``gh api --paginate`` emits each page separately, so the jq filter
    yields one run object per line and every line parses independently:
    single-page and multi-page output share the same shape.
    """
    try:
        result = subprocess.run(
            [
                'gh', 'api', f'repos/{repo}/commits/{sha}/check-runs',
                '--paginate', '--jq', '.check_runs[]',
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise SystemExit(f'wait_for_commit_checks: gh not found: {exc}') from exc
    if result.returncode != 0:
        raise SystemExit(
            f'wait_for_commit_checks: gh api failed: {(result.stderr or result.stdout).strip()}'
        )
    runs: list[dict[str, object]] = []
    for lineno, line in enumerate((result.stdout or '').splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(
                f'wait_for_commit_checks: bad gh output on line {lineno}: {exc}'
            ) from exc
        if not isinstance(obj, dict):
            raise SystemExit(
                f'wait_for_commit_checks: unexpected gh output shape on line {lineno}'
            )
        runs.append(obj)
    return runs


def wait_for_green(
    repo: str,
    sha: str,
    excluded_substrings: list[str],
    timeout_seconds: float,
    poll_seconds: float,
    settle_polls: int = 3,
) -> Verdict:
    """Poll until the commit's validation settles or the timeout expires.

    A fully green snapshot passes only once the observed run set has
    stayed identical for ``settle_polls`` consecutive polls, so checks
    that register late cannot miss the gate.
    """
    if settle_polls < 1:
        raise SystemExit('wait_for_commit_checks: --settle-polls must be >= 1')
    deadline = time.monotonic() + timeout_seconds
    last: Verdict = Verdict(pending=(), failed=(), passed=(), excluded=())
    stable_rounds = 0
    last_names: frozenset[str] | None = None
    while True:
        last = partition_runs(fetch_check_runs(repo, sha), excluded_substrings)
        if last.failed:
            return last
        watched = last.pending + last.passed
        if watched and not last.pending:
            names = frozenset(watched + last.excluded)
            if names == last_names:
                stable_rounds += 1
            else:
                stable_rounds = 1
                last_names = names
            if stable_rounds >= settle_polls:
                return last
        else:
            stable_rounds = 0
            last_names = None
        if time.monotonic() >= deadline:
            return last
        time.sleep(poll_seconds)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True)
    parser.add_argument('--sha', required=True)
    parser.add_argument('--timeout-seconds', type=float, default=1200.0)
    parser.add_argument('--poll-seconds', type=float, default=20.0)
    parser.add_argument('--settle-polls', type=int, default=3)
    parser.add_argument('--exclude-substring', action='append', default=[])
    args = parser.parse_args(argv)
    verdict = wait_for_green(
        args.repo, args.sha, args.exclude_substring,
        args.timeout_seconds, args.poll_seconds, args.settle_polls,
    )
    if verdict.failed:
        print('wait_for_commit_checks: FAILING CHECKS:', file=sys.stderr)
        for name in verdict.failed:
            print(f'  - {name}', file=sys.stderr)
        return 1
    if not verdict.pending and not verdict.passed:
        print(
            'wait_for_commit_checks: no validation check-runs observed; refusing to ship',
            file=sys.stderr,
        )
        return 1
    if verdict.pending:
        print('wait_for_commit_checks: timed out waiting for:', file=sys.stderr)
        for name in verdict.pending:
            print(f'  - {name}', file=sys.stderr)
        return 1
    print(f'wait_for_commit_checks: PASS ({len(verdict.passed)} green, '
          f'{len(verdict.excluded)} excluded)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
