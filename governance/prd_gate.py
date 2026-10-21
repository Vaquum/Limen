#!/usr/bin/env python3
"""PRD gate -- mechanical enforcement of the PR <-> issue contract.

Every PR closes exactly one OPEN issue. A ``feat`` PR's issue must be a
PRD (an issue carrying the ``prd`` label, filed via
``.github/ISSUE_TEMPLATE/prd.yml``); any other Conventional Commits type
may close any OPEN issue -- or a PRD when the author wants more eyes on it.

This gate blocks a PR that:

  1.  Does not cite exactly one issue via Closes/Fixes/Resolves #N. Zero
      references, or more than one, fail: one PR delivers one issue, and
      GitHub honors every closing keyword on merge, so an extra reference
      would close an issue this PR did not deliver.
  2.  Cites a number that does not resolve in the repo.
  2a. Cites a number that resolves to a pull request rather than an issue
      (GitHub's issues endpoint also returns PRs).
  3.  Cites an issue that is not in the OPEN state.
  4.  Has a ``feat``-typed title but cites an issue without the ``prd``
      label. The type comes from the PR title's Conventional Commits
      subject (``feat``, ``feat(scope)``, ``feat!``). Title format itself
      is cc_gate's concern: a title that does not parse as Conventional
      Commits skips rule 4 here and fails there instead, so the merge
      stays blocked either way.

GitHub also honors closing keywords followed by a qualified
``owner/repo#N`` reference or an issue URL. The gate cannot fold those
into the closing set it validates, so their presence hard-fails instead
of silently widening what a merge will close.

Usage:

  python governance/prd_gate.py \\
    --pr-title "<pr title>" \\
    --pr-body-file <path> \\
    --repo <owner>/<name>

Exit codes:

  0 -- all gates pass
  1 -- one or more gates failed
  2 -- gate itself could not run (bad args, gh failure, etc.)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Final

from _common import CC_RE, CLOSING_KEYWORD_RE, QUALIFIED_CLOSING_RE

# The label filed by .github/ISSUE_TEMPLATE/prd.yml. governance.yml mirrors
# it as a contract anchor; the contract test pins the two together.
PRD_LABEL: Final[str] = 'prd'


def find_closing_references(body: str) -> list[int]:
    """Return the deduplicated list of issue numbers referenced by
    closing keywords in the PR body, in order of first appearance.
    Deduplication matches GitHub's own closing behavior: repeating
    ``Closes #N`` resolves to one closed issue, not two."""
    return list(dict.fromkeys(
        int(m.group(1)) for m in CLOSING_KEYWORD_RE.finditer(body)
    ))


def fetch_issue(repo: str, number: int) -> dict[str, object] | None:
    """Fetch an issue via gh. Returns None if the issue does not exist.
    Any other gh failure (auth, network, permissions) raises SystemExit(2).

    Implemented with a single subprocess call so a 404 does not emit
    misleading stderr about an unrelated failure path.
    """
    # NOTE: /repos/{repo}/issues/{number} returns pull requests too.
    # Include ``is_pull_request`` in the projection so rule 2a can
    # reject a PR being used as a stand-in for an issue.
    try:
        result = subprocess.run(
            [
                'gh', 'api', f'repos/{repo}/issues/{number}',
                '--jq', '{state: .state, labels: [.labels[].name], '
                        'is_pull_request: (.pull_request != null)}',
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        print(f'prd_gate: gh not found: {exc}', file=sys.stderr)
        raise SystemExit(2) from exc

    if result.returncode == 0:
        return json.loads(result.stdout) if result.stdout.strip() else None

    # A 404 is the legitimate "issue does not exist" case; every other
    # non-zero exit is a setup failure.
    if 'Not Found' in result.stderr or '404' in result.stderr:
        return None

    print(
        f'prd_gate: gh api repos/{repo}/issues/{number} failed '
        f'(exit {result.returncode}): {result.stderr.strip()}',
        file=sys.stderr,
    )
    raise SystemExit(2)


def pr_type_is_feat(pr_title: str) -> bool:
    """True when the PR title's Conventional Commits type is ``feat``."""
    match = CC_RE.match(pr_title.split('\n', 1)[0])
    return match is not None and match.group('type') == 'feat'


def gate(
    pr_title: str,
    pr_body: str,
    repo: str,
) -> list[str]:
    """Run all PR <-> issue checks. Return a list of failure messages
    (empty list means PASS)."""
    hits = [m.group(0) for m in QUALIFIED_CLOSING_RE.finditer(pr_body)]
    if hits:
        return [
            f'PR body contains {len(hits)} qualified or URL closing '
            f'reference(s) ({", ".join(repr(h) for h in hits)}). GitHub '
            f'honors these on merge but the gate cannot fold them into '
            f'the validated closing set; use the bare `Closes #N` form '
            f'(rule 1).'
        ]

    refs = find_closing_references(pr_body)
    if len(refs) != 1:
        if not refs:
            return [
                'PR body has no closing reference. Every PR must close '
                'exactly one OPEN issue (`Closes #N`, or Fixes/Resolves). '
                'A `feat` PR closes a PRD; any other type closes any '
                'OPEN issue or a PRD (rule 1).'
            ]
        return [
            f'PR body has {len(refs)} closing references '
            f'({", ".join(f"#{n}" for n in refs)}). Every PR must close '
            f'exactly one OPEN issue (rule 1).'
        ]

    number = refs[0]
    issue = fetch_issue(repo, number)
    if issue is None:
        return [
            f'issue #{number} does not exist in {repo}. The closing '
            f'reference must point at a real OPEN issue.'
        ]
    if issue.get('is_pull_request'):
        return [
            f'#{number} is a pull request, not an issue. The closing '
            f'reference must point at an OPEN issue.'
        ]

    state = str(issue.get('state', ''))
    if state.lower() != 'open':
        return [
            f'issue #{number} state is {state!r}; must be OPEN '
            f'for a PR to close it.'
        ]

    if pr_type_is_feat(pr_title):
        labels = issue.get('labels', [])
        label_names = [str(x) for x in labels] if isinstance(labels, list) else []
        if PRD_LABEL not in label_names:
            return [
                f'PR title {pr_title!r} is `feat` but issue #{number} '
                f'does not carry the `{PRD_LABEL}` label (labels: '
                f'{label_names!r}). A `feat` PR must close a PRD filed '
                f'via `.github/ISSUE_TEMPLATE/prd.yml` (rule 4).'
            ]

    return []


def main() -> int:
    parser = argparse.ArgumentParser(description='PRD gate')
    parser.add_argument('--pr-title', required=True)
    parser.add_argument('--pr-body-file', required=True)
    parser.add_argument('--repo', required=True)
    args = parser.parse_args()

    try:
        pr_body = Path(args.pr_body_file).read_text(encoding='utf-8')
    except OSError as exc:
        print(
            f'prd_gate: cannot read --pr-body-file {args.pr_body_file}: {exc}',
            file=sys.stderr,
        )
        return 2

    failures = gate(args.pr_title, pr_body, args.repo)

    if failures:
        print('PRD GATE -- FAIL')
        print()
        for msg in failures:
            print(f'  - {msg}')
        print()
        print(f'{len(failures)} failure(s). Merge blocked.')
        return 1

    print('PRD GATE -- PASS')
    return 0


if __name__ == '__main__':
    sys.exit(main())
