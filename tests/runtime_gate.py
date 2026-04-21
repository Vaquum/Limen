from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tests.utils.runtime_tracking import evaluate_runtime_budget
from tests.utils.runtime_tracking import load_runtime_budget
from tests.utils.runtime_tracking import load_runtime_profile
from tests.utils.runtime_tracking import render_runtime_summary_markdown
from tests.utils.runtime_tracking import write_runtime_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Render Limen test runtime summaries and enforce suite budgets.',
    )
    parser.add_argument(
        '--profile',
        required=True,
        help='Path to the test runtime profile JSON emitted by tests.run.',
    )
    parser.add_argument(
        '--budget',
        required=True,
        help='Path to the committed runtime budget JSON.',
    )
    parser.add_argument(
        '--summary-file',
        help='Optional path for the rendered markdown summary.',
    )
    parser.add_argument(
        '--enforce',
        action='store_true',
        help='Fail with a non-zero exit code when the suite exceeds the budget.',
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile = load_runtime_profile(Path(args.profile))
    budget = load_runtime_budget(Path(args.budget))
    summary_markdown = render_runtime_summary_markdown(profile, budget=budget)

    if args.summary_file:
        write_runtime_summary(Path(args.summary_file), summary_markdown)

    sys.stdout.write(summary_markdown + '\n')

    if not args.enforce:
        return 0

    verdict = evaluate_runtime_budget(profile, budget)
    if verdict['within_budget']:
        sys.stdout.write(
            'Runtime budget passed: '
            f"{verdict['observed_total_seconds']:.3f}s <= "
            f"{verdict['max_total_seconds']:.3f}s\n",
        )
        return 0

    sys.stderr.write(
        'Runtime budget exceeded: '
        f"{verdict['observed_total_seconds']:.3f}s > "
        f"{verdict['max_total_seconds']:.3f}s "
        f"(over by {verdict['overage_seconds']:.3f}s)\n",
    )
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
