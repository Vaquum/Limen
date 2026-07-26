#!/usr/bin/env python3
"""File size balance gate: largest <= MAX_RATIO x median."""
from __future__ import annotations

import statistics
import sys
from pathlib import Path
from typing import Final

from _common import REPO_ROOT, resolve_package_dir

# A deliberately lenient bootstrap default: 16x lets a young package carry a
# framework-boundary module or two that must stay physically together, while
# still blocking an accidental "one huge file" architecture. The gate is
# dormant below MIN_FILES_FOR_GATE files, so it only bites once a package has
# real structure; tighten the ratio as it grows and the largest-file
# exception stops reflecting a real boundary.
MAX_RATIO: Final[float] = 16.00
MIN_FILES_FOR_GATE: Final[int] = 3


def count_lines(path: Path) -> int:
    # Total physical line count, blank lines included (file-size balance
    # cares about actual file size on disk, not logical SLOC).
    return len(path.read_text(encoding='utf-8').splitlines())


def main() -> int:
    source_dir = resolve_package_dir('FILE SIZE BALANCE GATE')
    sized: list[tuple[Path, int]] = [
        (p, count_lines(p)) for p in sorted(source_dir.rglob('*.py'))
    ]
    if len(sized) < MIN_FILES_FOR_GATE:
        # Too few files to have a size imbalance. The scan target exists
        # (resolved and checked above); this is a legitimately small
        # package, not a misconfiguration.
        print(
            f'FILE SIZE BALANCE GATE -- PASS '
            f'(only {len(sized)} source file(s), need >= {MIN_FILES_FOR_GATE} to balance)'
        )
        return 0
    # Exclude zero-line files from the median: a package with many
    # empty __init__.py files would otherwise produce median=0 and
    # ratio=inf for the smallest real file. Empty files don't have a
    # meaningful size to balance against.
    nonzero_sizes = [s for _, s in sized if s > 0]
    if len(nonzero_sizes) < MIN_FILES_FOR_GATE:
        print(
            f'FILE SIZE BALANCE GATE -- PASS '
            f'(only {len(nonzero_sizes)} non-empty source file(s) to balance)'
        )
        return 0
    largest_path, largest_size = max(sized, key=lambda item: item[1])
    median = statistics.median(nonzero_sizes)
    ratio = largest_size / median
    if ratio > MAX_RATIO:
        print('FILE SIZE BALANCE GATE -- FAIL', file=sys.stderr)
        print('', file=sys.stderr)
        rel = largest_path.relative_to(REPO_ROOT)
        print(f'  largest file:    {rel} ({largest_size} lines)', file=sys.stderr)
        print(f'  median file size: {int(median)} lines', file=sys.stderr)
        print(
            f'  ratio:            {ratio:.2f} (max allowed: {MAX_RATIO:.2f})',
            file=sys.stderr,
        )
        print('', file=sys.stderr)
        print('Merge blocked.', file=sys.stderr)
        return 1
    print(
        f'FILE SIZE BALANCE GATE -- PASS '
        f'(largest={largest_size}, median={int(median)}, ratio={ratio:.2f})'
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
