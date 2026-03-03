#!/usr/bin/env python3

import argparse
import re
from datetime import datetime
from pathlib import Path


SEMVER_PATTERN = re.compile(r'^(\d+)\.(\d+)\.(\d+)$')
VERSION_FILE = Path('limen/_version.py')
CHANGELOG_FILE = Path('CHANGELOG.md')


def _validate_semver(version: str) -> None:
    if not SEMVER_PATTERN.match(version):
        raise ValueError('Version must follow semantic versioning: MAJOR.MINOR.PATCH')


def _ordinal(day: int) -> str:
    if 10 <= day % 100 <= 20:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th')
    return f'{day}{suffix}'


def _release_date_text(now: datetime) -> str:
    day = _ordinal(now.day)
    month = now.strftime('%B')
    year = now.year
    return f'{day} of {month}, {year}'


def update_version_file(version: str) -> None:
    VERSION_FILE.write_text(f"__version__ = '{version}'\n")


def append_changelog_entry(version: str) -> None:
    if not CHANGELOG_FILE.exists():
        CHANGELOG_FILE.write_text('# Changelog\n\n')

    now = datetime.now()
    header = f'## v{version} on {_release_date_text(now)}\n\n'
    body = '- TBD\n'

    changelog = CHANGELOG_FILE.read_text().rstrip()
    updated = f'{changelog}\n\n{header}{body}\n'
    CHANGELOG_FILE.write_text(updated)


def main() -> None:
    parser = argparse.ArgumentParser(description='Bump project version and update changelog.')
    parser.add_argument('version', help='New semantic version (e.g. 1.38.1)')
    parser.add_argument(
        '--skip-changelog',
        action='store_true',
        help='Do not append a new changelog entry',
    )
    args = parser.parse_args()

    _validate_semver(args.version)
    update_version_file(args.version)
    if not args.skip_changelog:
        append_changelog_entry(args.version)

    print(f'Updated project version to {args.version}')
    if not args.skip_changelog:
        print('Appended changelog entry')


if __name__ == '__main__':
    main()
