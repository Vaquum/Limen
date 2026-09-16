from __future__ import annotations

import pytest

from governance import version_gate
from governance._common import REPO_ROOT


def _pyproject(version: str) -> str:
    return f'[project]\nname = "demo"\nversion = "{version}"\n'


def _changelog(version: str, body: str = '- Change.\n') -> str:
    return f'## [1.2.3] - 2026-01-01\n\n- Previous.\n\n## [{version}] - 2026-01-02\n\n{body}'


def test_gate_accepts_feat_with_minor_bump_and_changelog_content() -> None:
    failures = version_gate.gate(
        'feat: add law template',
        _pyproject('1.2.3'),
        _pyproject('1.3.0'),
        _changelog('1.2.3'),
        _changelog('1.3.0'),
    )
    assert failures == []


def test_gate_rejects_feat_with_patch_bump() -> None:
    failures = version_gate.gate(
        'feat: add law template',
        _pyproject('1.2.3'),
        _pyproject('1.2.4'),
        _changelog('1.2.3'),
        _changelog('1.2.4'),
    )
    assert any('requires at least a minor version bump' in item for item in failures)


def test_gate_requires_new_changelog_header_at_top() -> None:
    failures = version_gate.gate(
        'fix: tighten law template',
        _pyproject('1.2.3'),
        _pyproject('1.2.4'),
        _changelog('1.2.3'),
        '## [1.2.4] - 2026-01-02\n\n- New but misplaced.\n\n## [1.2.3] - 2026-01-01\n\n- Previous.\n',
    )
    assert any('1.2.3' in item for item in failures)


def test_gate_rejects_empty_top_changelog_section() -> None:
    failures = version_gate.gate(
        'fix: tighten law template',
        _pyproject('1.2.3'),
        _pyproject('1.2.4'),
        _changelog('1.2.3'),
        '## [1.2.3] - 2026-01-01\n\n- Previous.\n\n## [1.2.4] - 2026-01-02\n',
    )
    assert any('has no content before the next version header' in item for item in failures)


def test_strict_semver_rejects_prerelease_versions() -> None:
    with pytest.raises(SystemExit) as exc:
        version_gate.parse_semver('1.2.4-alpha')
    assert exc.value.code == 2


def test_gate_rejects_past_tense_changelog_bullet() -> None:
    failures = version_gate.gate(
        'fix: tighten law template',
        _pyproject('1.2.3'),
        _pyproject('1.2.4'),
        _changelog('1.2.3'),
        _changelog('1.2.4', body='- Added a new helper.\n'),
    )
    assert any('past tense' in item for item in failures)


def test_gate_rejects_changelog_placeholder() -> None:
    failures = version_gate.gate(
        'fix: tighten law template',
        _pyproject('1.2.3'),
        _pyproject('1.2.4'),
        _changelog('1.2.3'),
        _changelog('1.2.4', body='- Fix the parser (TODO: expand later).\n'),
    )
    assert any('placeholder' in item for item in failures)


def test_gate_accepts_imperative_changelog_bullet() -> None:
    failures = version_gate.gate(
        'fix: tighten law template',
        _pyproject('1.2.3'),
        _pyproject('1.2.4'),
        _changelog('1.2.3'),
        _changelog('1.2.4', body='- Fix the broken parser path.\n'),
    )
    assert failures == []


def test_gate_rejects_version_rewind_without_marker() -> None:
    failures = version_gate.gate(
        'ci(version): return to the 5.x line',
        _pyproject('7.0.0'),
        _pyproject('5.10.3'),
        _changelog('7.0.0'),
        _changelog('5.10.3'),
    )
    assert any('version did not move forward' in item for item in failures)


def test_gate_accepts_version_rewind_with_marker() -> None:
    failures = version_gate.gate(
        'ci(version): return to the 5.x line',
        _pyproject('7.0.0'),
        _pyproject('5.10.3'),
        _changelog('7.0.0'),
        _changelog('5.10.3'),
        pr_body='Body.\n\n[version-rewind: neither 6.0.0 nor 7.0.0 changed the public API]\n',
    )
    assert failures == []


def test_gate_rejects_rewind_marker_without_rewind() -> None:
    failures = version_gate.gate(
        'fix: tighten law template',
        _pyproject('1.2.3'),
        _pyproject('1.2.4'),
        _changelog('1.2.3'),
        _changelog('1.2.4'),
        pr_body='[version-rewind: idle marker]\n',
    )
    assert any('does not go down' in item for item in failures)


def test_gate_rejects_major_bump_without_marker() -> None:
    failures = version_gate.gate(
        'feat!: replace the public API',
        _pyproject('1.2.3'),
        _pyproject('2.0.0'),
        _changelog('1.2.3'),
        _changelog('2.0.0'),
    )
    assert any('without `[major-release: <reason>]`' in item for item in failures)


def test_gate_accepts_major_bump_with_marker() -> None:
    failures = version_gate.gate(
        'feat!: replace the public API',
        _pyproject('1.2.3'),
        _pyproject('2.0.0'),
        _changelog('1.2.3'),
        _changelog('2.0.0'),
        pr_body='[major-release: the Trainer constructor signature changes]\n',
    )
    assert failures == []


def test_constitution_names_both_markers() -> None:
    lines = (REPO_ROOT / 'CLAUDE.md').read_text(encoding='utf-8').splitlines()
    law = next(line for line in lines if line.startswith('5. '))
    assert law.count('[version-rewind: <reason>]') == 1
    assert law.count('[major-release: <reason>]') == 1
