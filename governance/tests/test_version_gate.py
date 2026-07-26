from __future__ import annotations

import pytest

from governance import version_gate


def _pyproject(version: str) -> str:
    return f'[project]\nname = "demo"\nversion = "{version}"\n'


def _changelog(version: str, body: str = '- Change.\n') -> str:
    return f'# v{version}\n\n{body}\n# v1.2.3\n\n- Previous.\n'


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
        '# v1.2.3\n\n- Previous.\n\n# v1.2.4\n\n- New but misplaced.\n',
    )
    assert any('top version header is `# v1.2.3`' in item for item in failures)


def test_gate_rejects_empty_top_changelog_section() -> None:
    failures = version_gate.gate(
        'fix: tighten law template',
        _pyproject('1.2.3'),
        _pyproject('1.2.4'),
        _changelog('1.2.3'),
        '# v1.2.4\n\n# v1.2.3\n\n- Previous.\n',
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
