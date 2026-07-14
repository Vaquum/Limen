from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]


def _load_create_release() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        'create_release',
        ROOT / 'scripts' / 'create_release.py',
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_release_tag_authority_and_traceability() -> None:
    module = _load_create_release()

    assert module.TAG_RE.pattern == r'^v\d+\.\d+\.\d+$'
    assert module.TAG_RE.match('v5.1.0')
    for rejected in ('5.1.0', 'v5.1', 'v5.1.0rc1', 'V5.1.0', 'v5.1.0 '):
        assert module.TAG_RE.match(rejected) is None, rejected

    version = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))['project']['version']
    assert module.compute_tag(version, {}) == f'v{version}'
    assert module.compute_tag(version, {'tag': f'v{version}', 'version': version}) == f'v{version}'
    with pytest.raises(ValueError, match='model may not choose identifiers'):
        module.compute_tag(version, {'tag': 'v99.0.0'})
    with pytest.raises(ValueError, match='model may not choose identifiers'):
        module.compute_tag(version, {'version': '99.0.0'})
    with pytest.raises(ValueError, match='model may not choose identifiers'):
        module.compute_tag('5.1.0rc1', {})

    module.validate_release_prose({'release_name': 'Fire Horse', 'release_notes': '## Summary'})
    for malformed in ({}, {'release_name': 'x'}, {'release_name': '', 'release_notes': 'y'}, {'release_name': 'x', 'release_notes': 3}):
        with pytest.raises(ValueError, match='non-empty string release_'):
            module.validate_release_prose(malformed)

    changelog = (ROOT / 'CHANGELOG.md').read_text(encoding='utf-8')
    anchor = module.changelog_anchor(version, changelog)
    assert anchor.startswith(''.join(version.split('.')))
    assert module.changelog_anchor('9.9.9', '## [9.9.9] - 2026-01-02\n') == '999---2026-01-02'
    with pytest.raises(ValueError, match=r'create_release CHANGELOG\.md has no entry'):
        module.changelog_anchor('0.0.0', changelog)

    assert module.extract_pr_numbers([
        'Merge pull request #687 from Vaquum/feat/687-release-supply-chain-hardening',
        'test: decouple docs inventory from formatting (#689)',
        'docs: mention #123 in passing',
        'chore: no reference',
    ]) == [687, 689]

    appendix = module.build_traceability('v5.1.0', 'v5.0.0', [687], '510---2026-07-14')
    assert appendix.startswith('## Traceability')
    assert '- Merged pull requests: #687' in appendix
    assert '- Compare: https://github.com/Vaquum/Limen/compare/v5.0.0...v5.1.0' in appendix
    assert '- Changelog: https://github.com/Vaquum/Limen/blob/v5.1.0/CHANGELOG.md#510---2026-07-14' in appendix
    assert 'SHA-256' in appendix

    first_release = module.build_traceability('v0.1.0', None, [], 'anchor')
    assert 'Compare:' not in first_release
    assert 'Merged pull requests:' not in first_release
