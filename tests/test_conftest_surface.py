from __future__ import annotations

from pathlib import Path
import os
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_darwin_openmp_thread_cap_is_pinned() -> None:
    conftest = (ROOT / 'tests' / 'conftest.py').read_text(encoding='utf-8')

    assert conftest.count('OMP_NUM_THREADS') == 1
    assert conftest.count('darwin') == 1
    assert "os.environ.setdefault('OMP_NUM_THREADS', '1')" in conftest
    assert "if sys.platform == 'darwin':" in conftest


@pytest.mark.parametrize('thread_count', [None, '2'])
def test_polars_thread_budget_is_set_before_import(thread_count: str | None) -> None:
    import polars as pl

    if os.environ['POLARS_MAX_THREADS'] == '1':
        assert pl.thread_pool_size() == 1
    environment = os.environ.copy()
    if thread_count is None:
        environment.pop('POLARS_MAX_THREADS', None)
    else:
        environment['POLARS_MAX_THREADS'] = thread_count
    code = "import runpy; runpy.run_path('tests/conftest.py'); import polars as pl; print(pl.thread_pool_size())"
    actual = subprocess.check_output([sys.executable, '-c', code], cwd=ROOT, env=environment, text=True)
    assert actual.strip() == (thread_count or '1')


@pytest.mark.parametrize('thread_count', [None, '2'])
def test_complete_proof_runner_preserves_polars_override(tmp_path: Path, thread_count: str | None) -> None:
    test_path = tmp_path / 'test_polars_pool.py'
    test_path.write_text(
        "import polars as pl\ndef test_pool():\n    assert pl.thread_pool_size() == "
        + (thread_count or '1') + "\n", encoding='utf-8',
    )
    environment = os.environ.copy()
    environment['LIMEN_RUNTIME_PROFILE_PATH'] = str(tmp_path / 'runtime.json')
    if thread_count is None:
        environment.pop('POLARS_MAX_THREADS', None)
    else:
        environment['POLARS_MAX_THREADS'] = thread_count
    result = subprocess.run(
        [sys.executable, '-m', 'tests.run', str(test_path), '-q'],
        cwd=ROOT, env=environment, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_cached_yaml_setup_preserves_content_and_nested_isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.utils import yaml_config as helper

    helper._parsed.cache_clear()
    parse = helper.parse
    calls = []

    def observe(text):
        calls.append(text)
        return parse(text)

    monkeypatch.setattr(helper, 'parse', observe)
    for template in ('ridge_regressor', 'logreg_binary'):
        text = (ROOT / 'limen/yaml/templates' / (template + '.yaml')).read_text()
        expected, errors = parse(text)
        assert not errors
        first, errors = helper.parse_cached(text)
        assert not errors
        assert first == expected
        first['sfd']['manifest']['backtest']['fee_bps'] = -999
        second, errors = helper.parse_cached(text)
        assert not errors
        assert second == expected
        assert second['sfd'] is not first['sfd']
    assert len(calls) == len(set(calls)) == 2
