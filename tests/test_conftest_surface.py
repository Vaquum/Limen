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
