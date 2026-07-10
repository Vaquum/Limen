from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_darwin_openmp_thread_cap_is_pinned() -> None:
    conftest = (ROOT / 'tests' / 'conftest.py').read_text(encoding='utf-8')

    assert conftest.count('OMP_NUM_THREADS') == 1
    assert conftest.count('darwin') == 1
    assert "os.environ.setdefault('OMP_NUM_THREADS', '1')" in conftest
    assert "if sys.platform == 'darwin':" in conftest
