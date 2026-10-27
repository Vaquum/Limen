from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import polars as pl
import pytest

from limen import Log

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'tests/fixtures/spot_1h_20240101_20241231.parquet'


@pytest.mark.parametrize('method', ['pearson', 'spearman', 'kendall'])
def test_recorded_polars_bootstrap_matches_pandas_oracle(method):
    frame = pl.read_parquet(FIXTURE, n_rows=180).select('close', 'volume', 'high')
    reference = frame.to_pandas().sort_values('close', ascending=False).reset_index(drop=True)
    result = Log(uel_object=SimpleNamespace(experiment_log=frame)).experiment_parameter_correlation(
        'close', method=method, heads=(1.0, 0.5), n_boot=8, min_n=2, random_state=42)
    rng = np.random.default_rng(42)
    for size in (180, 90):
        cohort = reference.head(size)
        base = cohort.corr(method=method)['close']
        boot = pd.concat([cohort.iloc[rng.integers(0, size, size)].corr(method=method)['close']
                          for _ in range(8)], axis=1)
        median = boot.median(axis=1)
        for feature in ('volume', 'high'):
            row = result.filter((pl.col('cohort_pct') == 100 * size / 180)
                                & (pl.col('feature') == feature)).row(0, named=True)
            expected = [base[feature], median[feature], boot.loc[feature].quantile(0.025),
                        boot.loc[feature].quantile(0.975),
                        (np.sign(boot.loc[feature]) == np.sign(median[feature])).mean()]
            np.testing.assert_allclose([row[key] for key in ('corr', 'corr_med', 'ci_lo', 'ci_hi', 'sign_stability')],
                                       expected, rtol=1e-12, atol=1e-12)


def test_log_analysis_runs_with_pandas_imports_blocked():
    script = '''
import importlib.abc
import sys
from types import SimpleNamespace
class BlockPandas(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'pandas' or fullname.startswith('pandas.'):
            raise ModuleNotFoundError('pandas is unavailable')
sys.meta_path.insert(0, BlockPandas())
import polars as pl
from limen import Log
frame = pl.read_parquet(sys.argv[1], n_rows=120).select('close', 'volume')
log = Log(uel_object=SimpleNamespace(experiment_log=frame))
result = log.experiment_parameter_correlation('close', heads=(1.0,), n_boot=3, min_n=2)
assert isinstance(result, pl.DataFrame) and result.height == 1
assert 'pandas' not in sys.modules
'''
    result = subprocess.run([sys.executable, '-c', script, str(FIXTURE)], cwd=ROOT,
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
