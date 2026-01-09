import time
from limen.utils.param_space import ParamSpace

def test_large_param_space():
    params = {
        'shift': [-1, -2, -3, -4, -5, -6, -7, -8, -9, -10],
        'q': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        'roc_period': [1, 2, 4, 8, 12, 16, 24, 48, 72, 96, 144, 288],
        'penalty': ['l1', 'l2', 'elasticnet', 'none'],
        'bar_type': ['base', 'time', 'volume', 'liquidity', 'tick', 'volume_imbalance', 'liquidity_imbalance'],
        'time_freq': ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d'],
        'volume_threshold': [100000, 500000, 1000000, 2000000, 5000000, 10000000],
        'liquidity_threshold': [1000000, 10000000, 50000000, 100000000, 500000000],
        'class_weight': [0.1, 0.3, 0.45, 0.55, 0.65, 0.75, 0.85, 0.9],
        'C': [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0],
        'max_iter': [10, 30, 60, 90, 120, 180, 240, 360, 480],
        'solver': ['lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga', 'newton-cholesky'],
        'tol': [0.0001, 0.001, 0.01, 0.03, 0.1, 0.3, 1.0],
        'feature_selection': ['none', 'univariate', 'recursive', 'l1', 'tree'],
        'scaler': ['standard', 'minmax', 'robust', 'quantile', 'power'],
    }

    start = time.time()
    ps = ParamSpace(params, 1000000)
    elapsed = time.time() - start

    assert ps.n_permutations == 1000000
    assert elapsed < 10.0

    combo = ps.generate()
    assert set(combo.keys()) == set(params.keys())
    for key, value in combo.items():
        assert value in params[key]