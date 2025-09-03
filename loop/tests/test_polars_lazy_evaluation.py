import timeit
import polars as pl

from loop.indicators import roc, atr, ppo, wilder_rsi
from loop.features import vwap, kline_imbalance
from loop.tests.utils.get_data import get_klines_data


def test_polars_lazy_evaluation_correctness():

    data = get_klines_data()

    transformations = {
        'indicators': [
            (roc, {'period': 12}),
            (atr, {}),
            (ppo, {}),
            (wilder_rsi, {}),
        ],
        'features': [
            (vwap, {}),
            (kline_imbalance, {})
        ],
        'filters': [
            ~pl.col('roc_12').is_null(),
            ~pl.col('wilder_rsi_14').is_null()
        ]
    }

    imperative_result = _imperative_evaluation(data.clone())
    declarative_result = _lazy_evaluation(data.clone(), transformations)

    assert imperative_result.shape == declarative_result.shape
    assert set(imperative_result.columns) == set(declarative_result.columns)
    assert imperative_result.equals(declarative_result)


def test_polars_lazy_evaluation_performance():

    data = get_klines_data()

    transformations = {
        'indicators': [
            (roc, {'period': 12}),
            (atr, {}),
            (ppo, {}),
            (wilder_rsi, {}),
        ],
        'features': [
            (vwap, {}),
            (kline_imbalance, {})
        ],
        'filters': [
            ~pl.col('roc_12').is_null(),
            ~pl.col('wilder_rsi_14').is_null()
        ]
    }

    num_iter = 3
    imperative_time = timeit.timeit(
        lambda: _imperative_evaluation(data.clone()),
        number=num_iter
    ) / num_iter

    declarative_time = timeit.timeit(
        lambda: _lazy_evaluation(data.clone(), transformations),
        number=num_iter
    ) / num_iter

    assert imperative_time > declarative_time * 1.25


def _lazy_evaluation(data, config):

    lazy_data = data.lazy()

    if 'indicators' in config:
        for func, params in config['indicators']:
            lazy_data = lazy_data.pipe(func, **params)

    if 'features' in config:
        for func, params in config['features']:
            lazy_data = lazy_data.pipe(func, **params)

    if 'filters' in config and config['filters']:
        combined_filter = config['filters'][0]
        for filter_expr in config['filters'][1:]:
            combined_filter = combined_filter & filter_expr
        lazy_data = lazy_data.filter(combined_filter)

    return lazy_data.collect()

def _imperative_evaluation(data):

    data = roc(data, period=12)
    data = atr(data)
    data = ppo(data)
    data = wilder_rsi(data)
    data = vwap(data)
    data = kline_imbalance(data)

    return data.filter(
        ~pl.col('roc_12').is_null() &
        ~pl.col('wilder_rsi_14').is_null()
    )