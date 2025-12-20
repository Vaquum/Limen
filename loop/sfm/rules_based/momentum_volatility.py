from loop.historical_data import HistoricalData
from loop.tests.utils.get_data import get_klines_data_small_fast
from loop.manifest import Manifest
from loop.sfm.model.momentum_volatility import momentum_volatility


def params():
    return {
        'window_size': [24, 48, 72],
        'momentum_buy_pct': [55, 60, 65, 70, 75, 80],
        'momentum_sell_pct': [30, 35, 40, 45, 50, 55],
        'momentum_short_pct': [20, 25, 30, 35, 40, 45],
        'momentum_cover_pct': [50, 55, 60, 65, 70],
        'volatility_entry_pct': [70, 75, 80, 85, 90],
        'volatility_exit_pct': [80, 85, 90, 95],
        'lookback_window': [300, 500, 750],
        'trading_cost': [0.001]
    }


def manifest():
    return (Manifest()
        .set_data_source(
            method=HistoricalData.get_spot_klines,
            params={'kline_size': 3600, 'start_date_limit': '2025-01-01'}
        )
        .set_test_data_source(method=get_klines_data_small_fast)
        .set_split_config(80, 10, 10)
        .with_model(momentum_volatility)
    )
