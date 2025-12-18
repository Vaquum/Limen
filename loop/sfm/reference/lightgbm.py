import polars as pl

from loop.historical_data import HistoricalData
from loop.tests.utils.get_data import get_klines_data_large
from loop.features.time_features import time_features
from loop.features.lagged_features import lag_range_cols
from loop.indicators.sma import sma
from loop.utils.add_breakout_ema import add_breakout_ema
from loop.utils.random_slice import random_slice
from loop.manifest import Manifest
from loop.data import compute_data_bars
from loop.sfm.model import lgb_binary


TARGET_COLUMN = 'liquidity_sum'
LOOKBACK_WINDOW_SIZE = 100
PREDICTION_HORIZON = 3
NUM_SLICES = 10
TARGET_COLUMN_CLASS = 'breakout_ema'


def params():

    return {
        'random_slice_size': [5000],
        'random_slice_min_pct': [0.10],
        'random_slice_max_pct': [0.90],
        'random_seed': [42],
        'bar_type': ['base', 'trade', 'volume', 'liquidity'],
        'trade_threshold': [5000, 10000, 30000, 100000, 500000],
        'volume_threshold': [100, 250, 500, 750, 1000, 5000],
        'liquidity_threshold': [50000, 1000000, 5000000, 50000000, 100000000],
        'objective': ['binary'],
        'metric': ['auc'],
        'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2],
        'num_leaves': [15, 31, 63, 127, 255],
        'max_depth': [3, 5, 7, 9, -1],
        'min_data_in_leaf': [20, 50, 100, 200, 500],
        'feature_fraction': [0.6, 0.7, 0.8, 0.9, 1.0],
        'bagging_fraction': [0.6, 0.7, 0.8, 0.9, 1.0],
        'bagging_freq': [0, 1, 5, 10, 20],
        'lambda_l1': [0.0, 0.1, 1.0, 10.0, 100.0],
        'lambda_l2': [0.0, 0.1, 1.0, 10.0, 100.0],
        'feature_pre_filter': ['false'],
    }


def manifest():

    return (Manifest()
        .set_data_source(
            method=HistoricalData.get_spot_klines,
            params={'kline_size': 3600, 'start_date_limit': '2024-01-01'}
        )
        .set_test_data_source(method=get_klines_data_large)
        .set_pre_split_data_selector(
            random_slice,
            rows='random_slice_size',
            safe_range_low='random_slice_min_pct',
            safe_range_high='random_slice_max_pct',
            seed='random_seed'
        )
        .set_split_config(6, 2, 2)
        .set_bar_formation(compute_data_bars,
            bar_type='bar_type',
            trade_threshold='trade_threshold',
            volume_threshold='volume_threshold',
            liquidity_threshold='liquidity_threshold')
        .set_required_bar_columns([
            'datetime', 'high', 'low', 'open', 'close', 'mean',
            'volume', 'maker_ratio', 'no_of_trades', 'maker_volume', 'maker_liquidity'
        ])
        .add_feature(time_features)
        .add_indicator(sma, column='liquidity_sum', period=3)
        .add_feature(lambda data: data.with_columns([
            pl.col('liquidity_sum').ewm_mean(span=5).alias('liq_ewm_5'),
            pl.col('liquidity_sum').ewm_mean(span=10).alias('liq_ewm_10')
        ]))
        .add_feature(lag_range_cols,
            cols=['liquidity_sum', 'hour', 'minute', 'weekday',
                  'liquidity_sum_sma_3', 'liq_ewm_5', 'liq_ewm_10'],
            start=1,
            end=LOOKBACK_WINDOW_SIZE)
        .add_feature(lambda data: data.with_columns([
            (pl.col('liquidity_sum') - pl.col('liquidity_sum_lag_1')).alias('delta_liq')
        ]))
        .with_target(TARGET_COLUMN_CLASS)
            .add_transform(add_breakout_ema, target_col=TARGET_COLUMN)
            .done()
        .with_model(lgb_binary)
    )
