import polars as pl

from limen.data import HistoricalData
from limen.experiment import Manifest
from limen.indicators.window_return import window_return
from limen.indicators.sma import sma
from limen.indicators.rolling_volatility import rolling_volatility
from limen.indicators.stochastic_oscillator import stochastic_oscillator
from limen.indicators.body_pct import body_pct
from limen.features.range_pct import range_pct
from limen.features.volume_ratio import volume_ratio
from limen.features.sma_ratios import sma_ratios
from limen.features.lagged_features import lag_columns
from limen.sfd.reference_architecture import xgboost_regressor


def params():
    return {
        'learning_rate': [0.01, 0.02, 0.03],
        'max_depth': [2, 3, 4],
        'n_estimators': [300, 500, 700],
        'min_child_weight': [5, 10, 20],
        'subsample': [0.5, 0.6, 0.7],
        'colsample_bytree': [0.5, 0.6, 0.7],
        'gamma': [0.1, 0.5, 1.0],
        'reg_alpha': [0.1, 0.5, 1.0],
        'reg_lambda': [1.0, 5.0, 10.0],
        'objective': ['reg:squarederror'],
        'booster': ['gbtree'],
        'early_stopping_rounds': [50],
    }


def manifest():

    return (
        Manifest()
        .set_data_source(
            method=HistoricalData.get_spot_klines,
            params={'kline_size': 3600, 'start_date_limit': '2025-01-01'}
        )
        .set_test_data_source(method=HistoricalData._get_data_for_test)
        .set_split_config(8, 1, 2)
        .add_indicator(window_return, period=1)
        .add_indicator(window_return, period=5)
        .add_indicator(window_return, period=20)
        .add_indicator(rolling_volatility, column='ret_1', window=20)
        .add_indicator(sma, column='volume', period=20)
        .add_indicator(sma, column='maker_ratio', period=10)
        .add_indicator(sma, column='no_of_trades', period=20)
        .add_indicator(stochastic_oscillator)
        .add_feature(body_pct)
        .add_feature(range_pct)
        .add_feature(volume_ratio, period=20)
        .add_feature(sma_ratios, periods=[10, 50], price_col='close')
        .add_feature(
            lambda df: df.with_columns([
                (pl.col('volume') * pl.col('close')).alias('dollar_volume'),
                (pl.col('maker_ratio') - 0.5).alias('order_flow_imbalance'),
                (pl.col('no_of_trades') / pl.col('no_of_trades_sma_20')).alias('trade_intensity'),
            ])
        )
        .add_feature(lag_columns, cols=[
            'range_pct',
            'volume_sma_20',
            'volume_ratio',
            'dollar_volume',
            'maker_ratio_sma_10',
            'trade_intensity',
            'body_pct',
            'order_flow_imbalance',
            'stoch_k',
        ], lag=1)
        .with_target('next_return')
            .add_transform(
                lambda df: df.with_columns([
                    (((pl.col('close').shift(-1) - pl.col('close')) / pl.col('close')) * 100).alias('next_return')
                ])
            )
            .done()
        .with_model(xgboost_regressor)
    )
