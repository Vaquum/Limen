from inspect import signature
from typing import cast

from limen.data import HistoricalData
from limen.experiment import MLManifest
from limen.indicators.window_return import window_return
from limen.features.lagged_features import lag_columns
from limen.sfd.foundational_sfd._xgboost_features import (
    momentum as _momentum,
    structure as _structure,
    trend as _trend,
    volatility as _volatility,
    volume as _volume,
)
from limen.sfd.reference_architecture.xgboost_regressor import xgboost_regressor
from limen.targets import NextReturnTarget

__all__ = ['manifest', 'params']


def params() -> dict[str, list[object]]:
    """Expose model defaults and a discrete research space over existing features."""
    return {
        **{key: [param.default] for key, param in signature(xgboost_regressor).parameters.items() if key != 'data'},
        'learning_rate': [0.01, 0.02, 0.03],
        'max_depth': [2, 3, 4],
        'n_estimators': [300, 500, 700],
        'min_child_weight': [5, 10, 20],
        'subsample': [0.5, 0.6, 0.7],
        'colsample_bytree': [0.5, 0.6, 0.7],
        'gamma': [0.1, 0.5, 1.0],
        'reg_alpha': [0.1, 0.5, 1.0],
        'reg_lambda': [1.0, 5.0, 10.0],
        'eval_metric': ['rmse', 'mae'],
        'max_leaves': [0, 31],
        'max_bin': [128, 256],
        'tree_method': ['hist'],
        'device': ['cpu'],
        'n_jobs': [1],
        'target_horizon': [1, 3, 6],
        'return_fast': [1, 2],
        'return_medium': [5, 10],
        'return_slow': [20, 40],
        'volatility_window': [20, 40],
        'volume_window': [20, 40],
        'flow_window': [10, 20],
        'stochastic_k': [14, 21],
        'stochastic_d': [3, 5],
        'rsi_period': [14, 21],
        'price_sma_fast': [10, 20],
        'price_sma_slow': [50, 100],
        'price_range_window': [20, 40],
        'bollinger_window': [20, 40],
        'bollinger_std': [2.0, 2.5],
        'volume_regime_window': [48, 96],
        'lag': [1, 2, 3],
        'feature_groups': ['all', 'momentum|trend', 'volatility|volume|structure'],
        'scaler_type': ['robust', 'linear'],
        'feature_drop_count': [0, 1],
        'feature_drop_seed': [42],
    }


def manifest() -> MLManifest:
    """Build independently selectable research groups with per-round preparation."""
    base = (MLManifest()
        .set_data_source(HistoricalData.get_spot_klines, params={'kline_size': 3600, 'start_date_limit': '2025-01-01'})
        .set_test_data_source(HistoricalData.get_spot_klines, params={'kline_size': 7200, 'row_count_limit': 5000})
        .set_split_config(8, 1, 2)
        .add_indicator(window_return, period=1)
        .add_feature(lag_columns, cols=['ret_1'], lag='lag')
        .add_feature(_momentum, return_fast='return_fast', return_medium='return_medium', return_slow='return_slow', stochastic_k='stochastic_k', stochastic_d='stochastic_d', rsi_period='rsi_period', lag='lag', group='momentum')
        .add_feature(_trend, price_sma_fast='price_sma_fast', price_sma_slow='price_sma_slow', lag='lag', group='trend')
        .add_feature(_volume, volume_window='volume_window', flow_window='flow_window', volume_regime_window='volume_regime_window', lag='lag', group='volume')
        .add_feature(_volatility, volatility_window='volatility_window', bollinger_window='bollinger_window', bollinger_std='bollinger_std', lag='lag', group='volatility')
        .add_feature(_structure, price_range_window='price_range_window', lag='lag', group='structure')
        .with_target_label('next_return', NextReturnTarget, transform_params={'periods': 'target_horizon'})
    )
    return (cast(MLManifest, base)
        .set_scaler_from_params('scaler_type')
        .set_feature_ablation()
        .set_strict_mode(True)
        .with_reference_architecture(xgboost_regressor)
    )
