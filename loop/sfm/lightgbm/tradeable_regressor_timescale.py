#!/usr/bin/env python3
'''
LightGBM Tradeable Regressor - Time Architecture Sweep

Focuses on fundamental time-scale parameters:
- lookahead_minutes: How far ahead to predict (30min to 24hr)
- feature_lookback_candles: How far back to look (2hr to 24hr in 5min candles)
- (kline_size would require data re-fetching, so kept at 5min for now)

This explores whether short-term scalping or longer-term positioning works better.
'''

import numpy as np
import polars as pl
from scipy.signal import savgol_filter

from loop.sfm.lightgbm.tradeable_regressor import (
    prep as base_prep,
    params as base_params,
    CONFIG as BASE_CONFIG
)
from loop.sfm.lightgbm.tradeable_regressor import model as base_model


def params():
    """
    Parameter grid focused on time architecture.

    Reduced model param search to focus on time scales.
    """
    # Simpler model param space
    model_params = {
        'objective': ['regression'],
        'metric': ['rmse'],
        'boosting_type': ['gbdt'],
        'num_leaves': [31, 63, 127],
        'learning_rate': [0.01, 0.05, 0.1],
        'feature_fraction': [0.7, 0.8, 0.9],
        'bagging_fraction': [0.8],
        'bagging_freq': [5],
        'verbose': [-1],
        'num_iterations': [100, 200],
        'force_col_wise': [True],
    }

    # Core time architecture parameters
    time_params = {
        # Prediction horizon: 30min to 24hr
        'lookahead_minutes': [30, 60, 120, 360, 720, 1440],

        # Lookback period in candles (at 5min: 2hr to 48hr)
        'feature_lookback_candles': [24, 48, 96, 144, 288, 576],

        # Volatility lookback (should scale with feature lookback)
        'volatility_lookback_candles': [48, 96, 144, 288, 576, 720],
    }

    # SG filter params from best 1st deriv results
    sg_params = {
        'sg_window': [5, 7, 9],  # Smaller windows worked better
        'sg_polyorder': [3],
        'min_momentum_threshold': [0.00001, 0.00005],
        'min_prediction_level': [0.003],  # Strict threshold worked best
        'min_crossing_spacing': [5, 10, 15],
        'skip_boundary_bars': [10],
        'commission_rate': [0.002],
    }

    return {**model_params, **time_params, **sg_params}


def prep(data, round_params=None):
    """
    Prep with parametrized time architecture.

    Overrides CONFIG with round-specific time parameters.
    """
    if round_params is None:
        round_params = {}

    # Save original config
    original_lookahead = BASE_CONFIG['lookahead_minutes']
    original_feature_lookback = BASE_CONFIG['feature_lookback_period']
    original_vol_lookback = BASE_CONFIG['volatility_lookback_candles']
    original_vol_lookback_regular = BASE_CONFIG['volatility_lookback']

    # Override with round params
    if 'lookahead_minutes' in round_params:
        BASE_CONFIG['lookahead_minutes'] = round_params['lookahead_minutes']

    if 'feature_lookback_candles' in round_params:
        lookback = round_params['feature_lookback_candles']
        BASE_CONFIG['feature_lookback_period'] = lookback
        # Volatility lookback should be similar scale
        BASE_CONFIG['volatility_lookback'] = lookback

    if 'volatility_lookback_candles' in round_params:
        BASE_CONFIG['volatility_lookback_candles'] = round_params['volatility_lookback_candles']

    try:
        result = base_prep(data, round_params)
        return result
    finally:
        # Restore original config
        BASE_CONFIG['lookahead_minutes'] = original_lookahead
        BASE_CONFIG['feature_lookback_period'] = original_feature_lookback
        BASE_CONFIG['volatility_lookback_candles'] = original_vol_lookback
        BASE_CONFIG['volatility_lookback'] = original_vol_lookback_regular


def apply_sg_filter(predictions, window=11, polyorder=3, deriv=0):
    """Apply Savitzky-Golay filter."""
    if len(predictions) < window:
        return np.full_like(predictions, np.nan)
    return savgol_filter(predictions, window_length=window, polyorder=polyorder,
                        deriv=deriv, mode='nearest')


def detect_momentum_crossings(first_deriv, smoothed_pred, config):
    """Detect 1st derivative momentum crossings."""
    crossings = []

    for i in range(1, len(first_deriv)):
        if i < config['skip_boundary_bars'] or i >= len(first_deriv) - config['skip_boundary_bars']:
            continue

        prev_momentum = first_deriv[i-1]
        curr_momentum = first_deriv[i]

        if np.isnan(prev_momentum) or np.isnan(curr_momentum):
            continue

        if prev_momentum * curr_momentum < 0:
            if abs(prev_momentum) < config['min_momentum_threshold']:
                continue
            if smoothed_pred[i] < config['min_prediction_level']:
                continue

            if prev_momentum < 0 and curr_momentum > 0:
                crossing_type = 'positive'
            elif prev_momentum > 0 and curr_momentum < 0:
                crossing_type = 'negative'
            else:
                continue

            crossings.append({
                'bar': i,
                'type': crossing_type,
                'momentum': abs(prev_momentum),
                'prediction': smoothed_pred[i]
            })

    return pl.DataFrame(crossings) if crossings else pl.DataFrame()


def filter_crossing_spacing(crossings, min_spacing):
    """Filter crossings by minimum spacing."""
    if len(crossings) == 0:
        return crossings

    filtered = []
    last_bar = -999

    for row in crossings.iter_rows(named=True):
        if row['bar'] - last_bar >= min_spacing:
            filtered.append(row)
            last_bar = row['bar']

    return pl.DataFrame(filtered) if filtered else pl.DataFrame()


def simulate_momentum_trading(data, predictions, crossings, commission_rate):
    """Simulate LONG-only momentum trading."""
    if len(crossings) == 0:
        return {
            'total_return': 0,
            'num_trades': 0,
            'win_rate': 0,
            'avg_trade': 0,
            'sharpe_ratio': 0,
        }

    trades = []
    position = None
    prices = data['close'].to_numpy()

    for row in crossings.iter_rows(named=True):
        bar = row['bar']
        crossing_type = row['type']

        if crossing_type == 'positive' and position is None:
            position = {'entry_bar': bar, 'entry_price': prices[bar]}
        elif crossing_type == 'negative' and position is not None:
            exit_price = prices[bar]
            gross_return = (exit_price - position['entry_price']) / position['entry_price']
            net_return = gross_return - commission_rate
            trades.append({'net_return': net_return, 'bars_held': bar - position['entry_bar']})
            position = None

    if position is not None:
        bar = len(prices) - 1
        exit_price = prices[bar]
        gross_return = (exit_price - position['entry_price']) / position['entry_price']
        net_return = gross_return - commission_rate
        trades.append({'net_return': net_return, 'bars_held': bar - position['entry_bar']})

    if len(trades) == 0:
        return {'total_return': 0, 'num_trades': 0, 'win_rate': 0, 'avg_trade': 0, 'sharpe_ratio': 0}

    trades_df = pl.DataFrame(trades)
    wins = trades_df.filter(pl.col('net_return') > 0)
    returns = trades_df['net_return'].to_numpy()
    sharpe = (returns.mean() / returns.std()) if returns.std() > 0 else 0

    return {
        'total_return': trades_df['net_return'].sum(),
        'num_trades': len(trades_df),
        'win_rate': len(wins) / len(trades_df) if len(trades_df) > 0 else 0,
        'avg_trade': trades_df['net_return'].mean(),
        'sharpe_ratio': sharpe,
        'avg_bars_held': trades_df['bars_held'].mean(),
    }


def model(data, round_params):
    """
    Train model and evaluate with time-architecture aware strategy.
    """
    if round_params is None:
        round_params = {}

    sg_config = {
        'sg_window': round_params.get('sg_window', 7),
        'sg_polyorder': round_params.get('sg_polyorder', 3),
        'min_momentum_threshold': round_params.get('min_momentum_threshold', 0.00005),
        'min_prediction_level': round_params.get('min_prediction_level', 0.003),
        'min_crossing_spacing': round_params.get('min_crossing_spacing', 10),
        'skip_boundary_bars': round_params.get('skip_boundary_bars', 10),
        'commission_rate': round_params.get('commission_rate', 0.002),
    }

    # Train base model
    model_result = base_model(data, round_params)

    # Get validation predictions
    lgb_model = model_result['models'][0]
    numeric_features = model_result['extras']['numeric_features']
    val_clean = data['_val_clean']

    val_predictions = lgb_model.predict(val_clean.select(numeric_features).to_numpy())

    # Apply SG filter
    smoothed_pred = apply_sg_filter(val_predictions, sg_config['sg_window'], sg_config['sg_polyorder'], 0)
    first_deriv = apply_sg_filter(val_predictions, sg_config['sg_window'], sg_config['sg_polyorder'], 1)

    # Detect crossings
    crossings = detect_momentum_crossings(first_deriv, smoothed_pred, sg_config)
    crossings_filtered = filter_crossing_spacing(crossings, sg_config['min_crossing_spacing'])

    # Simulate trading on validation
    trading_metrics = simulate_momentum_trading(val_clean, val_predictions,
                                               crossings_filtered, sg_config['commission_rate'])

    # Test set
    test_predictions = model_result['_preds']
    test_clean = model_result['extras']['test_clean']

    test_smoothed = apply_sg_filter(test_predictions, sg_config['sg_window'], sg_config['sg_polyorder'], 0)
    test_first_deriv = apply_sg_filter(test_predictions, sg_config['sg_window'], sg_config['sg_polyorder'], 1)

    test_crossings = detect_momentum_crossings(test_first_deriv, test_smoothed, sg_config)
    test_crossings_filtered = filter_crossing_spacing(test_crossings, sg_config['min_crossing_spacing'])

    test_trading_metrics = simulate_momentum_trading(test_clean, test_predictions,
                                                     test_crossings_filtered, sg_config['commission_rate'])

    val_trading_score = -float(trading_metrics['sharpe_ratio']) if trading_metrics['num_trades'] >= 5 else 999.0

    return {
        'models': model_result['models'],
        'val_rmse': model_result['val_rmse'],
        'val_trading_score': val_trading_score,
        'n_regimes_trained': 0,
        '_preds': test_predictions,
        'universal_val_rmse': model_result['universal_val_rmse'],
        'universal_samples': model_result['universal_samples'],
        'extras': {
            **model_result['extras'],
            'sg_config': sg_config,
            'time_config': {
                'lookahead_minutes': round_params.get('lookahead_minutes', 90),
                'feature_lookback_candles': round_params.get('feature_lookback_candles', 48),
                'volatility_lookback_candles': round_params.get('volatility_lookback_candles', 720),
            },
            'val_trading_metrics': trading_metrics,
            'test_trading_metrics': test_trading_metrics,
            'val_crossings_detected': len(crossings_filtered),
            'test_crossings_detected': len(test_crossings_filtered),
        }
    }
