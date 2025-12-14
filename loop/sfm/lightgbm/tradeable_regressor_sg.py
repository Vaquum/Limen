#!/usr/bin/env python3
'''
LightGBM Tradeable Regressor with SG Filter Trading - UEL Compatible

This SFM extends tradeable_regressor by adding Savitzky-Golay filter-based
inflection point detection for trading signals. Designed for parameter sweeps
via UEL to optimize both model and trading parameters.

Key additions:
- SG filter parameters (window, polyorder)
- Inflection detection thresholds
- Trading simulation on validation set
- Returns trading metrics alongside standard model metrics
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
    Parameter grid including both LightGBM and SG filter parameters.
    """
    base = base_params()

    # Add SG filter and trading parameters
    sg_params = {
        # Prediction horizon - how far ahead to predict (in minutes)
        # This affects model target and SG filter timing
        'lookahead_minutes': [30, 60, 90, 120],  # 30min to 2hr

        # SG Filter parameters
        'sg_window': [5, 7, 9, 11, 13],  # Must be odd
        'sg_polyorder': [2, 3, 4],

        # Trading signal filters
        'min_curvature_threshold': [0.00001, 0.00003, 0.00005, 0.0001, 0.0003],
        'min_prediction_level': [0.0005, 0.001, 0.0015, 0.002, 0.003],
        'min_inflection_spacing': [5, 10, 15, 20],

        # Keep these fixed for now
        'skip_boundary_bars': [10],
        'commission_rate': [0.002],
    }

    # Merge with base params
    return {**base, **sg_params}


def prep(data, round_params=None):
    """
    Prep with parametrized lookahead_minutes.

    Temporarily overrides BASE_CONFIG with round_params values before calling base prep.
    """
    if round_params is None:
        round_params = {}

    # Save original config values
    original_lookahead = BASE_CONFIG['lookahead_minutes']

    # Override with round_params if provided
    if 'lookahead_minutes' in round_params:
        BASE_CONFIG['lookahead_minutes'] = round_params['lookahead_minutes']

    try:
        # Call base prep with modified config
        result = base_prep(data, round_params)
        return result
    finally:
        # Restore original config
        BASE_CONFIG['lookahead_minutes'] = original_lookahead


def apply_sg_filter(predictions, window=11, polyorder=3, deriv=0):
    """
    Apply Savitzky-Golay filter to predictions.
    """
    if len(predictions) < window:
        return np.full_like(predictions, np.nan)

    return savgol_filter(predictions, window_length=window, polyorder=polyorder,
                        deriv=deriv, mode='nearest')


def detect_inflection_points(second_deriv, smoothed_pred, config):
    """
    Detect inflection points where 2nd derivative crosses zero.
    """
    inflections = []

    for i in range(1, len(second_deriv)):
        if i < config['skip_boundary_bars'] or i >= len(second_deriv) - config['skip_boundary_bars']:
            continue

        prev_deriv = second_deriv[i-1]
        curr_deriv = second_deriv[i]

        if np.isnan(prev_deriv) or np.isnan(curr_deriv):
            continue

        # Zero crossing detected
        if prev_deriv * curr_deriv < 0:
            # Check minimum curvature
            if abs(prev_deriv) < config['min_curvature_threshold']:
                continue

            # Check prediction level
            if smoothed_pred[i] < config['min_prediction_level']:
                continue

            # Classify inflection type
            if prev_deriv < 0 and curr_deriv > 0:
                inflection_type = 'bottom'
            elif prev_deriv > 0 and curr_deriv < 0:
                inflection_type = 'top'
            else:
                continue

            inflections.append({
                'bar': i,
                'type': inflection_type,
                'curvature': abs(prev_deriv),
                'prediction': smoothed_pred[i]
            })

    return pl.DataFrame(inflections) if inflections else pl.DataFrame()


def filter_inflection_spacing(inflections, min_spacing):
    """
    Filter out inflections that are too close together.
    """
    if len(inflections) == 0:
        return inflections

    filtered = []
    last_bar = -999

    for row in inflections.iter_rows(named=True):
        if row['bar'] - last_bar >= min_spacing:
            filtered.append(row)
            last_bar = row['bar']

    return pl.DataFrame(filtered) if filtered else pl.DataFrame()


def simulate_sg_trading(data, predictions, inflections, commission_rate):
    """
    Simulate trading based on inflection points (LONG only).
    """
    if len(inflections) == 0:
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

    for row in inflections.iter_rows(named=True):
        bar = row['bar']
        inflection_type = row['type']

        if inflection_type == 'bottom' and position is None:
            # LONG entry
            position = {
                'entry_bar': bar,
                'entry_price': prices[bar],
            }

        elif inflection_type == 'top' and position is not None:
            # Exit LONG
            exit_price = prices[bar]
            gross_return = (exit_price - position['entry_price']) / position['entry_price']
            net_return = gross_return - commission_rate

            trades.append({
                'net_return': net_return,
                'bars_held': bar - position['entry_bar'],
            })

            position = None

    # Close any open position at end
    if position is not None:
        bar = len(prices) - 1
        exit_price = prices[bar]
        gross_return = (exit_price - position['entry_price']) / position['entry_price']
        net_return = gross_return - commission_rate

        trades.append({
            'net_return': net_return,
            'bars_held': bar - position['entry_bar'],
        })

    if len(trades) == 0:
        return {
            'total_return': 0,
            'num_trades': 0,
            'win_rate': 0,
            'avg_trade': 0,
            'sharpe_ratio': 0,
        }

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
    Train model and evaluate SG filter trading strategy.

    Returns standard UEL format with trading metrics in extras.
    """
    if round_params is None:
        round_params = {}

    # Extract SG params from round_params
    sg_config = {
        'sg_window': round_params.get('sg_window', 11),
        'sg_polyorder': round_params.get('sg_polyorder', 3),
        'min_curvature_threshold': round_params.get('min_curvature_threshold', 0.00005),
        'min_prediction_level': round_params.get('min_prediction_level', 0.001),
        'min_inflection_spacing': round_params.get('min_inflection_spacing', 10),
        'skip_boundary_bars': round_params.get('skip_boundary_bars', 10),
        'commission_rate': round_params.get('commission_rate', 0.002),
    }

    # Train base model
    model_result = base_model(data, round_params)

    # Get validation predictions for trading simulation
    lgb_model = model_result['models'][0]
    numeric_features = model_result['extras']['numeric_features']
    val_clean = data['_val_clean']

    val_predictions = lgb_model.predict(val_clean.select(numeric_features).to_numpy())

    # Apply SG filter to validation predictions
    smoothed_pred = apply_sg_filter(val_predictions,
                                    window=sg_config['sg_window'],
                                    polyorder=sg_config['sg_polyorder'],
                                    deriv=0)
    second_deriv = apply_sg_filter(val_predictions,
                                   window=sg_config['sg_window'],
                                   polyorder=sg_config['sg_polyorder'],
                                   deriv=2)

    # Detect inflections on validation set
    inflections = detect_inflection_points(second_deriv, smoothed_pred, sg_config)
    inflections_filtered = filter_inflection_spacing(inflections, sg_config['min_inflection_spacing'])

    # Simulate trading on validation set
    trading_metrics = simulate_sg_trading(val_clean, val_predictions,
                                         inflections_filtered, sg_config['commission_rate'])

    # Also get test set metrics for final evaluation
    test_predictions = model_result['_preds']
    test_clean = model_result['extras']['test_clean']

    test_smoothed = apply_sg_filter(test_predictions,
                                    window=sg_config['sg_window'],
                                    polyorder=sg_config['sg_polyorder'],
                                    deriv=0)
    test_second_deriv = apply_sg_filter(test_predictions,
                                       window=sg_config['sg_window'],
                                       polyorder=sg_config['sg_polyorder'],
                                       deriv=2)

    test_inflections = detect_inflection_points(test_second_deriv, test_smoothed, sg_config)
    test_inflections_filtered = filter_inflection_spacing(test_inflections, sg_config['min_inflection_spacing'])

    test_trading_metrics = simulate_sg_trading(test_clean, test_predictions,
                                               test_inflections_filtered, sg_config['commission_rate'])

    # Compute trading-optimized metric for UEL selection
    # Use validation Sharpe ratio as primary metric (higher is better)
    # But UEL minimizes, so negate it
    val_trading_score = -float(trading_metrics['sharpe_ratio']) if trading_metrics['num_trades'] >= 5 else 999.0

    # Return UEL-compatible format
    return {
        'models': model_result['models'],
        'val_rmse': model_result['val_rmse'],
        'val_trading_score': val_trading_score,  # New metric for SG strategy
        'n_regimes_trained': 0,
        '_preds': test_predictions,
        'universal_val_rmse': model_result['universal_val_rmse'],
        'universal_samples': model_result['universal_samples'],
        'extras': {
            **model_result['extras'],
            'sg_config': sg_config,
            'val_trading_metrics': trading_metrics,
            'test_trading_metrics': test_trading_metrics,
            'val_inflections_detected': len(inflections_filtered),
            'test_inflections_detected': len(test_inflections_filtered),
        }
    }
