#!/usr/bin/env python3
"""
SG Filter Trading Strategy V1
Uses Savitzky-Golay filter on predictions to detect inflection points for trading signals.
"""

import numpy as np
import polars as pl
from scipy.signal import savgol_filter
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import loop


# SG Filter Strategy Configuration
SG_CONFIG = {
    'prediction_window': 60,  # Number of past predictions to use for SG filter
    'sg_window': 11,  # SG filter window (must be odd)
    'sg_polyorder': 3,  # Polynomial order for SG filter
    'min_curvature_threshold': 0.00001,  # Minimum |2nd derivative| before crossing
    'min_prediction_level': 0.0008,  # Only trade if smoothed prediction > 0.08%
    'min_inflection_spacing': 10,  # Minimum bars between inflections (avoid whipsaws)
    'skip_boundary_bars': 10,  # Don't trade on first/last N bars (boundary effects)
    'commission_rate': 0.002,  # 0.2% per trade
}


def apply_sg_filter(predictions, window=11, polyorder=3, deriv=0):
    """
    Apply Savitzky-Golay filter to predictions.

    Args:
        predictions: Array of predictions
        window: Window length (must be odd)
        polyorder: Polynomial order
        deriv: Derivative order (0=smoothed, 1=1st deriv, 2=2nd deriv)

    Returns:
        Filtered predictions
    """
    if len(predictions) < window:
        return np.full_like(predictions, np.nan)

    # Apply SG filter with mode='nearest' for boundaries
    return savgol_filter(predictions, window_length=window, polyorder=polyorder,
                        deriv=deriv, mode='nearest')


def detect_inflection_points(second_deriv, smoothed_pred, config):
    """
    Detect inflection points where 2nd derivative crosses zero.

    Args:
        second_deriv: 2nd derivative of smoothed predictions
        smoothed_pred: Smoothed predictions
        config: Strategy configuration

    Returns:
        DataFrame with inflection points and their types
    """
    inflections = []

    for i in range(1, len(second_deriv)):
        # Skip boundary bars
        if i < config['skip_boundary_bars'] or i >= len(second_deriv) - config['skip_boundary_bars']:
            continue

        # Check for zero crossing
        prev_deriv = second_deriv[i-1]
        curr_deriv = second_deriv[i]

        if np.isnan(prev_deriv) or np.isnan(curr_deriv):
            continue

        # Zero crossing detected
        if prev_deriv * curr_deriv < 0:
            # Check minimum curvature requirement (before crossing)
            if abs(prev_deriv) < config['min_curvature_threshold']:
                continue

            # Check prediction level requirement
            if smoothed_pred[i] < config['min_prediction_level']:
                continue

            # Classify inflection type
            if prev_deriv < 0 and curr_deriv > 0:
                inflection_type = 'bottom'  # Potential buy
            elif prev_deriv > 0 and curr_deriv < 0:
                inflection_type = 'top'  # Potential sell
            else:
                continue

            inflections.append({
                'bar': i,
                'type': inflection_type,
                'curvature': abs(prev_deriv),
                'prediction': smoothed_pred[i],
                'second_deriv_before': prev_deriv,
                'second_deriv_after': curr_deriv
            })

    return pl.DataFrame(inflections) if inflections else pl.DataFrame()


def filter_inflection_spacing(inflections, config):
    """
    Filter out inflections that are too close together to avoid whipsaws.

    Args:
        inflections: DataFrame of inflection points
        config: Strategy configuration

    Returns:
        Filtered DataFrame of inflection points
    """
    if len(inflections) == 0:
        return inflections

    filtered = []
    last_bar = -999

    for row in inflections.iter_rows(named=True):
        if row['bar'] - last_bar >= config['min_inflection_spacing']:
            filtered.append(row)
            last_bar = row['bar']

    return pl.DataFrame(filtered) if filtered else pl.DataFrame()


def simulate_sg_trading(test_data, predictions, inflections, config):
    """
    Simulate trading based on inflection points.
    LONG only - no shorts.

    Args:
        test_data: Test dataset with OHLC data
        predictions: Array of predictions
        inflections: DataFrame of inflection points
        config: Strategy configuration

    Returns:
        Dictionary with trade results and metrics
    """
    if len(inflections) == 0:
        return {
            'trades': [],
            'total_return': 0,
            'num_trades': 0,
            'win_rate': 0,
            'avg_trade': 0,
        }

    trades = []
    position = None

    prices = test_data['close'].to_numpy()
    datetimes = test_data['datetime'].to_list()

    for row in inflections.iter_rows(named=True):
        bar = row['bar']
        inflection_type = row['type']

        if inflection_type == 'bottom' and position is None:
            # LONG entry
            entry_price = prices[bar]
            position = {
                'entry_bar': bar,
                'entry_price': entry_price,
                'entry_datetime': datetimes[bar],
                'entry_prediction': row['prediction'],
                'entry_curvature': row['curvature']
            }

        elif inflection_type == 'top' and position is not None:
            # Exit LONG
            exit_price = prices[bar]
            gross_return = (exit_price - position['entry_price']) / position['entry_price']
            net_return = gross_return - config['commission_rate']

            trades.append({
                'entry_bar': position['entry_bar'],
                'exit_bar': bar,
                'entry_datetime': position['entry_datetime'],
                'exit_datetime': datetimes[bar],
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'gross_return': gross_return,
                'net_return': net_return,
                'bars_held': bar - position['entry_bar'],
                'entry_prediction': position['entry_prediction'],
                'entry_curvature': position['entry_curvature']
            })

            position = None

    # Close any open position at end
    if position is not None:
        bar = len(prices) - 1
        exit_price = prices[bar]
        gross_return = (exit_price - position['entry_price']) / position['entry_price']
        net_return = gross_return - config['commission_rate']

        trades.append({
            'entry_bar': position['entry_bar'],
            'exit_bar': bar,
            'entry_datetime': position['entry_datetime'],
            'exit_datetime': datetimes[bar],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'gross_return': gross_return,
            'net_return': net_return,
            'bars_held': bar - position['entry_bar'],
            'entry_prediction': position['entry_prediction'],
            'entry_curvature': position['entry_curvature']
        })

    # Calculate metrics
    if len(trades) == 0:
        return {
            'trades': [],
            'total_return': 0,
            'num_trades': 0,
            'win_rate': 0,
            'avg_trade': 0,
        }

    trades_df = pl.DataFrame(trades)
    wins = trades_df.filter(pl.col('net_return') > 0)

    return {
        'trades': trades_df,
        'total_return': trades_df['net_return'].sum(),
        'num_trades': len(trades_df),
        'win_rate': len(wins) / len(trades_df) if len(trades_df) > 0 else 0,
        'avg_trade': trades_df['net_return'].mean(),
        'avg_bars_held': trades_df['bars_held'].mean(),
        'best_trade': trades_df['net_return'].max(),
        'worst_trade': trades_df['net_return'].min(),
    }


def main():
    print("=" * 80)
    print("SG Filter Trading Strategy V1 - Testing on Tradeable Regressor")
    print("=" * 80)

    # Load data
    print("\n[1/7] Loading data...")
    kline_size = 300
    end_date = datetime.now()
    # To get 3-month test period with 15% test split, need ~600 days total
    start_date = end_date - timedelta(days=600)
    start_date_str = start_date.strftime('%Y-%m-%d')

    print(f"Fetching data: {start_date_str} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Target: 3-month test period (~90 days)")

    historical = loop.HistoricalData()
    historical.get_spot_klines(kline_size=kline_size, start_date_limit=start_date_str)

    data = historical.data
    print(f"Loaded {len(data)} candles")

    # Prepare data with existing SFM
    print("\n[2/7] Preparing features with tradeable_regressor...")
    prep_data = loop.sfm.lightgbm.tradeable_regressor.prep(data)
    print(f"Train: {len(prep_data['x_train'])}, Val: {len(prep_data['x_val'])}, Test: {len(prep_data['x_test'])}")

    # Train model
    print("\n[3/7] Training LightGBM model...")
    default_params = loop.sfm.lightgbm.tradeable_regressor.params()
    round_params = {k: v[0] if isinstance(v, list) else v for k, v in default_params.items()}
    model_result = loop.sfm.lightgbm.tradeable_regressor.model(prep_data, round_params)

    print(f"Model trained - Val RMSE: {model_result['val_rmse']:.6f}")

    # Get predictions on test set
    predictions = model_result['_preds']
    test_data = model_result['extras']['test_clean']

    print(f"\nTest set: {len(test_data)} samples")
    print(f"Predictions: min={predictions.min():.6f}, max={predictions.max():.6f}, mean={predictions.mean():.6f}")

    # Apply SG filter
    print(f"\n[4/7] Applying SG filter (window={SG_CONFIG['sg_window']}, polyorder={SG_CONFIG['sg_polyorder']})...")
    smoothed_pred = apply_sg_filter(predictions, SG_CONFIG['sg_window'], SG_CONFIG['sg_polyorder'], deriv=0)
    second_deriv = apply_sg_filter(predictions, SG_CONFIG['sg_window'], SG_CONFIG['sg_polyorder'], deriv=2)

    print(f"Smoothed predictions: min={np.nanmin(smoothed_pred):.6f}, max={np.nanmax(smoothed_pred):.6f}")
    print(f"2nd derivative: min={np.nanmin(second_deriv):.8f}, max={np.nanmax(second_deriv):.8f}")

    # Detect inflection points
    print(f"\n[5/7] Detecting inflection points...")
    inflections = detect_inflection_points(second_deriv, smoothed_pred, SG_CONFIG)
    print(f"Found {len(inflections)} inflections before spacing filter")

    if len(inflections) > 0:
        bottoms = inflections.filter(pl.col('type') == 'bottom')
        tops = inflections.filter(pl.col('type') == 'top')
        print(f"  - Bottoms: {len(bottoms)}")
        print(f"  - Tops: {len(tops)}")

    # Filter by spacing
    print(f"\n[6/7] Filtering inflections (min spacing: {SG_CONFIG['min_inflection_spacing']} bars)...")
    inflections_filtered = filter_inflection_spacing(inflections, SG_CONFIG)
    print(f"Inflections after spacing filter: {len(inflections_filtered)}")

    if len(inflections_filtered) > 0:
        bottoms = inflections_filtered.filter(pl.col('type') == 'bottom')
        tops = inflections_filtered.filter(pl.col('type') == 'top')
        print(f"  - Bottoms: {len(bottoms)}")
        print(f"  - Tops: {len(tops)}")

    # Simulate trading
    print(f"\n[7/7] Simulating trading (LONG only)...")
    results = simulate_sg_trading(test_data, predictions, inflections_filtered, SG_CONFIG)

    # Print results
    print("\n" + "=" * 80)
    print("TRADING RESULTS")
    print("=" * 80)
    print(f"Total Return: {results['total_return']*100:.2f}%")
    print(f"Number of Trades: {results['num_trades']}")
    print(f"Win Rate: {results['win_rate']*100:.1f}%")
    print(f"Average Trade: {results['avg_trade']*100:.2f}%")

    if results['num_trades'] > 0:
        print(f"Average Bars Held: {results['avg_bars_held']:.1f} ({results['avg_bars_held']*5:.0f} minutes)")
        print(f"Best Trade: {results['best_trade']*100:.2f}%")
        print(f"Worst Trade: {results['worst_trade']*100:.2f}%")

        # Show some trades
        trades_df = results['trades']
        print(f"\nFirst 5 trades:")
        print(trades_df.head(5).select(['entry_datetime', 'exit_datetime', 'net_return', 'bars_held']))

        # Save trades to CSV
        output_file = '/Users/beyondsyntax/Loop/catalysis/sg_filter_trades_v1.csv'
        trades_df.write_csv(output_file)
        print(f"\nAll trades saved to: {output_file}")

    # Buy and hold comparison
    first_price = test_data['close'][0]
    last_price = test_data['close'][-1]
    buy_hold_return = (last_price - first_price) / first_price - SG_CONFIG['commission_rate']

    print(f"\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print(f"Buy & Hold Return: {buy_hold_return*100:.2f}%")
    print(f"SG Strategy Return: {results['total_return']*100:.2f}%")
    print(f"Outperformance: {(results['total_return'] - buy_hold_return)*100:.2f}%")

    print("\n" + "=" * 80)
    print("CONFIGURATION USED")
    print("=" * 80)
    for key, value in SG_CONFIG.items():
        print(f"{key}: {value}")

    return results


if __name__ == '__main__':
    results = main()
