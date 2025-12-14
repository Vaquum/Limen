#!/usr/bin/env python3
"""
Backtest Resistance-Based Trading Strategy

Uses multi-threshold resistance finder to backtest a trading strategy:
1. Train resistance models
2. Generate signals for each test timestamp
3. Simulate trades with resistance-based TP/SL
4. Compare vs baseline (fixed TP/SL)
"""

import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, '/Users/beyondsyntax/Loop/catalysis')

import loop
import numpy as np
import polars as pl
from datetime import datetime, timedelta
import multi_threshold_resistance_finder as mtrf


def identify_resistance_zones_for_sample(probabilities: dict, thresholds: list,
                                        sample_idx: int, strength_threshold: float = 0.05) -> list:
    """
    Identify resistance zones for a specific sample index.

    Returns list of resistance zones for this timestamp.
    """
    zones = []
    probs = [probabilities[t][sample_idx] for t in thresholds]

    for i in range(len(thresholds) - 1):
        delta = probs[i] - probs[i + 1]

        if delta > strength_threshold:
            strength = 'STRONG'
        elif delta > strength_threshold / 2:
            strength = 'MEDIUM'
        elif delta > strength_threshold / 4:
            strength = 'WEAK'
        else:
            continue

        zones.append({
            'threshold_low': thresholds[i],
            'threshold_high': thresholds[i + 1],
            'prob_low': probs[i],
            'prob_high': probs[i + 1],
            'delta': delta,
            'strength': strength
        })

    return zones


def generate_resistance_signal(current_price: float, resistance_zones: list,
                               atr_stop_pct: float = 0.002) -> dict:
    """
    Generate LONG signal based on resistance zones.

    Returns signal dict with entry, tp1, tp2, sl
    """
    if not resistance_zones:
        return None

    # Find nearest strong resistance
    strong_zones = [z for z in resistance_zones if z['strength'] == 'STRONG']

    if not strong_zones:
        return None

    nearest = strong_zones[0]

    # TP1: 90% of the way to resistance_low
    tp1_pct = nearest['threshold_low'] * 0.9
    tp1_price = current_price * (1 + tp1_pct)

    # TP2: Midpoint of resistance zone
    tp2_pct = (nearest['threshold_low'] + nearest['threshold_high']) / 2
    tp2_price = current_price * (1 + tp2_pct)

    # SL: Based on ATR
    sl_price = current_price * (1 - atr_stop_pct)

    # Position sizing
    if nearest['threshold_low'] < 0.005:
        position_size = 0.5
    elif nearest['threshold_low'] < 0.01:
        position_size = 0.75
    else:
        position_size = 1.0

    return {
        'entry': current_price,
        'tp1': tp1_price,
        'tp1_pct': tp1_pct,
        'tp2': tp2_price,
        'tp2_pct': tp2_pct,
        'sl': sl_price,
        'sl_pct': atr_stop_pct,
        'position_size': position_size,
        'resistance_pct': nearest['threshold_low']
    }


def generate_baseline_signal(current_price: float, atr_stop_pct: float = 0.002,
                             fixed_tp_pct: float = 0.005) -> dict:
    """
    Generate baseline signal with fixed TP/SL (no resistance).

    Fixed TP at 0.5% (average of our resistance zones)
    """
    return {
        'entry': current_price,
        'tp1': current_price * (1 + fixed_tp_pct),
        'tp1_pct': fixed_tp_pct,
        'sl': current_price * (1 - atr_stop_pct),
        'sl_pct': atr_stop_pct,
        'position_size': 1.0
    }


def simulate_trade(signal: dict, future_prices: np.ndarray, commission: float = 0.002) -> dict:
    """
    Simulate a single trade given a signal and future price action.

    Returns trade result with pnl, outcome, exit_reason
    """
    if signal is None:
        return None

    entry = signal['entry']
    tp1 = signal['tp1']
    sl = signal['sl']
    position_size = signal['position_size']

    # For resistance strategy with TP2
    tp2 = signal.get('tp2', tp1)  # Use tp1 if no tp2 (baseline)

    trade_result = {
        'entry': entry,
        'tp1': tp1,
        'tp2': tp2,
        'sl': sl,
        'position_size': position_size,
        'pnl': 0,
        'pnl_pct': 0,
        'outcome': 'unknown',
        'exit_bar': len(future_prices) - 1,
        'exit_reason': 'timeout'
    }

    # Simulate price action
    position_remaining = 1.0  # Track how much position is still open (0.0 - 1.0)

    for i, price in enumerate(future_prices):
        # Check stop-loss
        if price <= sl:
            gross_return = (sl - entry) / entry
            net_return = gross_return - commission
            pnl = net_return * position_remaining

            trade_result['pnl'] = pnl
            trade_result['pnl_pct'] = net_return
            trade_result['outcome'] = 'loss'
            trade_result['exit_bar'] = i
            trade_result['exit_reason'] = 'sl_hit'
            break

        # Check TP1 (exit 70% of position)
        if price >= tp1 and position_remaining > 0.7:
            gross_return = (tp1 - entry) / entry
            net_return = gross_return - commission
            pnl_partial = net_return * 0.7

            trade_result['pnl'] += pnl_partial
            position_remaining -= 0.7

            # Continue with remaining 30% toward TP2

        # Check TP2 (exit remaining position)
        if price >= tp2 and position_remaining > 0:
            gross_return = (tp2 - entry) / entry
            net_return = gross_return - commission
            pnl_partial = net_return * position_remaining

            trade_result['pnl'] += pnl_partial
            trade_result['pnl_pct'] = trade_result['pnl']  # Weighted average
            trade_result['outcome'] = 'win'
            trade_result['exit_bar'] = i
            trade_result['exit_reason'] = 'tp2_hit'
            position_remaining = 0
            break

    # If position still open at end (timeout)
    if position_remaining > 0:
        final_price = future_prices[-1]
        gross_return = (final_price - entry) / entry
        net_return = gross_return - commission
        pnl_partial = net_return * position_remaining

        trade_result['pnl'] += pnl_partial
        trade_result['pnl_pct'] = trade_result['pnl']
        trade_result['outcome'] = 'win' if trade_result['pnl'] > 0 else 'loss'

    return trade_result


def backtest_strategy(test_data: pl.DataFrame, test_probabilities: dict,
                     thresholds: list, lookahead_bars: int = 96) -> dict:
    """
    Backtest resistance-based strategy vs baseline on test data.

    Returns dict with both strategies' performance metrics.
    """
    print('\nRunning backtest...')

    n_samples = len(test_probabilities[thresholds[0]])

    resistance_trades = []
    baseline_trades = []

    atr_stop_pct = 0.002  # 0.2% default

    # Calculate ATR if available
    if 'atr' in test_data.columns:
        test_atr = test_data['atr'].to_numpy()
        avg_atr = np.nanmean(test_atr)
        if avg_atr > 0:
            atr_stop_pct = avg_atr / test_data['close'].mean()

    test_closes = test_data['close'].to_numpy()

    for i in range(n_samples - lookahead_bars):
        current_price = test_closes[i]

        # Get future prices for this trade
        future_prices = test_closes[i:i+lookahead_bars]

        # Generate resistance-based signal
        resistance_zones = identify_resistance_zones_for_sample(
            test_probabilities, thresholds, i, strength_threshold=0.05
        )

        resistance_signal = generate_resistance_signal(
            current_price, resistance_zones, atr_stop_pct
        )

        # Generate baseline signal (fixed TP/SL)
        baseline_signal = generate_baseline_signal(
            current_price, atr_stop_pct, fixed_tp_pct=0.005
        )

        # Simulate both trades
        if resistance_signal:
            resistance_trade = simulate_trade(resistance_signal, future_prices)
            if resistance_trade:
                resistance_trades.append(resistance_trade)

        baseline_trade = simulate_trade(baseline_signal, future_prices)
        if baseline_trade:
            baseline_trades.append(baseline_trade)

    # Calculate metrics for both strategies
    def calculate_metrics(trades):
        if not trades:
            return {
                'num_trades': 0,
                'total_return': 0,
                'avg_return': 0,
                'win_rate': 0,
                'sharpe': 0,
                'max_drawdown': 0
            }

        returns = np.array([t['pnl'] for t in trades])
        wins = sum(1 for t in trades if t['outcome'] == 'win')

        # Calculate drawdown
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        max_dd = np.max(drawdown) if len(drawdown) > 0 else 0

        return {
            'num_trades': len(trades),
            'total_return': np.sum(returns),
            'avg_return': np.mean(returns),
            'win_rate': wins / len(trades) if len(trades) > 0 else 0,
            'sharpe': (np.mean(returns) / np.std(returns)) if np.std(returns) > 0 else 0,
            'max_drawdown': max_dd
        }

    resistance_metrics = calculate_metrics(resistance_trades)
    baseline_metrics = calculate_metrics(baseline_trades)

    return {
        'resistance': resistance_metrics,
        'baseline': baseline_metrics,
        'resistance_trades': resistance_trades,
        'baseline_trades': baseline_trades
    }


def print_backtest_results(results: dict):
    """Pretty print backtest results"""

    print('\n' + '=' * 80)
    print('BACKTEST RESULTS')
    print('=' * 80)

    res_metrics = results['resistance']
    base_metrics = results['baseline']

    print('\nRESISTANCE-BASED STRATEGY:')
    print('-' * 80)
    print(f"  Trades:          {res_metrics['num_trades']}")
    print(f"  Total Return:    {res_metrics['total_return']*100:.2f}%")
    print(f"  Avg Return:      {res_metrics['avg_return']*100:.3f}%")
    print(f"  Win Rate:        {res_metrics['win_rate']*100:.1f}%")
    print(f"  Sharpe Ratio:    {res_metrics['sharpe']:.3f}")
    print(f"  Max Drawdown:    {res_metrics['max_drawdown']*100:.2f}%")

    print('\nBASELINE STRATEGY (Fixed TP/SL):')
    print('-' * 80)
    print(f"  Trades:          {base_metrics['num_trades']}")
    print(f"  Total Return:    {base_metrics['total_return']*100:.2f}%")
    print(f"  Avg Return:      {base_metrics['avg_return']*100:.3f}%")
    print(f"  Win Rate:        {base_metrics['win_rate']*100:.1f}%")
    print(f"  Sharpe Ratio:    {base_metrics['sharpe']:.3f}")
    print(f"  Max Drawdown:    {base_metrics['max_drawdown']*100:.2f}%")

    print('\nCOMPARISON:')
    print('-' * 80)

    if base_metrics['total_return'] != 0:
        improvement = ((res_metrics['total_return'] - base_metrics['total_return']) /
                      abs(base_metrics['total_return'])) * 100
    else:
        improvement = 0

    print(f"  Return Improvement:  {improvement:+.1f}%")
    print(f"  Sharpe Improvement:  {res_metrics['sharpe'] - base_metrics['sharpe']:+.3f}")
    print(f"  Win Rate Improvement: {(res_metrics['win_rate'] - base_metrics['win_rate'])*100:+.1f}pp")

    winner = 'RESISTANCE' if res_metrics['total_return'] > base_metrics['total_return'] else 'BASELINE'
    print(f"\n  Winner: {winner}")

    print('\n' + '=' * 80)


def main():
    print('=' * 80)
    print('RESISTANCE STRATEGY BACKTEST')
    print('=' * 80)

    # Load data
    print('\nLoading 20 months of data...')
    kline_size = 300
    end_date = datetime.now()
    start_date = end_date - timedelta(days=20 * 30)
    start_date_str = start_date.strftime('%Y-%m-%d')

    historical = loop.HistoricalData()
    historical.get_spot_klines(kline_size=kline_size, start_date_limit=start_date_str)
    full_data = historical.data
    print(f'Loaded {len(full_data):,} candles')

    # Train resistance finder
    print('\nTraining multi-threshold resistance models...')

    uel = loop.UniversalExperimentLoop(
        data=full_data,
        single_file_model=mtrf
    )

    uel.run(
        experiment_name='resistance_backtest',
        n_permutations=1,
        random_search=False
    )

    # Get test data and probabilities
    print('\nExtracting test data and probabilities...')

    extras = uel.extras[0]
    test_probabilities = extras['test_probabilities']

    # Get test data from alignment
    alignment = uel._alignment[0]
    first_test_dt = alignment['first_test_datetime']
    last_test_dt = alignment['last_test_datetime']

    test_data = full_data.filter(
        (pl.col('datetime') >= first_test_dt) &
        (pl.col('datetime') <= last_test_dt)
    )

    print(f'Test period: {first_test_dt} to {last_test_dt}')
    print(f'Test samples: {len(test_data)}')

    # Run backtest
    results = backtest_strategy(
        test_data,
        test_probabilities,
        mtrf.THRESHOLDS,
        lookahead_bars=96  # 96 * 5min = 8 hours lookahead
    )

    # Print results
    print_backtest_results(results)

    # Save results
    print('\nSaving backtest results...')

    # Save to CSV
    res_df = pl.DataFrame({
        'strategy': ['Resistance', 'Baseline'],
        'num_trades': [results['resistance']['num_trades'], results['baseline']['num_trades']],
        'total_return': [results['resistance']['total_return'], results['baseline']['total_return']],
        'avg_return': [results['resistance']['avg_return'], results['baseline']['avg_return']],
        'win_rate': [results['resistance']['win_rate'], results['baseline']['win_rate']],
        'sharpe': [results['resistance']['sharpe'], results['baseline']['sharpe']],
        'max_drawdown': [results['resistance']['max_drawdown'], results['baseline']['max_drawdown']]
    })

    output_file = '/Users/beyondsyntax/Loop/catalysis/resistance_backtest_results.csv'
    res_df.write_csv(output_file)
    print(f'Results saved to: {output_file}')

    print('\nDone!')


if __name__ == '__main__':
    main()
