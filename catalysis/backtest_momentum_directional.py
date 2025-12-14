#!/usr/bin/env python3
"""
Backtest Momentum-Directional Strategy

Rethought approach based on validation results:
- Zones predict DIRECTION/MOMENTUM, not resistance
- STRONG zones (high reached rate) = bullish signal
- Enter long toward STRONG zones with trailing stops
- Exit on exhaustion (WEAK zone reached) or time-based
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


def identify_zones_by_strength(probabilities: dict, thresholds: list,
                                sample_idx: int) -> dict:
    """
    Identify zones and categorize by strength.

    Returns dict with 'strong', 'medium', 'weak' zone lists
    """
    zones = {'strong': [], 'medium': [], 'weak': []}
    probs = [probabilities[t][sample_idx] for t in thresholds]

    for i in range(len(thresholds) - 1):
        delta = probs[i] - probs[i + 1]

        if delta > 0.05:
            strength = 'strong'
        elif delta > 0.025:
            strength = 'medium'
        elif delta > 0.0125:
            strength = 'weak'
        else:
            continue

        zone = {
            'threshold_low': thresholds[i],
            'threshold_high': thresholds[i + 1],
            'prob_low': probs[i],
            'prob_high': probs[i + 1],
            'delta': delta,
            'distance_pct': thresholds[i]  # Distance from current price
        }

        zones[strength].append(zone)

    return zones


def generate_momentum_signal(current_price: float, zones: dict,
                              atr_stop_pct: float = 0.002) -> dict:
    """
    Generate momentum-directional signal.

    Strategy:
    - Enter LONG when STRONG zone within 0.5-2.5%
    - Use trailing stop (not fixed TP at zone)
    - Position size based on zone distance and strength
    """

    # Look for STRONG zones in favorable distance range
    strong_zones = zones.get('strong', [])

    if not strong_zones:
        return None

    # Filter zones by distance (0.5% - 2.5%)
    nearby_zones = [z for z in strong_zones
                    if 0.005 <= z['distance_pct'] <= 0.025]

    if not nearby_zones:
        return None

    # Use nearest STRONG zone as momentum target
    nearest = min(nearby_zones, key=lambda z: z['distance_pct'])

    # Position sizing based on distance
    # Closer zones = higher probability of quick move = larger position
    distance = nearest['distance_pct']
    if distance < 0.01:  # <1%
        position_size = 1.0
    elif distance < 0.015:  # 1-1.5%
        position_size = 0.8
    elif distance < 0.02:  # 1.5-2%
        position_size = 0.6
    else:  # 2-2.5%
        position_size = 0.4

    # Trailing stop distance (tighter for closer zones)
    if distance < 0.01:
        trailing_stop_pct = atr_stop_pct * 1.5  # 0.3%
    else:
        trailing_stop_pct = atr_stop_pct * 2.0  # 0.4%

    # Initial stop loss
    sl_price = current_price * (1 - atr_stop_pct)

    return {
        'entry': current_price,
        'initial_sl': sl_price,
        'trailing_stop_pct': trailing_stop_pct,
        'position_size': position_size,
        'target_zone': nearest,
        'signal_type': 'momentum_long'
    }


def check_exhaustion_exit(current_price: float, entry_price: float,
                          zones: dict) -> bool:
    """
    Check if price reached a WEAK zone (exhaustion signal).

    WEAK zones have only 18% reached rate - if we get there, it's exhaustion.
    """
    weak_zones = zones.get('weak', [])

    # Calculate current gain
    gain_pct = (current_price - entry_price) / entry_price

    # Check if we reached any WEAK zone
    for zone in weak_zones:
        zone_low = zone['threshold_low']
        zone_high = zone['threshold_high']

        # If current gain is within a WEAK zone
        if zone_low <= gain_pct <= zone_high:
            return True

    return False


def simulate_momentum_trade(signal: dict, future_prices: np.ndarray,
                            probabilities: dict, thresholds: list,
                            sample_idx: int, commission: float = 0.002) -> dict:
    """
    Simulate momentum-directional trade with trailing stop.

    Exit conditions:
    1. Trailing stop hit
    2. Exhaustion (WEAK zone reached)
    3. Time-based (lookahead bars exhausted)
    """
    if signal is None:
        return None

    entry = signal['entry']
    initial_sl = signal['initial_sl']
    trailing_stop_pct = signal['trailing_stop_pct']
    position_size = signal['position_size']

    # Track highest price for trailing stop
    highest_price = entry
    current_sl = initial_sl

    trade_result = {
        'entry': entry,
        'initial_sl': initial_sl,
        'trailing_stop_pct': trailing_stop_pct,
        'position_size': position_size,
        'pnl': 0,
        'pnl_pct': 0,
        'outcome': 'unknown',
        'exit_bar': len(future_prices) - 1,
        'exit_reason': 'timeout',
        'max_gain': 0
    }

    for i, price in enumerate(future_prices):
        # Update highest price
        if price > highest_price:
            highest_price = price
            # Update trailing stop
            current_sl = highest_price * (1 - trailing_stop_pct)

        # Track max gain
        gain = (price - entry) / entry
        if gain > trade_result['max_gain']:
            trade_result['max_gain'] = gain

        # Check trailing stop
        if price <= current_sl:
            gross_return = (current_sl - entry) / entry
            net_return = gross_return - commission
            pnl = net_return * position_size

            trade_result['pnl'] = pnl
            trade_result['pnl_pct'] = net_return
            trade_result['outcome'] = 'win' if pnl > 0 else 'loss'
            trade_result['exit_bar'] = i
            trade_result['exit_reason'] = 'trailing_stop'
            break

        # Check exhaustion (WEAK zone reached)
        # Get zones for current position
        zones = identify_zones_by_strength(probabilities, thresholds, sample_idx + i)
        if check_exhaustion_exit(price, entry, zones):
            gross_return = (price - entry) / entry
            net_return = gross_return - commission
            pnl = net_return * position_size

            trade_result['pnl'] = pnl
            trade_result['pnl_pct'] = net_return
            trade_result['outcome'] = 'win' if pnl > 0 else 'loss'
            trade_result['exit_bar'] = i
            trade_result['exit_reason'] = 'exhaustion'
            break

    # If still open at end (timeout)
    if trade_result['exit_reason'] == 'timeout':
        final_price = future_prices[-1]
        gross_return = (final_price - entry) / entry
        net_return = gross_return - commission
        pnl = net_return * position_size

        trade_result['pnl'] = pnl
        trade_result['pnl_pct'] = net_return
        trade_result['outcome'] = 'win' if pnl > 0 else 'loss'

    return trade_result


def backtest_momentum_strategy(test_data: pl.DataFrame, test_probabilities: dict,
                               thresholds: list, lookahead_bars: int = 96) -> dict:
    """
    Backtest momentum-directional strategy.
    """
    print('\nRunning momentum-directional backtest...')

    n_samples = len(test_probabilities[thresholds[0]])
    test_closes = test_data['close'].to_numpy()

    momentum_trades = []
    atr_stop_pct = 0.002

    # Calculate ATR if available
    if 'atr' in test_data.columns:
        test_atr = test_data['atr'].to_numpy()
        avg_atr = np.nanmean(test_atr)
        if avg_atr > 0:
            atr_stop_pct = avg_atr / test_data['close'].mean()

    for i in range(n_samples - lookahead_bars):
        if i % 5000 == 0:
            print(f'  Processing sample {i}/{n_samples - lookahead_bars}...')

        current_price = test_closes[i]
        future_prices = test_closes[i:i+lookahead_bars]

        # Identify zones
        zones = identify_zones_by_strength(test_probabilities, thresholds, i)

        # Generate momentum signal
        signal = generate_momentum_signal(current_price, zones, atr_stop_pct)

        if signal:
            trade = simulate_momentum_trade(
                signal, future_prices, test_probabilities, thresholds, i
            )
            if trade:
                momentum_trades.append(trade)

    # Calculate metrics
    def calculate_metrics(trades):
        if not trades:
            return {
                'num_trades': 0,
                'total_return': 0,
                'avg_return': 0,
                'win_rate': 0,
                'sharpe': 0,
                'max_drawdown': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'avg_bars_held': 0
            }

        returns = np.array([t['pnl'] for t in trades])
        wins = [t for t in trades if t['outcome'] == 'win']
        losses = [t for t in trades if t['outcome'] == 'loss']

        # Calculate drawdown
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        max_dd = np.max(drawdown) if len(drawdown) > 0 else 0

        # Win/loss stats
        avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
        avg_loss = np.mean([t['pnl'] for t in losses]) if losses else 0

        # Profit factor
        gross_profit = sum([t['pnl'] for t in wins]) if wins else 0
        gross_loss = abs(sum([t['pnl'] for t in losses])) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Avg bars held
        avg_bars = np.mean([t['exit_bar'] for t in trades])

        return {
            'num_trades': len(trades),
            'total_return': np.sum(returns),
            'avg_return': np.mean(returns),
            'win_rate': len(wins) / len(trades) if trades else 0,
            'sharpe': (np.mean(returns) / np.std(returns)) if np.std(returns) > 0 else 0,
            'max_drawdown': max_dd,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_bars_held': avg_bars
        }

    momentum_metrics = calculate_metrics(momentum_trades)

    return {
        'momentum': momentum_metrics,
        'momentum_trades': momentum_trades
    }


def print_backtest_results(results: dict):
    """Pretty print backtest results"""

    print('\n' + '=' * 80)
    print('MOMENTUM-DIRECTIONAL STRATEGY BACKTEST')
    print('=' * 80)

    mom_metrics = results['momentum']

    print('\nMOMENTUM-DIRECTIONAL STRATEGY:')
    print('-' * 80)
    print(f"  Trades:          {mom_metrics['num_trades']}")
    print(f"  Total Return:    {mom_metrics['total_return']*100:.2f}%")
    print(f"  Avg Return:      {mom_metrics['avg_return']*100:.3f}%")
    print(f"  Win Rate:        {mom_metrics['win_rate']*100:.1f}%")
    print(f"  Sharpe Ratio:    {mom_metrics['sharpe']:.3f}")
    print(f"  Max Drawdown:    {mom_metrics['max_drawdown']*100:.2f}%")
    print(f"  Avg Win:         {mom_metrics['avg_win']*100:.3f}%")
    print(f"  Avg Loss:        {mom_metrics['avg_loss']*100:.3f}%")
    print(f"  Profit Factor:   {mom_metrics['profit_factor']:.2f}")
    print(f"  Avg Bars Held:   {mom_metrics['avg_bars_held']:.1f}")

    print('\n' + '=' * 80)

    # Exit reason breakdown
    trades = results['momentum_trades']
    if trades:
        print('\nEXIT REASON BREAKDOWN:')
        print('-' * 80)

        exit_reasons = {}
        for trade in trades:
            reason = trade['exit_reason']
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

        for reason, count in exit_reasons.items():
            pct = (count / len(trades)) * 100
            print(f"  {reason:20s}: {count:5d} ({pct:5.1f}%)")

        print('\n' + '=' * 80)


def main():
    print('=' * 80)
    print('MOMENTUM-DIRECTIONAL STRATEGY BACKTEST')
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
        experiment_name='momentum_directional_backtest',
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
    results = backtest_momentum_strategy(
        test_data,
        test_probabilities,
        mtrf.THRESHOLDS,
        lookahead_bars=96  # 96 * 5min = 8 hours
    )

    # Print results
    print_backtest_results(results)

    # Save results
    print('\nSaving backtest results...')

    if results['momentum']['num_trades'] > 0:
        trades_df = pl.DataFrame(results['momentum_trades'])
        output_file = '/Users/beyondsyntax/Loop/catalysis/momentum_directional_backtest_results.csv'
        trades_df.write_csv(output_file)
        print(f'Trade details saved to: {output_file}')

    print('\nDone!')


if __name__ == '__main__':
    main()
