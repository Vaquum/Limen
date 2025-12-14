#!/usr/bin/env python3
"""
Real-Time Resistance-Based Trading Signal Generator

Uses multi-threshold resistance finder to generate trading signals:
- Identifies ML-predicted resistance zones
- Suggests entry, TP, and SL levels
- Calculates position sizing based on resistance proximity
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


def calculate_atr_stop(data: pl.DataFrame, atr_multiplier: float = 1.0) -> float:
    """Calculate stop-loss distance based on ATR"""
    if 'atr' in data.columns:
        recent_atr = data['atr'].tail(20).mean()
        if recent_atr is not None and not np.isnan(recent_atr):
            return float(recent_atr * atr_multiplier)
    # Fallback to 0.2% if ATR not available
    return 0.002


def identify_resistance_zones(probabilities: dict, thresholds: list, strength_threshold: float = 0.05) -> list:
    """
    Identify resistance zones from probability curves.

    Returns list of dicts with:
    - threshold_low, threshold_high: Zone boundaries
    - delta: Probability drop
    - strength: 'STRONG', 'MEDIUM', 'WEAK'
    """
    zones = []

    # Get the most recent sample (last in test set)
    latest_idx = len(probabilities[thresholds[0]]) - 1

    probs = [probabilities[t][latest_idx] for t in thresholds]

    for i in range(len(thresholds) - 1):
        delta = probs[i] - probs[i + 1]

        if delta > strength_threshold:
            strength = 'STRONG'
        elif delta > strength_threshold / 2:
            strength = 'MEDIUM'
        elif delta > strength_threshold / 4:
            strength = 'WEAK'
        else:
            continue  # Too weak to matter

        zones.append({
            'threshold_low': thresholds[i],
            'threshold_high': thresholds[i + 1],
            'prob_low': probs[i],
            'prob_high': probs[i + 1],
            'delta': delta,
            'strength': strength
        })

    return zones


def generate_trading_signals(current_price: float, resistance_zones: list, atr_stop_pct: float,
                             commission: float = 0.002) -> dict:
    """
    Generate trading signals based on resistance zones.

    Returns dict with LONG and SHORT signals
    """

    signals = {
        'LONG': None,
        'SHORT': None
    }

    # LONG SIGNAL
    if resistance_zones:
        # Find nearest strong resistance
        strong_zones = [z for z in resistance_zones if z['strength'] == 'STRONG']

        if strong_zones:
            nearest = strong_zones[0]

            # TP1: Just before resistance (90% of the way to resistance_low)
            tp1_pct = nearest['threshold_low'] * 0.9
            tp1_price = current_price * (1 + tp1_pct)

            # TP2: Within resistance zone (midpoint)
            tp2_pct = (nearest['threshold_low'] + nearest['threshold_high']) / 2
            tp2_price = current_price * (1 + tp2_pct)

            # SL: Based on ATR
            sl_price = current_price * (1 - atr_stop_pct)

            # Position sizing: Reduce if resistance is very close
            if nearest['threshold_low'] < 0.005:  # Less than 0.5% away
                position_size = 0.5  # 50% normal size
            elif nearest['threshold_low'] < 0.01:  # Less than 1% away
                position_size = 0.75  # 75% normal size
            else:
                position_size = 1.0  # 100% normal size

            # Risk:Reward for TP1
            risk = current_price - sl_price
            reward_tp1 = tp1_price - current_price
            rr_tp1 = reward_tp1 / risk if risk > 0 else 0

            signals['LONG'] = {
                'entry': current_price,
                'tp1': tp1_price,
                'tp1_pct': tp1_pct,
                'tp2': tp2_price,
                'tp2_pct': tp2_pct,
                'sl': sl_price,
                'sl_pct': atr_stop_pct,
                'position_size': position_size,
                'risk_reward_tp1': rr_tp1,
                'nearest_resistance': nearest
            }

    # SHORT SIGNAL
    # For shorts, resistance zones become support
    # We want to SHORT when price is BELOW resistance (which acts as support)
    # and target moves DOWN to next support level

    if resistance_zones:
        # For shorting, we look at resistance zones BELOW current price
        # These act as support levels we don't want to short through

        # Find if we're above any resistance (which becomes support for shorts)
        below_support = all(current_price > current_price * (1 + z['threshold_high'])
                           for z in resistance_zones if z['strength'] == 'STRONG')

        if below_support:
            # Safe to short - no strong support nearby below
            # Use mirror of long strategy

            if resistance_zones:
                nearest = resistance_zones[0]

                # TP1: Mirror of long TP1 (90% down to support)
                tp1_pct = nearest['threshold_low'] * 0.9
                tp1_price = current_price * (1 - tp1_pct)

                # TP2: Within support zone
                tp2_pct = (nearest['threshold_low'] + nearest['threshold_high']) / 2
                tp2_price = current_price * (1 - tp2_pct)

                # SL: Based on ATR (above entry)
                sl_price = current_price * (1 + atr_stop_pct)

                # Position sizing: Same logic as longs
                if nearest['threshold_low'] < 0.005:
                    position_size = 0.5
                elif nearest['threshold_low'] < 0.01:
                    position_size = 0.75
                else:
                    position_size = 1.0

                risk = sl_price - current_price
                reward_tp1 = current_price - tp1_price
                rr_tp1 = reward_tp1 / risk if risk > 0 else 0

                signals['SHORT'] = {
                    'entry': current_price,
                    'tp1': tp1_price,
                    'tp1_pct': tp1_pct,
                    'tp2': tp2_price,
                    'tp2_pct': tp2_pct,
                    'sl': sl_price,
                    'sl_pct': atr_stop_pct,
                    'position_size': position_size,
                    'risk_reward_tp1': rr_tp1,
                    'nearest_support': nearest
                }

    return signals


def print_signals(signals: dict, current_price: float, resistance_zones: list):
    """Pretty print trading signals"""

    print('\n' + '=' * 80)
    print('TRADING SIGNALS')
    print('=' * 80)
    print(f'\nCurrent Price: ${current_price:,.2f}')
    print(f'Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

    print('\n' + '-' * 80)
    print('RESISTANCE ZONES:')
    print('-' * 80)

    if not resistance_zones:
        print('  No significant resistance zones identified')
    else:
        for i, zone in enumerate(resistance_zones, 1):
            low_price = current_price * (1 + zone['threshold_low'])
            high_price = current_price * (1 + zone['threshold_high'])
            print(f"  Zone {i}: ${low_price:,.2f} - ${high_price:,.2f} "
                  f"({zone['threshold_low']*100:.1f}% - {zone['threshold_high']*100:.1f}%) "
                  f"- Δ={zone['delta']:.4f} [{zone['strength']}]")

    # LONG SIGNAL
    print('\n' + '=' * 80)
    print('LONG SIGNAL')
    print('=' * 80)

    if signals['LONG']:
        sig = signals['LONG']
        print(f"  Entry:     ${sig['entry']:,.2f}")
        print(f"  TP1:       ${sig['tp1']:,.2f} ({sig['tp1_pct']*100:.2f}%) - Exit 70% of position")
        print(f"  TP2:       ${sig['tp2']:,.2f} ({sig['tp2_pct']*100:.2f}%) - Exit 30% of position")
        print(f"  Stop-Loss: ${sig['sl']:,.2f} ({sig['sl_pct']*100:.2f}%)")
        print(f"  Position Size: {sig['position_size']*100:.0f}% of normal")
        print(f"  Risk:Reward (TP1): 1:{sig['risk_reward_tp1']:.2f}")
        print(f"\n  Rationale: Nearest resistance at {sig['nearest_resistance']['threshold_low']*100:.1f}%")
        print(f"             TP1 set before resistance, TP2 targets move through resistance")
    else:
        print("  No clear LONG setup - resistance too close or unclear")

    # SHORT SIGNAL
    print('\n' + '=' * 80)
    print('SHORT SIGNAL')
    print('=' * 80)

    if signals['SHORT']:
        sig = signals['SHORT']
        print(f"  Entry:     ${sig['entry']:,.2f}")
        print(f"  TP1:       ${sig['tp1']:,.2f} ({sig['tp1_pct']*100:.2f}%) - Exit 70% of position")
        print(f"  TP2:       ${sig['tp2']:,.2f} ({sig['tp2_pct']*100:.2f}%) - Exit 30% of position")
        print(f"  Stop-Loss: ${sig['sl']:,.2f} ({sig['sl_pct']*100:.2f}%)")
        print(f"  Position Size: {sig['position_size']*100:.0f}% of normal")
        print(f"  Risk:Reward (TP1): 1:{sig['risk_reward_tp1']:.2f}")
        print(f"\n  Rationale: Resistance acts as support below at {sig['nearest_support']['threshold_low']*100:.1f}%")
        print(f"             TP1 set before support, TP2 targets move through support")
    else:
        print("  No clear SHORT setup - support too close below or unclear")

    print('\n' + '=' * 80)


def main():
    print('=' * 80)
    print('REAL-TIME RESISTANCE SIGNAL GENERATOR')
    print('=' * 80)

    # Load recent data
    print('\nLoading 20 months of data...')
    kline_size = 300
    end_date = datetime.now()
    start_date = end_date - timedelta(days=20 * 30)
    start_date_str = start_date.strftime('%Y-%m-%d')

    historical = loop.HistoricalData()
    historical.get_spot_klines(kline_size=kline_size, start_date_limit=start_date_str)
    full_data = historical.data
    print(f'Loaded {len(full_data):,} candles')

    # Get current price (last close)
    current_price = float(full_data['close'][-1])

    # Train resistance finder
    print('\nTraining multi-threshold resistance models...')

    uel = loop.UniversalExperimentLoop(
        data=full_data,
        single_file_model=mtrf
    )

    uel.run(
        experiment_name='resistance_signals',
        n_permutations=1,
        random_search=False
    )

    print('\nExtracting resistance zones...')

    # Get results
    extras = uel.extras[0]
    test_probabilities = extras['test_probabilities']

    # Calculate ATR-based stop
    atr_stop_pct = calculate_atr_stop(full_data, atr_multiplier=1.0)

    # Identify resistance zones from LATEST test sample
    resistance_zones = identify_resistance_zones(
        test_probabilities,
        mtrf.THRESHOLDS,
        strength_threshold=0.05
    )

    # Generate signals
    signals = generate_trading_signals(
        current_price,
        resistance_zones,
        atr_stop_pct
    )

    # Print signals
    print_signals(signals, current_price, resistance_zones)

    print('\nDone!')


if __name__ == '__main__':
    main()
