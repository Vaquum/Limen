#!/usr/bin/env python3
"""
Validate Resistance Predictions

Tests whether ML-predicted resistance zones actually correspond to price resistance.
For each test sample:
1. Identify predicted resistance zones
2. Track future price action
3. Measure if price stalls/reverses at predicted levels
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


def validate_resistance_zone(current_price: float, zone: dict,
                             future_prices: np.ndarray, lookahead_bars: int = 96) -> dict:
    """
    Validate a single resistance zone prediction.

    Returns:
    - reached: Did price reach the zone?
    - penetration: How far into the zone did price go? (0.0 = at low edge, 1.0 = at high edge, >1.0 = broke through)
    - stalled: Did price stall within zone? (failed to break through high edge)
    - reversed: Did price reverse after entering zone?
    - max_price_reached: Highest price reached
    """

    zone_low = current_price * (1 + zone['threshold_low'])
    zone_high = current_price * (1 + zone['threshold_high'])

    max_price = np.max(future_prices)

    result = {
        'threshold_low': zone['threshold_low'],
        'threshold_high': zone['threshold_high'],
        'strength': zone['strength'],
        'delta': zone['delta'],
        'reached': False,
        'penetration': 0.0,
        'stalled': False,
        'reversed': False,
        'max_price': max_price,
        'max_gain_pct': (max_price - current_price) / current_price
    }

    # Did price reach the zone?
    if max_price >= zone_low:
        result['reached'] = True

        # How far into the zone?
        if max_price <= zone_high:
            # Stalled within zone
            penetration = (max_price - zone_low) / (zone_high - zone_low)
            result['penetration'] = penetration
            result['stalled'] = True
        else:
            # Broke through
            result['penetration'] = 1.0 + (max_price - zone_high) / (zone_high - zone_low)
            result['stalled'] = False

        # Check for reversal (price went back below zone_low after reaching zone)
        reached_zone_idx = None
        for i, price in enumerate(future_prices):
            if price >= zone_low:
                reached_zone_idx = i
                break

        if reached_zone_idx is not None and reached_zone_idx < len(future_prices) - 1:
            # Check if price reversed after reaching zone
            post_zone_prices = future_prices[reached_zone_idx:]
            if len(post_zone_prices) > 0:
                final_price = post_zone_prices[-1]
                if final_price < zone_low:
                    result['reversed'] = True

    return result


def analyze_resistance_accuracy(test_data: pl.DataFrame, test_probabilities: dict,
                                thresholds: list, lookahead_bars: int = 96) -> dict:
    """
    Analyze accuracy of resistance predictions across all test samples.
    """

    print('\nAnalyzing resistance prediction accuracy...')

    n_samples = len(test_probabilities[thresholds[0]])
    test_closes = test_data['close'].to_numpy()

    all_validations = []

    # For each test sample
    for i in range(n_samples - lookahead_bars):
        if i % 1000 == 0:
            print(f'  Processing sample {i}/{n_samples - lookahead_bars}...')

        current_price = test_closes[i]
        future_prices = test_closes[i:i+lookahead_bars]

        # Identify resistance zones for this sample
        zones = []
        probs = [test_probabilities[t][i] for t in thresholds]

        for j in range(len(thresholds) - 1):
            delta = probs[j] - probs[j + 1]

            if delta > 0.05:
                strength = 'STRONG'
            elif delta > 0.025:
                strength = 'MEDIUM'
            elif delta > 0.0125:
                strength = 'WEAK'
            else:
                continue

            zones.append({
                'threshold_low': thresholds[j],
                'threshold_high': thresholds[j + 1],
                'delta': delta,
                'strength': strength
            })

        # Validate each zone
        for zone in zones:
            validation = validate_resistance_zone(
                current_price, zone, future_prices, lookahead_bars
            )
            all_validations.append(validation)

    print(f'  Completed. Analyzed {len(all_validations)} resistance zone predictions')

    # Calculate statistics
    if len(all_validations) == 0:
        print('No resistance zones found!')
        return {}

    # Overall stats
    reached = [v for v in all_validations if v['reached']]
    stalled = [v for v in all_validations if v['stalled']]
    reversed = [v for v in all_validations if v['reversed']]
    broke_through = [v for v in reached if not v['stalled']]

    # By strength
    strong_zones = [v for v in all_validations if v['strength'] == 'STRONG']
    medium_zones = [v for v in all_validations if v['strength'] == 'MEDIUM']
    weak_zones = [v for v in all_validations if v['strength'] == 'WEAK']

    stats = {
        'total_zones': len(all_validations),
        'reached_rate': len(reached) / len(all_validations),
        'stalled_rate': len(stalled) / len(reached) if len(reached) > 0 else 0,
        'reversed_rate': len(reversed) / len(reached) if len(reached) > 0 else 0,
        'breakthrough_rate': len(broke_through) / len(reached) if len(reached) > 0 else 0,
        'avg_penetration': np.mean([v['penetration'] for v in reached]) if len(reached) > 0 else 0,
        'by_strength': {
            'STRONG': {
                'count': len(strong_zones),
                'reached': len([v for v in strong_zones if v['reached']]) / len(strong_zones) if len(strong_zones) > 0 else 0,
                'stalled': len([v for v in strong_zones if v['stalled']]) / len([v for v in strong_zones if v['reached']]) if len([v for v in strong_zones if v['reached']]) > 0 else 0,
                'reversed': len([v for v in strong_zones if v['reversed']]) / len([v for v in strong_zones if v['reached']]) if len([v for v in strong_zones if v['reached']]) > 0 else 0
            },
            'MEDIUM': {
                'count': len(medium_zones),
                'reached': len([v for v in medium_zones if v['reached']]) / len(medium_zones) if len(medium_zones) > 0 else 0,
                'stalled': len([v for v in medium_zones if v['stalled']]) / len([v for v in medium_zones if v['reached']]) if len([v for v in medium_zones if v['reached']]) > 0 else 0,
                'reversed': len([v for v in medium_zones if v['reversed']]) / len([v for v in medium_zones if v['reached']]) if len([v for v in medium_zones if v['reached']]) > 0 else 0
            },
            'WEAK': {
                'count': len(weak_zones),
                'reached': len([v for v in weak_zones if v['reached']]) / len(weak_zones) if len(weak_zones) > 0 else 0,
                'stalled': len([v for v in weak_zones if v['stalled']]) / len([v for v in weak_zones if v['reached']]) if len([v for v in weak_zones if v['reached']]) > 0 else 0,
                'reversed': len([v for v in weak_zones if v['reversed']]) / len([v for v in weak_zones if v['reached']]) if len([v for v in weak_zones if v['reached']]) > 0 else 0
            }
        },
        'all_validations': all_validations
    }

    return stats


def print_validation_results(stats: dict):
    """Pretty print validation results"""

    print('\n' + '=' * 80)
    print('RESISTANCE PREDICTION VALIDATION')
    print('=' * 80)

    print(f'\nTotal Resistance Zones Predicted: {stats["total_zones"]}')
    print(f'Reached Rate: {stats["reached_rate"]*100:.1f}% (price reached the predicted zone)')

    print('\n' + '-' * 80)
    print('BEHAVIOR AT PREDICTED RESISTANCE:')
    print('-' * 80)
    print(f'Stalled Rate: {stats["stalled_rate"]*100:.1f}% (price stopped within zone)')
    print(f'Reversed Rate: {stats["reversed_rate"]*100:.1f}% (price reversed after reaching zone)')
    print(f'Breakthrough Rate: {stats["breakthrough_rate"]*100:.1f}% (price broke through zone)')
    print(f'Avg Penetration: {stats["avg_penetration"]:.2f} (0=low edge, 1=high edge, >1=broke through)')

    print('\n' + '-' * 80)
    print('BY STRENGTH:')
    print('-' * 80)

    for strength in ['STRONG', 'MEDIUM', 'WEAK']:
        s = stats['by_strength'][strength]
        print(f'\n{strength} Zones ({s["count"]} total):')
        print(f'  Reached: {s["reached"]*100:.1f}%')
        print(f'  Stalled: {s["stalled"]*100:.1f}%')
        print(f'  Reversed: {s["reversed"]*100:.1f}%')

    print('\n' + '=' * 80)

    # Interpretation guide
    print('\nINTERPRETATION:')
    print('-' * 80)
    print('Good resistance prediction should show:')
    print('  - High reached rate (price actually gets to predicted zones)')
    print('  - High stalled rate (price stops at zones rather than breaking through)')
    print('  - High reversed rate (price reverses at zones)')
    print('  - STRONG zones should have higher stalled/reversed rates than WEAK zones')
    print('\nPoor resistance prediction would show:')
    print('  - Low reached rate (predicted zones are never tested)')
    print('  - High breakthrough rate (zones don\'t actually provide resistance)')
    print('  - WEAK zones performing same as STRONG zones (no discriminative power)')
    print('\n' + '=' * 80)


def main():
    print('=' * 80)
    print('RESISTANCE PREDICTION VALIDATION')
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
        experiment_name='resistance_validation',
        n_permutations=1,
        random_search=False
    )

    # Get results
    print('\nExtracting test data and probabilities...')

    extras = uel.extras[0]
    test_probabilities = extras['test_probabilities']

    # Get test data
    alignment = uel._alignment[0]
    first_test_dt = alignment['first_test_datetime']
    last_test_dt = alignment['last_test_datetime']

    test_data = full_data.filter(
        (pl.col('datetime') >= first_test_dt) &
        (pl.col('datetime') <= last_test_dt)
    )

    print(f'Test period: {first_test_dt} to {last_test_dt}')
    print(f'Test samples: {len(test_data)}')

    # Validate resistance predictions
    stats = analyze_resistance_accuracy(
        test_data,
        test_probabilities,
        mtrf.THRESHOLDS,
        lookahead_bars=96  # 96 * 5min = 8 hours
    )

    # Print results
    print_validation_results(stats)

    # Save detailed results
    print('\nSaving detailed validation results...')

    validations_df = pl.DataFrame(stats['all_validations'])
    output_file = '/Users/beyondsyntax/Loop/catalysis/resistance_validation_results.csv'
    validations_df.write_csv(output_file)
    print(f'Detailed results saved to: {output_file}')

    print('\nDone!')


if __name__ == '__main__':
    main()
