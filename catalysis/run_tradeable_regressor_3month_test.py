#!/usr/bin/env python3
'''
Run tradeable_regressor with UEL and generate 1-week prediction graph.

Fetches sufficient data to ensure 1 week in test split, runs UEL,
extracts predictions, and creates visualization.
'''

import warnings
warnings.filterwarnings('ignore')

import loop
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def run_tradeable_regressor_1week():
    '''
    Execute tradeable_regressor with UEL and plot 1-week predictions.
    '''

    print('Fetching data for 1-week test period...')

    kline_size = 300

    end_date = datetime.now()
    start_date = end_date - timedelta(days=70)
    start_date_str = start_date.strftime('%Y-%m-%d')

    print(f'Start date: {start_date_str}')
    print(f'End date: {end_date.strftime("%Y-%m-%d")}')
    print(f'Kline size: {kline_size}s (5 minutes)')

    historical = loop.HistoricalData()
    historical.get_spot_klines(kline_size=kline_size, start_date_limit=start_date_str)

    total_rows = len(historical.data)
    test_rows = int(total_rows * 0.15)
    test_days = (test_rows * kline_size) / (60 * 60 * 24)

    print(f'Total data points: {total_rows:,}')
    print(f'Expected test points: {test_rows:,} (~{test_days:.1f} days)')

    print('\nInitializing UEL with tradeable_regressor...')

    uel = loop.UniversalExperimentLoop(
        data=historical.data,
        single_file_model=loop.sfm.lightgbm.tradeable_regressor
    )

    print('Running experiment (this may take several minutes)...')

    uel.run(
        experiment_name='tradeable_regressor_1week',
        n_permutations=20,
        prep_each_round=True,
        random_search=True
    )

    print('\nExtracting predictions from best round...')

    best_round = uel.experiment_log.sort('val_rmse')[0]
    best_round_idx = best_round['id'].item()

    print(f'Best round: {best_round_idx} (val_rmse: {best_round["val_rmse"].item():.6f})')

    predictions = uel.preds[best_round_idx]
    test_clean = uel.extras[best_round_idx]['test_clean']
    test_datetimes = test_clean['datetime'].to_list()

    if hasattr(predictions, 'to_numpy'):
        predictions = predictions.to_numpy()

    predictions = predictions.flatten()
    predictions_pct = predictions * 100

    print(f'Test predictions: {len(predictions):,} points')
    print(f'Date range: {test_datetimes[0]} to {test_datetimes[-1]}')
    print(f'Prediction stats: min={predictions_pct.min():.4f}%, max={predictions_pct.max():.4f}%, mean={predictions_pct.mean():.4f}%')

    print('\nCreating prediction plot...')

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    ax.plot(test_datetimes, predictions_pct, linewidth=0.8, alpha=0.7, color='blue')
    ax.set_title('Tradeable Regressor Predictions - 1 Week Test Period', fontsize=14, fontweight='bold')
    ax.set_xlabel('Datetime', fontsize=11)
    ax.set_ylabel('Tradeable Score (%)', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

    percentiles = [50, 75, 90, 95, 99]
    p_values = np.percentile(predictions_pct, percentiles)

    for p, pval in zip(percentiles, p_values):
        ax.axhline(y=pval, color='red', linestyle='--', alpha=0.4, linewidth=1)
        ax.text(test_datetimes[-1], pval, f' P{p}: {pval:.3f}%', fontsize=9, va='center')

    plt.tight_layout()

    output_path = '/Users/beyondsyntax/Loop/catalysis/tradeable_regressor_predictions_1week.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'\nPlot saved to: {output_path}')

    results_summary = {
        'total_data_points': total_rows,
        'test_data_points': len(predictions),
        'test_days': (len(predictions) * kline_size) / (60 * 60 * 24),
        'best_round': best_round_idx,
        'best_val_rmse': float(best_round['val_rmse'].item()),
        'prediction_min_pct': float(predictions_pct.min()),
        'prediction_max_pct': float(predictions_pct.max()),
        'prediction_mean_pct': float(predictions_pct.mean()),
        'prediction_std_pct': float(predictions_pct.std()),
        'percentiles_pct': {f'p{p}': float(pval) for p, pval in zip(percentiles, p_values)}
    }

    print('\n=== Results Summary ===')
    for key, value in results_summary.items():
        if isinstance(value, dict):
            print(f'{key}:')
            for k, v in value.items():
                print(f'  {k}: {v:.4f}%')
        elif 'pct' in key:
            print(f'{key}: {value:.4f}%')
        else:
            print(f'{key}: {value}')

    return uel, predictions, test_datetimes, results_summary


if __name__ == '__main__':
    uel, predictions, datetimes, summary = run_tradeable_regressor_1week()
    print('\nDone!')
