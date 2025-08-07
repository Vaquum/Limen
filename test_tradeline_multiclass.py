#!/usr/bin/env python3
"""
Test script for Tradeline Multiclass SFM
"""

import sys
sys.path.append('/Users/beyondsyntax/Loop')

from loop.data import HistoricalData
from loop.universal_experiment_loop import UniversalExperimentLoop
from loop.sfm.lightgbm import tradeline_multiclass

# Fetch data
print("Fetching historical data...")
historical = HistoricalData()
historical.get_historical_klines(
    kline_size=3600,  # 1 hour candles
    start_date_limit='2019-01-01',
    n_rows=50000  # About 2 years of hourly data
)

print(f"Data shape: {historical.data.shape}")
print(f"Columns: {historical.data.columns}")

# Initialize UEL with the SFM
print("\nInitializing Universal Experiment Loop...")
uel = UniversalExperimentLoop(
    data=historical.data,
    single_file_model=tradeline_multiclass
)

# Run a small experiment
print("\nRunning experiment with 3 permutations...")
print("This will test different quantile thresholds: [0.60, 0.70, 0.75]")

# Override params to test just quantile variations
def test_params():
    return {
        # Test different quantiles
        'quantile_threshold': [0.60, 0.70, 0.75],
        
        # Fix other parameters for testing
        'min_height_pct': [0.003],
        'max_duration_hours': [48],
        'objective': ['multiclass'],
        'num_class': [3],
        'metric': ['multi_logloss'],
        'boosting_type': ['gbdt'],
        'num_leaves': [31],
        'learning_rate': [0.05],
        'feature_fraction': [0.9],
        'bagging_fraction': [0.8],
        'bagging_freq': [5],
        'min_child_samples': [20],
        'lambda_l1': [0],
        'lambda_l2': [0],
        'verbose': [-1],
        'n_estimators': [100],  # Reduced for faster testing
        'lookahead_hours': [48],
        'profit_threshold': [0.034],
        'use_calibration': [False],  # Disabled for faster testing
        'calibration_method': ['isotonic'],
        'calibration_cv': [3]
    }

# Run experiment
uel.run(
    experiment_name='tradeline_multiclass_test',
    n_permutations=3,
    prep_each_round=True,  # Need to recompute lines for each quantile
    random_search=False,   # Test specific parameter combinations
    params=test_params
)

# Display results
print("\n" + "="*80)
print("EXPERIMENT RESULTS")
print("="*80)
print(uel.log_df)

# Show key metrics
print("\n" + "="*80)
print("SUMMARY BY QUANTILE THRESHOLD")
print("="*80)
results_df = uel.log_df.select(['quantile_threshold', 'accuracy', 'precision', 'recall', 'auc', 'val_loss'])
print(results_df)

# Check extras for one run
if uel.extras:
    print("\n" + "="*80)
    print("SAMPLE EXTRAS (from first run)")
    print("="*80)
    extras = uel.extras[0]
    print(f"Quantile threshold: {extras['quantile_threshold']}")
    print(f"Number of long lines: {extras['n_long_lines']}")
    print(f"Number of short lines: {extras['n_short_lines']}")
    print(f"Class distribution (test): {extras['class_distribution']['test']}")

print("\nTest completed successfully!")