#!/usr/bin/env python3
import loop
from loop.tests.utils.get_data import get_klines_data_small
import polars as pl
import pandas as pd
import numpy as np
from sfm_micro_period_analysis import micro_period_sensitivity_analysis

# Get small test data
data = get_klines_data_small()

# Test with just first 600 rows for very quick testing
test_data = data.head(600)

print(f"Running quick test with {len(test_data)} rows...")
print(f"Data shape: {test_data.shape}")
print(f"Data columns: {test_data.columns}")

# Show date range for context
print(f"Date range: {test_data['datetime'].min()} to {test_data['datetime'].max()}")

# Run the analysis
try:
    results = micro_period_sensitivity_analysis(
        data=test_data,
        sfm_model=loop.sfm.lightgbm.tradeline_long_binary,
        window_days_list=[3, 5],  # Two window sizes for quick test
        max_windows=5,  # Maximum 5 windows per size
        verbose=True
    )

    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    for window_key, window_results in results.items():
        print(f"\n{window_key.upper()} WINDOWS:")
        print(f"  Total windows: {window_results.get('total_windows', 0)}")
        print(f"  Valid windows: {window_results.get('valid_windows', 0)}")

        if 'stability' in window_results and window_results.get('valid_windows', 0) > 0:
            print("  Stability metrics:")
            key_metrics = ['auc', 'trading_return_net_pct', 'win_loss_ratio']
            for metric in key_metrics:
                if metric in window_results['stability']:
                    stats = window_results['stability'][metric]
                    if isinstance(stats, dict) and 'mean' in stats:
                        mean_val = stats.get('mean')
                        std_val = stats.get('std')
                        if mean_val is not None and not np.isnan(mean_val):
                            print(f"    {metric}: mean={mean_val:.4f}, std={std_val:.4f}")
                        else:
                            print(f"    {metric}: No valid data")

except Exception as e:
    print(f"Error during analysis: {e}")
    import traceback
    traceback.print_exc()