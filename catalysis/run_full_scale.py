#!/usr/bin/env python3
"""
Run full-scale micro-period sensitivity analysis with all windows
"""
import loop
import numpy as np
from loop.tests.utils.get_data import get_klines_data
from sfm_micro_period_analysis import micro_period_sensitivity_analysis

print("Loading full dataset...")
data = get_klines_data()
print(f"Dataset size: {len(data)} rows")
print(f"Date range: {data['datetime'].min()} to {data['datetime'].max()}")

print("\nRunning full-scale micro-period sensitivity analysis...")
results = micro_period_sensitivity_analysis(
    data=data,
    sfm_model=loop.sfm.lightgbm.tradeline_long_binary,
    window_days_list=[1, 2, 3],  # Original micro-periods as requested
    max_windows=None,  # NO LIMIT - process all windows
    verbose=True
)

print("\n" + "="*80)
print("FULL-SCALE RESULTS SUMMARY")
print("="*80)

for window_key, window_results in results.items():
    print(f"\n{window_key.upper()} WINDOWS:")
    print(f"  Total windows: {window_results.get('total_windows', 0)}")
    print(f"  Valid windows: {window_results.get('valid_windows', 0)}")

    if 'stability' in window_results:
        key_metrics = ['auc', 'trading_return_net_pct', 'win_loss_ratio']
        for metric in key_metrics:
            if metric in window_results['stability']:
                stats = window_results['stability'][metric]
                mean_val = stats.get('mean')
                if mean_val is not None and not np.isnan(mean_val):
                    print(f"    {metric}: mean={mean_val:.4f}, std={stats.get('std', 0):.4f}")