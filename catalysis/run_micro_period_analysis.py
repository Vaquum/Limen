#!/usr/bin/env python3
"""
Optimized Micro-Period Sensitivity Analysis
Tests model stability across very short time windows (4, 5, 6, 7 days)
"""
import loop
import numpy as np
from loop.tests.utils.get_data import get_klines_data
from sfm_micro_period_analysis import micro_period_sensitivity_analysis

print("Loading full dataset...")
data = get_klines_data()
print(f"Dataset size: {len(data)} rows")
print(f"Date range: {data['datetime'].min()} to {data['datetime'].max()}")

print("\nRunning micro-period sensitivity analysis...")
print("Testing micro-periods: 4, 5, 6, 7 days (52-91 rows per window)")
print("This provides adequate data for SFM while maintaining micro-period focus")

results = micro_period_sensitivity_analysis(
    data=data,
    sfm_model=loop.sfm.lightgbm.tradeline_long_binary,
    window_days_list=[4, 5, 6, 7],  # Micro-periods with adequate data
    max_windows=None,  # NO LIMIT - process all windows
    verbose=True
)

print("\n" + "="*80)
print("MICRO-PERIOD SENSITIVITY ANALYSIS RESULTS")
print("="*80)

for window_key, window_results in results.items():
    print(f"\n{window_key.upper()} WINDOWS:")
    print(f"  Total windows: {window_results.get('total_windows', 0)}")
    print(f"  Valid windows: {window_results.get('valid_windows', 0)}")

    if 'stability' in window_results and window_results.get('valid_windows', 0) > 0:
        key_metrics = ['auc', 'trading_return_net_pct', 'win_loss_ratio']
        print(f"  Stability metrics:")
        for metric in key_metrics:
            if metric in window_results['stability']:
                stats = window_results['stability'][metric]
                mean_val = stats.get('mean')
                if mean_val is not None and not np.isnan(mean_val):
                    print(f"    {metric}: mean={mean_val:.4f}, std={stats.get('std', 0):.4f}, coeff_var={stats.get('coeff_var', 0):.4f}")
                else:
                    print(f"    {metric}: No valid data")
    else:
        print("  No valid results obtained")

# Summary insights
print(f"\n{'='*80}")
print("MICRO-PERIOD STABILITY INSIGHTS")
print(f"{'='*80}")

valid_results = {k: v for k, v in results.items() if v.get('valid_windows', 0) > 0}

if valid_results:
    print(f"\n✓ Successfully analyzed {len(valid_results)} micro-period window sizes")

    # Find most stable window size by coefficient of variation
    for metric in ['auc', 'trading_return_net_pct']:
        print(f"\n{metric.upper()} Stability Ranking (lower coefficient of variation = more stable):")
        stability_scores = []

        for window_key, window_results in valid_results.items():
            if metric in window_results['stability']:
                coeff_var = window_results['stability'][metric].get('coeff_var')
                if coeff_var is not None and not np.isnan(coeff_var):
                    stability_scores.append((window_key, coeff_var, window_results['stability'][metric].get('mean')))

        # Sort by coefficient of variation (ascending = more stable)
        stability_scores.sort(key=lambda x: x[1])

        for i, (window_key, coeff_var, mean_val) in enumerate(stability_scores[:3], 1):
            print(f"  {i}. {window_key}: CoeffVar={coeff_var:.4f}, Mean={mean_val:.4f}")
else:
    print("✗ No valid results obtained. Micro-periods may be too small for this dataset.")
    print("Consider using larger windows (7-14 days) for meaningful analysis.")

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE")
print(f"{'='*80}")