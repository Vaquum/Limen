#!/usr/bin/env python3
"""
Standalone test for micro-period sensitivity analysis using synthetic data.
This test doesn't require any external data files.
"""

import sys
import os
sys.path.insert(0, '/Users/beyondsyntax/Loop')
sys.path.insert(0, '/Users/beyondsyntax/Loop/catalysis')

import polars as pl
import numpy as np
from datetime import datetime, timedelta
from sfm_micro_period_analysis import micro_period_sensitivity_analysis
import loop.sfm.lightgbm

# Create synthetic data that mimics real market data
np.random.seed(42)
n_rows = 2000  # Enough for multiple windows

# Generate timestamps (hourly data)
base_time = datetime(2024, 1, 1)
timestamps = [base_time + timedelta(hours=i) for i in range(n_rows)]

# Generate price data with realistic patterns
prices = [100.0]
for _ in range(1, n_rows):
    # Add trend and volatility
    change = np.random.normal(0.0001, 0.01)  # Small drift with volatility
    new_price = prices[-1] * (1 + change)
    prices.append(max(new_price, 1.0))  # Ensure positive prices

# Create OHLCV data
data_dict = {
    'datetime': timestamps,  # Use 'datetime' instead of 'timestamp'
    'open': [],
    'high': [],
    'low': [],
    'close': prices,
    'volume': []
}

for i, close in enumerate(prices):
    # Generate OHLC based on close price
    daily_range = abs(np.random.normal(0, 0.005)) * close
    open_price = close + np.random.uniform(-daily_range/2, daily_range/2)
    high_price = max(open_price, close) + np.random.uniform(0, daily_range/2)
    low_price = min(open_price, close) - np.random.uniform(0, daily_range/2)
    volume = int(1000000 * (1 + np.random.uniform(-0.5, 1.0)))

    data_dict['open'].append(open_price)
    data_dict['high'].append(high_price)
    data_dict['low'].append(low_price)
    data_dict['volume'].append(volume)

# Create Polars DataFrame
df = pl.DataFrame(data_dict)

print(f"Created synthetic dataset with {len(df)} rows")
print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
print(f"Price range: ${df['close'].min():.2f} to ${df['close'].max():.2f}")
print(f"\nFirst few rows:")
print(df.head())

# Run the sensitivity analysis
print("\n" + "="*60)
print("Running micro-period sensitivity analysis...")
print("="*60)

try:
    results = micro_period_sensitivity_analysis(
        data=df,
        sfm_model=loop.sfm.lightgbm.tradeline_long_binary,
        window_days_list=[5, 10]  # Test with 5 and 10 day windows
    )

    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    for window_key, window_results in results.items():
        print(f"\n{window_key} Analysis:")
        print(f"  Total windows analyzed: {window_results.get('total_windows', 0)}")
        print(f"  Windows with valid metrics: {window_results.get('valid_windows', 0)}")

        if 'stability' in window_results and window_results.get('valid_windows', 0) > 0:
            print("\n  Key Stability Metrics:")
            # Focus on most important metrics
            key_metrics = ['auc', 'trading_return_net_pct', 'trading_win_rate_pct', 'f1']

            for metric in key_metrics:
                if metric in window_results['stability']:
                    stats = window_results['stability'][metric]
                    if isinstance(stats, dict) and 'mean' in stats:
                        mean_val = stats.get('mean')
                        std_val = stats.get('std')
                        cv_val = stats.get('coeff_var')

                        if mean_val is not None and not pl.Series([mean_val]).is_nan()[0]:
                            print(f"\n    {metric.upper()}:")
                            print(f"      Mean: {mean_val:.4f}")
                            print(f"      Std Dev: {std_val:.4f}")
                            if cv_val is not None and not pl.Series([cv_val]).is_nan()[0]:
                                print(f"      Coeff of Variation: {cv_val:.4f}")

                            # Show range if available
                            p5 = stats.get('percentile_5')
                            p95 = stats.get('percentile_95')
                            if p5 is not None and p95 is not None:
                                if not pl.Series([p5]).is_nan()[0] and not pl.Series([p95]).is_nan()[0]:
                                    print(f"      90% Range: [{p5:.4f}, {p95:.4f}]")

            # Show raw metrics sample if available
            if 'raw_metrics' in window_results:
                print("\n  Sample Raw Metrics (first 3 windows):")
                for metric in key_metrics:
                    if metric in window_results['raw_metrics']:
                        values = window_results['raw_metrics'][metric][:3]
                        if values:
                            print(f"    {metric}: {[f'{v:.4f}' for v in values]}")

    print("\n" + "="*60)
    print("Analysis complete!")

except Exception as e:
    print(f"\nError during analysis: {e}")
    import traceback
    traceback.print_exc()

    # Try to provide more context
    print("\n\nDebugging information:")
    print(f"DataFrame shape: {df.shape}")
    print(f"DataFrame columns: {df.columns}")
    print(f"DataFrame dtypes: {df.dtypes}")