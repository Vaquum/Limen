#!/usr/bin/env python3
"""
Quick test to verify the sensitivity runner is working with the bug fix
"""

import warnings
warnings.filterwarnings('ignore')

import loop
from loop.tests.utils.get_data import get_klines_data
from sfm_micro_period_analysis import micro_period_sensitivity_analysis
from sfm_sensitivity_runner import run_multi_permutation_analysis
import numpy as np

def test_fix():
    print("Loading test data...")
    data = get_klines_data()

    print("\nTesting with 1 permutation and small windows (5 windows per size)...")

    # Limit the data to speed up testing
    data = data.head(2000)  # Use only first few months of data

    # Run the multi-permutation analysis
    results = run_multi_permutation_analysis(
        data=data,
        n_permutations=1,  # Just 1 permutation for testing
        window_days_list=[30, 60]  # Just 2 window sizes
    )

    # Check if we got results
    print("\n=== RESULTS CHECK ===")
    for perm_name, perm_data in results.items():
        print(f"\n{perm_name}:")
        print(f"  Parameters provided: {len(perm_data['parameters'])} parameters")

        for window_size, window_results in perm_data['results'].items():
            raw_metrics = window_results['raw_metrics']
            stability = window_results['stability']

            # Count valid metrics
            auc_values = [v for v in raw_metrics['auc'] if not np.isnan(v)]
            return_values = [v for v in raw_metrics['trading_return_net_pct'] if not np.isnan(v)]

            print(f"  {window_size}:")
            print(f"    Total windows: {window_results['total_windows']}")
            print(f"    Valid AUC values: {len(auc_values)}")
            print(f"    Valid return values: {len(return_values)}")

            if auc_values:
                print(f"    AUC mean: {np.mean(auc_values):.4f}")
                print(f"    AUC std: {np.std(auc_values):.4f}")
            else:
                print(f"    No valid AUC values")

    # Verify fix worked
    success = False
    for perm_name, perm_data in results.items():
        for window_size, window_results in perm_data['results'].items():
            if window_results['valid_windows'] > 0:
                success = True
                break

    if success:
        print("\n✅ SUCCESS: The bug is fixed! Windows are being processed and producing metrics.")
    else:
        print("\n❌ FAILURE: Still no valid windows. The bug may not be fully resolved.")

    return success

if __name__ == "__main__":
    test_fix()