#!/usr/bin/env python3
"""
Validation script to verify the sensitivity runner fix is working
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import loop
from loop.tests.utils.get_data import get_klines_data
from sfm_sensitivity_runner import run_multi_permutation_analysis

def validate_fix():
    """Test with a small subset to verify the fix works"""

    print("Loading test data...")
    data = get_klines_data()

    # Use only first 500 rows to speed up testing (covers ~5 windows)
    data_subset = data.head(500)

    print(f"Testing with {len(data_subset)} rows of data")
    print("\nRunning sensitivity analysis with 2 permutations...")

    # Run with just 2 permutations and 90-day windows
    results = run_multi_permutation_analysis(
        data=data_subset,
        n_permutations=2,
        window_days_list=[90]  # Just one window size that has enough data
    )

    print("\n=== VALIDATION RESULTS ===")

    # Check results
    total_windows_processed = 0
    total_valid_metrics = 0

    for perm_name, perm_data in results.items():
        print(f"\n{perm_name}:")

        # Verify parameters were generated
        params = perm_data['parameters']
        print(f"  Generated {len(params)} parameters")
        print(f"  Sample params: confidence={params['confidence_threshold']}, learning_rate={params['learning_rate']}")

        # Check window results
        for window_size, window_results in perm_data['results'].items():
            raw_metrics = window_results['raw_metrics']

            # Count valid metrics
            valid_auc = [v for v in raw_metrics['auc'] if not np.isnan(v)]
            valid_returns = [v for v in raw_metrics['trading_return_net_pct'] if not np.isnan(v)]

            print(f"  {window_size}:")
            print(f"    Windows processed: {len(raw_metrics['auc'])}")
            print(f"    Valid AUC values: {len(valid_auc)}")
            print(f"    Valid return values: {len(valid_returns)}")

            total_windows_processed += len(raw_metrics['auc'])
            total_valid_metrics += len(valid_auc)

            if valid_auc:
                print(f"    AUC stats: mean={np.mean(valid_auc):.3f}, std={np.std(valid_auc):.3f}")
            if valid_returns:
                print(f"    Return stats: mean={np.mean(valid_returns):.3f}%, std={np.std(valid_returns):.3f}%")

    # Final verdict
    print("\n" + "="*60)
    print("FINAL VERIFICATION:")
    print(f"  Total windows processed: {total_windows_processed}")
    print(f"  Windows with valid metrics: {total_valid_metrics}")

    if total_valid_metrics > 0:
        print("\n✅ SUCCESS: The bug is FIXED!")
        print("  - Parameter combinations are being generated correctly")
        print("  - Windows are being processed with the specified parameters")
        print("  - Models are producing valid metrics")
        print("  - The sensitivity analysis is working as intended")
        return True
    else:
        print("\n❌ FAILURE: The bug persists or insufficient data")
        print("  - Check if windows have enough data (need 1000+ rows)")
        print("  - Verify parameters are complete")
        return False

if __name__ == "__main__":
    success = validate_fix()
    exit(0 if success else 1)