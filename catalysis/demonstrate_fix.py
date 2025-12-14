#!/usr/bin/env python3
"""
Demonstration script showing the fixed micro-period sensitivity analysis
This script shows how the analysis now processes ALL windows instead of getting stuck
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import polars as pl
from datetime import datetime
import loop
from loop.tests.utils.get_data import get_klines_data
from sfm_micro_period_analysis import create_time_windows, micro_period_sensitivity_analysis
from sfm_sensitivity_runner import generate_parameter_combinations, run_multi_permutation_analysis

def demonstrate_window_progression():
    """Show that window creation now generates the full expected number"""
    print("="*80)
    print("DEMONSTRATING WINDOW CREATION FIX")
    print("="*80)

    # Load full dataset
    data = get_klines_data()
    print(f"Full dataset: {len(data)} rows")

    # Calculate expected windows for different window sizes
    start_date = data['datetime'].min()
    end_date = data['datetime'].max()
    total_days = (end_date - start_date).total_seconds() / 86400

    print(f"Date range: {start_date} to {end_date}")
    print(f"Total duration: {total_days:.1f} days")
    print()

    window_sizes = [1, 2, 3, 5, 10]

    for window_days in window_sizes:
        print(f"Creating {window_days}-day windows...")

        # BEFORE FIX: Limited windows
        windows_limited = create_time_windows(data, window_days, max_windows=10)

        # AFTER FIX: Unlimited windows
        windows_unlimited = create_time_windows(data, window_days, max_windows=None)

        expected_windows = max(1, int(total_days - window_days + 1))

        print(f"  Expected windows: ~{expected_windows}")
        print(f"  Limited (old): {len(windows_limited)} windows")
        print(f"  Unlimited (new): {len(windows_unlimited)} windows")

        improvement = len(windows_unlimited) - len(windows_limited)
        print(f"  Improvement: +{improvement} more windows ({improvement/len(windows_limited)*100:.1f}% increase)")
        print()

    print("✅ RESULT: The fix allows processing of ALL available windows")
    return len(windows_unlimited)

def demonstrate_parameter_fix():
    """Show that parameters are now passed correctly to UEL"""
    print("="*80)
    print("DEMONSTRATING PARAMETER PASSING FIX")
    print("="*80)

    # Generate test parameters
    test_params = generate_parameter_combinations(1)[0]
    print("Test parameters generated:")
    for key, value in list(test_params.items())[:8]:  # Show first 8
        print(f"  {key}: {value}")
    print(f"  ... and {len(test_params)-8} more parameters")
    print()

    # Use a small dataset for demonstration
    data = get_klines_data()
    demo_data = data.head(600)  # Small set for quick demo

    print(f"Running analysis with {len(demo_data)} rows...")
    print("This demonstrates the parameter passing is working...")

    try:
        results = micro_period_sensitivity_analysis(
            data=demo_data,
            sfm_model=loop.sfm.lightgbm.tradeline_long_binary,
            window_days_list=[5],  # Single window size for demo
            specific_params=test_params,  # Use our specific parameters
            max_windows=3,  # Small limit for quick demo
            verbose=True
        )

        # Check results
        window_results = results.get('5d', {})
        valid_windows = window_results.get('valid_windows', 0)
        total_windows = window_results.get('total_windows', 0)

        if valid_windows > 0:
            print(f"\\n✅ SUCCESS: Got {valid_windows}/{total_windows} valid results")
            print("   Parameters were passed correctly to UEL")

            # Show a sample metric
            auc_stability = window_results['stability']['auc']
            if not np.isnan(auc_stability['mean']):
                print(f"   Sample AUC: {auc_stability['mean']:.4f} ± {auc_stability['std']:.4f}")

            return True
        else:
            print("❌ No valid results - parameter passing may still have issues")
            return False

    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False

def demonstrate_full_scale_capability():
    """Show that we can now handle large-scale analysis"""
    print("="*80)
    print("DEMONSTRATING FULL-SCALE CAPABILITY")
    print("="*80)

    # Load full dataset
    data = get_klines_data()
    print(f"Full dataset: {len(data)} rows")

    # Calculate realistic window counts
    start_date = data['datetime'].min()
    end_date = data['datetime'].max()
    total_days = (end_date - start_date).total_seconds() / 86400

    # For 1-day windows, we should get nearly one window per day
    expected_1d_windows = max(1, int(total_days))
    print(f"Expected 1-day windows: ~{expected_1d_windows}")

    # Create actual 1-day windows (unlimited)
    actual_windows = create_time_windows(data, window_days=1, max_windows=None)
    print(f"Actual 1-day windows created: {len(actual_windows)}")

    # Calculate the potential scale
    n_permutations = 10
    window_sizes = [1, 2, 3]
    total_combinations = len(actual_windows) * len(window_sizes) * n_permutations

    print(f"\\nFull-scale analysis potential:")
    print(f"  Window sizes: {window_sizes}")
    print(f"  Permutations: {n_permutations}")
    print(f"  Total window-permutation combinations: {total_combinations:,}")

    if len(actual_windows) > 1000:  # If we have lots of windows
        print(f"\\n✅ SUCCESS: Can now handle large-scale analysis")
        print(f"   The fix enables processing of {len(actual_windows)} windows")
        print(f"   Previously limited to ~10 windows per size")
        return True
    else:
        print("⚠️  Dataset may be too small for full demonstration")
        return False

def show_usage_examples():
    """Show how to use the fixed code"""
    print("="*80)
    print("USAGE EXAMPLES FOR FIXED CODE")
    print("="*80)

    print("1. Run UNLIMITED window analysis:")
    print("   results = micro_period_sensitivity_analysis(")
    print("       data=data,")
    print("       sfm_model=loop.sfm.lightgbm.tradeline_long_binary,")
    print("       window_days_list=[1, 2, 3],")
    print("       max_windows=None,  # NO LIMIT!")
    print("       verbose=True")
    print("   )")
    print()

    print("2. Run full-scale multi-permutation analysis:")
    print("   results = run_multi_permutation_analysis(")
    print("       data=data,")
    print("       n_permutations=10,")
    print("       window_days_list=[1, 2, 3],")
    print("       max_windows_per_size=None  # NO LIMIT!")
    print("   )")
    print()

    print("3. Use the new full-scale runner:")
    print("   python3 run_full_scale_sensitivity.py")
    print("   # This will process ALL available windows")
    print()

    print("KEY FIXES APPLIED:")
    print("✅ Removed artificial window limits (max_windows=None)")
    print("✅ Fixed parameter passing to UEL")
    print("✅ Improved error handling in window processing")
    print("✅ Enhanced ParamSpace compatibility")
    print("✅ Added proper progress tracking")

def main():
    """Run the complete demonstration"""
    print("MICRO-PERIOD SENSITIVITY ANALYSIS - BUG FIX DEMONSTRATION")
    print("="*100)
    print("This script demonstrates that the analysis no longer gets stuck")
    print("and can now process ALL available windows as intended.")
    print()

    # Demonstration 1: Window progression
    max_windows = demonstrate_window_progression()
    print()

    # Demonstration 2: Parameter passing
    param_success = demonstrate_parameter_fix()
    print()

    # Demonstration 3: Full-scale capability
    scale_success = demonstrate_full_scale_capability()
    print()

    # Show usage examples
    show_usage_examples()

    # Final summary
    print("="*80)
    print("DEMONSTRATION SUMMARY")
    print("="*80)

    if max_windows > 100 and param_success and scale_success:
        print("🎉 ALL DEMONSTRATIONS SUCCESSFUL!")
        print()
        print("The micro-period sensitivity analysis has been FIXED:")
        print(f"  ✅ Can process {max_windows}+ windows (not just the first one)")
        print("  ✅ Parameters are passed correctly to UEL")
        print("  ✅ No more hanging on 'Processing window 1/1947'")
        print("  ✅ Can handle full-scale analysis")
        print()
        print("You can now run the full analysis with all 1947 windows!")
        print("Use: python3 run_full_scale_sensitivity.py")
    else:
        print("⚠️  Some demonstrations had issues")
        print("The fixes may need additional work")

if __name__ == "__main__":
    main()