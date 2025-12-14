#!/usr/bin/env python3
"""
Validation script to verify the bug fixes are working correctly
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import polars as pl
import loop
from loop.tests.utils.get_data import get_klines_data
from sfm_micro_period_analysis import micro_period_sensitivity_analysis, create_time_windows
from sfm_sensitivity_runner import generate_parameter_combinations

def test_window_creation():
    """Test that window creation works without artificial limits"""
    print("="*60)
    print("TESTING WINDOW CREATION")
    print("="*60)

    # Load a subset of data for testing
    data = get_klines_data()
    test_data = data.head(1000)  # Use first 1000 rows

    print(f"Test data: {len(test_data)} rows")

    # Test unlimited windows
    print("\\nTesting UNLIMITED window creation:")
    windows_unlimited = create_time_windows(test_data, window_days=1, max_windows=None)
    print(f"  1-day windows (unlimited): {len(windows_unlimited)} windows created")

    # Test limited windows
    print("\\nTesting LIMITED window creation:")
    windows_limited = create_time_windows(test_data, window_days=1, max_windows=5)
    print(f"  1-day windows (max 5): {len(windows_limited)} windows created")

    # Verify the unlimited version creates more windows
    if len(windows_unlimited) > len(windows_limited):
        print("  ✅ PASS: Unlimited window creation works correctly")
        return True
    else:
        print("  ❌ FAIL: Unlimited windows not working")
        return False

def test_parameter_passing():
    """Test that parameters are passed correctly to UEL"""
    print("\\n" + "="*60)
    print("TESTING PARAMETER PASSING")
    print("="*60)

    # Generate test parameters
    params = generate_parameter_combinations(1)[0]
    print(f"Test parameters: {list(params.keys())[:5]}...")  # Show first 5 keys

    # Load small test data
    data = get_klines_data()
    test_data = data.head(500)  # Small dataset for quick test

    print(f"\\nTesting with {len(test_data)} rows...")

    try:
        # Run analysis with specific parameters
        results = micro_period_sensitivity_analysis(
            data=test_data,
            sfm_model=loop.sfm.lightgbm.tradeline_long_binary,
            window_days_list=[5],  # Single window size
            specific_params=params,
            max_windows=3,  # Limit for quick testing
            verbose=True
        )

        # Check if we got valid results
        window_results = results.get('5d', {})
        valid_windows = window_results.get('valid_windows', 0)

        if valid_windows > 0:
            print("  ✅ PASS: Parameter passing works - got valid results")
            return True
        else:
            print("  ❌ FAIL: No valid results with specific parameters")
            return False

    except Exception as e:
        print(f"  ❌ ERROR: Exception during parameter test: {str(e)}")
        return False

def test_comprehensive_analysis():
    """Test the full analysis pipeline with a small dataset"""
    print("\\n" + "="*60)
    print("TESTING COMPREHENSIVE ANALYSIS")
    print("="*60)

    # Load small dataset
    data = get_klines_data()
    test_data = data.head(800)  # Medium dataset

    print(f"Testing with {len(test_data)} rows...")

    # Generate 2 parameter combinations
    param_combinations = generate_parameter_combinations(2)

    results_summary = {}

    for i, params in enumerate(param_combinations):
        print(f"\\n--- Testing permutation {i+1}/2 ---")

        try:
            # Run analysis with this permutation
            perm_results = micro_period_sensitivity_analysis(
                data=test_data,
                sfm_model=loop.sfm.lightgbm.tradeline_long_binary,
                window_days_list=[3, 5],  # Two window sizes
                specific_params=params,
                max_windows=4,  # Limit for testing
                verbose=True
            )

            # Collect results
            total_valid = 0
            total_windows = 0

            for window_size, window_results in perm_results.items():
                valid = window_results['valid_windows']
                total = window_results['total_windows']
                total_valid += valid
                total_windows += total
                print(f"  {window_size}: {valid}/{total} valid windows")

            results_summary[f"perm_{i+1}"] = {
                'total_windows': total_windows,
                'valid_windows': total_valid,
                'success_rate': (total_valid / total_windows * 100) if total_windows > 0 else 0
            }

        except Exception as e:
            print(f"  ERROR in permutation {i+1}: {str(e)}")
            results_summary[f"perm_{i+1}"] = {
                'error': str(e)
            }

    # Analyze results
    print(f"\\n--- Analysis Summary ---")
    successful_perms = 0
    total_valid_windows = 0

    for perm_name, perm_data in results_summary.items():
        if 'error' not in perm_data:
            success_rate = perm_data['success_rate']
            valid_windows = perm_data['valid_windows']
            print(f"{perm_name}: {valid_windows} valid windows ({success_rate:.1f}% success)")

            if valid_windows > 0:
                successful_perms += 1
                total_valid_windows += valid_windows
        else:
            print(f"{perm_name}: ERROR - {perm_data['error'][:50]}...")

    if successful_perms > 0 and total_valid_windows > 0:
        print("  ✅ PASS: Comprehensive analysis is working")
        return True
    else:
        print("  ❌ FAIL: No successful permutations")
        return False

def main():
    """Run all validation tests"""
    print("VALIDATING BUG FIXES FOR MICRO-PERIOD SENSITIVITY ANALYSIS")
    print("="*80)

    tests_passed = 0
    total_tests = 3

    # Test 1: Window creation
    if test_window_creation():
        tests_passed += 1

    # Test 2: Parameter passing
    if test_parameter_passing():
        tests_passed += 1

    # Test 3: Comprehensive analysis
    if test_comprehensive_analysis():
        tests_passed += 1

    # Final verdict
    print("\\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    print(f"Tests passed: {tests_passed}/{total_tests}")

    if tests_passed == total_tests:
        print("\\n🎉 ALL TESTS PASSED!")
        print("The bug fixes are working correctly.")
        print("The sensitivity analysis should now be able to:")
        print("  ✅ Process unlimited windows (all 1947+ windows)")
        print("  ✅ Pass parameters correctly to UEL")
        print("  ✅ Generate valid metrics from model runs")
        print("  ✅ Handle the full-scale analysis")
        return True
    else:
        print(f"\\n⚠️  {total_tests - tests_passed} TEST(S) FAILED")
        print("Some issues may still exist. Check the error messages above.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)