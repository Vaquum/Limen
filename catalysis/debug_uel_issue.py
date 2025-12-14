#!/usr/bin/env python3
"""
Debug script to find the exact issue with UEL returning no results
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import polars as pl
import loop
from loop.tests.utils.get_data import get_klines_data

def test_uel_directly():
    """Test UEL directly with a window to understand what's failing"""

    print("Loading data...")
    data = get_klines_data()

    # Get a substantial window of data (first 1000 rows)
    window_data = data.head(1000)
    print(f"Window data shape: {window_data.shape}")
    print(f"Window data columns: {window_data.columns}")
    print(f"Date range: {window_data['datetime'].min()} to {window_data['datetime'].max()}")

    print("\nTesting UEL with no specific parameters (random selection)...")

    try:
        # Create UEL instance
        uel = loop.UniversalExperimentLoop(
            data=window_data,
            single_file_model=loop.sfm.lightgbm.tradeline_long_binary
        )

        print("UEL created successfully")

        # Run with default params (UEL will randomly select)
        print("Running UEL experiment...")
        uel.run(
            experiment_name='debug_test',
            n_permutations=1,
            prep_each_round=True
        )

        print(f"Experiment completed. Log shape: {uel.experiment_log.shape}")

        if len(uel.experiment_log) > 0:
            print("\nUEL Results found!")
            result_row = uel.experiment_log.row(0, named=True)

            # Print all available keys
            print(f"Available keys in result: {list(result_row.keys())}")

            # Check for our expected metrics
            expected_metrics = ['auc', 'val_loss', 'precision', 'recall', 'f1',
                              'trading_return_net_pct', 'trading_win_rate_pct',
                              'trading_avg_win', 'trading_avg_loss']

            print("\nChecking expected metrics:")
            for metric in expected_metrics:
                value = result_row.get(metric, 'NOT_FOUND')
                print(f"  {metric}: {value}")

            return True
        else:
            print("❌ NO RESULTS from UEL experiment_log")
            return False

    except Exception as e:
        print(f"❌ ERROR during UEL execution: {str(e)}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()
        return False

def test_with_specific_params():
    """Test UEL with specific parameters to ensure it's not a param generation issue"""

    print("\n" + "="*60)
    print("Testing UEL with specific parameters...")

    data = get_klines_data()
    window_data = data.head(1000)

    # Define specific params
    specific_params = {
        'quantile_threshold': 0.75,
        'min_height_pct': 0.002,
        'max_duration_hours': 48,
        'lookahead_hours': 48,
        'long_threshold_percentile': 75,
        'confidence_threshold': 0.50,
        'position_size': 0.20,
        'min_stop_loss': 0.010,
        'max_stop_loss': 0.040,
        'atr_stop_multiplier': 1.5,
        'trailing_activation': 0.02,
        'trailing_distance': 0.5,
        'loser_timeout_hours': 24,
        'max_hold_hours': 48,
        'default_atr_pct': 0.015,
        'num_leaves': 63,
        'learning_rate': 0.1,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.9,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'n_estimators': 500
    }

    try:
        uel = loop.UniversalExperimentLoop(
            data=window_data,
            single_file_model=loop.sfm.lightgbm.tradeline_long_binary
        )

        # Create custom params function that returns our specific params
        def custom_params():
            return {k: [v] for k, v in specific_params.items()}

        print("Running UEL with specific parameters...")
        uel.run(
            experiment_name='debug_specific_params',
            n_permutations=1,
            prep_each_round=True,
            params=custom_params
        )

        print(f"Experiment completed. Log shape: {uel.experiment_log.shape}")

        if len(uel.experiment_log) > 0:
            print("✅ SUCCESS with specific params!")
            result_row = uel.experiment_log.row(0, named=True)

            # Check key metrics
            auc = result_row.get('auc', np.nan)
            trading_return = result_row.get('trading_return_net_pct', np.nan)

            print(f"AUC: {auc}")
            print(f"Trading Return: {trading_return}%")

            return True
        else:
            print("❌ STILL NO RESULTS with specific params")
            return False

    except Exception as e:
        print(f"❌ ERROR with specific params: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== DEBUGGING UEL ISSUE ===\n")

    # Test 1: Basic UEL functionality
    basic_success = test_uel_directly()

    # Test 2: UEL with specific parameters
    specific_success = test_with_specific_params()

    print("\n" + "="*60)
    print("DEBUG SUMMARY:")
    print(f"  Basic UEL test: {'✅ PASS' if basic_success else '❌ FAIL'}")
    print(f"  Specific params test: {'✅ PASS' if specific_success else '❌ FAIL'}")

    if not (basic_success or specific_success):
        print("\n🚨 CRITICAL: UEL is not producing any results at all!")
        print("This explains why the sensitivity analysis shows all NaN values.")
        print("The issue is likely in the UEL configuration or model execution.")
    elif basic_success or specific_success:
        print("\n✅ UEL can produce results, so the issue is elsewhere.")
        print("The problem might be in the parameter passing or window creation.")