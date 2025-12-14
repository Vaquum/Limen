#!/usr/bin/env python3
"""
Test UEL experiment with tradeline multiclass and explorer
Uses test data since database connection is not available
"""

import loop
import warnings
warnings.filterwarnings('ignore')
from loop.tests.utils.get_data import get_klines_data

def test_uel_with_explorer():
    """Run UEL experiment and launch explorer"""

    print("Getting test data...")
    data = get_klines_data()  # Use test data instead of live database

    print(f"Data shape: {data.shape}")
    print(f"Date range: {data['datetime'].min()} to {data['datetime'].max()}")

    print("Creating UEL with tradeline long binary...")
    uel = loop.UniversalExperimentLoop(data=data,
                                       single_file_model=loop.sfm.lightgbm.tradeline_long_binary)

    print("Running experiment with 20 permutations...")
    uel.run(experiment_name='tradeline_test_demo',
            n_permutations=20,
            prep_each_round=True)

    print("Experiment completed!")
    print(f"Results shape: {uel.experiment_log.shape}")

    # Clean non-serializable columns from experiment log before explorer
    print("Cleaning experiment log for explorer...")
    original_log = uel.experiment_log

    # Remove columns that contain non-serializable objects
    serializable_cols = []
    for col in original_log.columns:
        try:
            # Test if column contains serializable data
            test_val = original_log[col][0] if len(original_log) > 0 else None
            if col.startswith('_') and hasattr(test_val, '__class__') and 'sklearn' in str(type(test_val)):
                print(f"Skipping non-serializable column: {col}")
                continue
            serializable_cols.append(col)
        except:
            serializable_cols.append(col)

    # Create clean version for explorer
    uel.experiment_log = original_log.select(serializable_cols)
    print(f"Cleaned results shape: {uel.experiment_log.shape}")

    print("Launching explorer on localhost...")
    # Import the explorer function directly to specify localhost
    from loop.explorer.loop_explorer import loop_explorer
    loop_explorer(uel, 'localhost')

if __name__ == "__main__":
    test_uel_with_explorer()