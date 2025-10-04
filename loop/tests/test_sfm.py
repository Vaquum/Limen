import uuid
import sys
import traceback
import pandas as pd
import polars as pl

import loop
from loop import sfm
from loop.tests.utils.cleanup import cleanup_csv_files, setup_cleanup_handlers
# from loop.tests.utils.get_data import get_klines_data_small


def get_klines_data_fast():
    '''
    Get optimized klines data for fast testing.
    Uses every 4th row (8h intervals) and limited rows for speed.
    '''
    df = pd.read_csv('datasets/klines_2h_2020_2025.csv', nrows=8000)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.iloc[::4].reset_index(drop=True)  # Every 4th row (8h intervals)
    df = pl.from_pandas(df)
    return df


def get_klines_data_medium():
    '''
    Get medium-sized optimized klines data for models requiring larger datasets.
    Uses full dataset with 6h intervals to ensure sufficient data for regime models.
    '''
    df = pd.read_csv('datasets/klines_2h_2020_2025.csv')  # Use full dataset
    df['datetime'] = pd.to_datetime(df['datetime'])
    # Every 3rd row (6h intervals) -> ~7.8k rows
    df = df.iloc[::3].reset_index(drop=True)
    df = pl.from_pandas(df)
    return df


def get_klines_data_large():
    '''
    Get large dataset for models requiring 20k+ rows (like regime_multiclass).
    Uses full dataset to ensure sufficient data.
    '''
    df = pd.read_csv(
        'datasets/klines_2h_2020_2025.csv')  # Use full dataset (~23k rows)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = pl.from_pandas(df)
    return df


def get_klines_data_small_fast():
    '''
    Get small optimized klines data for fast testing.
    Uses every 6th row (12h intervals) for maximum speed.
    '''
    df = pd.read_csv('datasets/klines_2h_2020_2025.csv', nrows=3000)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.iloc[::6].reset_index(drop=True)  # Every 6th row (12h intervals)
    df = pl.from_pandas(df)
    return df


def test_sfm():

    tests = [
        # COLUMN ORDER: sfm, data_endpoint, prep_each_round, log, uses_manifest
        (sfm.reference.random, get_klines_data_fast, True, False),
        (sfm.reference.xgboost, get_klines_data_fast, False, False),
        (sfm.reference.logreg, get_klines_data_fast, True, True),
        (sfm.logreg.regime_multiclass, get_klines_data_large, False, False),
        (sfm.logreg.breakout_regressor_ridge, get_klines_data_large, False, False),
        (sfm.reference.lightgbm, get_klines_data_large, False, False),
        (sfm.lightgbm.tradeable_regressor, get_klines_data_large, False, False),
        # Enabling this is pushing the time from 30s to 260s
        # (sfm.lightgbm.tradeline_multiclass, get_klines_data_small, True, False),
        (sfm.rules_based.momentum_volatility_longonly,
         get_klines_data_small_fast, True, False),
        (sfm.rules_based.momentum_volatility,
         get_klines_data_small_fast, True, False),
        (sfm.ridge.ridge_classifier, get_klines_data_fast, True, False)
    ]

    for test in tests:

        try:

            uel = loop.UniversalExperimentLoop(data=test[1](),
                                               single_file_model=test[0])

            experiment_name = uuid.uuid4().hex[:8]

            if test[3]:
                manifest = test[0].manifest()
                uel.run(experiment_name=experiment_name,
                        n_permutations=2,
                        prep_each_round=test[2],
                        manifest=manifest)
            else:
                uel.run(experiment_name=experiment_name,
                        n_permutations=2,
                        prep_each_round=test[2])

            print(f'    ✅ {test[0].__name__}: PASSED')

        except Exception as e:

            print(f'    ❌ {test[0].__name__}: FAILED - {e}')

            cleanup_csv_files()
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":

    test_sfm()
