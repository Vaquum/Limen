import uuid
import sys
import loop
import traceback
import polars as pl

from loop import sfm

from loop.tests.utils.cleanup import cleanup_csv_files
from loop.tests.utils.get_data import get_klines_data, get_trades_data


def test_sfm():
    
    tests = [
        (sfm.reference.random, get_klines_data, True),
        (sfm.reference.xgboost, get_klines_data, False),
        (sfm.reference.logreg, get_klines_data, True),
        (sfm.logreg.regime_multiclass, get_klines_data, False),
        (sfm.logreg.breakout_regressor_ridge, get_klines_data, False),
        (sfm.reference.lightgbm, get_trades_data, False),
        (sfm.lightgbm.regime_multiclass, get_klines_data, False),
        (sfm.lightgbm.breakout_regressor, get_klines_data, False),
        ]

    for test in tests:
        print(f"Starting test iteration for {test[0].__name__}")
        try:
            data = pl.read_csv('datasets/klines_2h_2020_2025.csv').with_columns(
                pl.col('datetime').str.to_datetime()
            )
            print("data loaded")
            uel = loop.UniversalExperimentLoop(data=data,
                                                single_file_model=test[0])
            uel.run(experiment_name=uuid.uuid4().hex[:8],
                    n_permutations=2,
                    prep_each_round=test[2])
            print(f'    ✅ {test[0].__name__}: PASSED')
        
        except Exception as e:
            print(f'    ❌ {test[0].__name__}: FAILED - {e}')
            cleanup_csv_files()
            traceback.print_exc()
            sys.exit(1)