import loop
import uuid

import loop.sfm as sfm
from loop.tests.utils.cleanup import cleanup_csv_files, setup_cleanup_handlers
from loop.tests.utils.get_data import get_klines_data, get_trades_data

from mega_model_test import test_mega_model_with_live_labeling

from test_create_megamodel_predictions import test_create_megamodel_predictions
from test_confidence_filtering_system import (
    test_calibrate_confidence_threshold,
    test_apply_confidence_filtering, 
    test_confidence_filtering_system,
    test_edge_cases
)

from test_quantile_model import test_quantile_model
from test_moving_average_correction_model import test_moving_average_correction
from test_regime_stability import test_regime_stability

from test_account_conviction import test_account_conviction
from test_backtest_conviction import test_backtest_conviction


setup_cleanup_handlers()

try:
    try:
        test_account_conviction()
        print(f'    ✅ {test_account_conviction.__name__}: PASSED')
    except Exception as e:
        print(f'    ❌ {test_account_conviction.__name__}: FAILED - {e}')

    try:
        test_backtest_conviction()
        print(f'    ✅ {test_backtest_conviction.__name__}: PASSED')
    except Exception as e:
        print(f'    ❌ {test_backtest_conviction.__name__}: FAILED - {e}')

    try:
        mega_results = test_mega_model_with_live_labeling()
        print(f'    ✅ {test_mega_model_with_live_labeling.__name__}: PASSED')
    except Exception as e:
        print(f'    ❌ {test_mega_model_with_live_labeling.__name__}: FAILED - {e}')

    try:
        test_create_megamodel_predictions()
        print(f'    ✅ {test_create_megamodel_predictions.__name__}: PASSED')
    except Exception as e:
        print(f'    ❌ {test_create_megamodel_predictions.__name__}: FAILED - {e}')

    confidence_tests = [
        ('calibrate_confidence_threshold', test_calibrate_confidence_threshold),
        ('apply_confidence_filtering', test_apply_confidence_filtering),
        ('confidence_filtering_system', test_confidence_filtering_system),
        ('edge_cases', test_edge_cases)
    ]

    for test_name, test_func in confidence_tests:
        try:
            test_func()
            print(f'    ✅ {test_func.__name__}: PASSED')
        except Exception as e:
            print(f'    ❌ {test_func.__name__}: FAILED - {e}')

    try:
        test_quantile_model()
        print(f'    ✅ {test_quantile_model.__name__}: PASSED')
    except Exception as e:
        print(f'    ❌ {test_quantile_model.__name__}: FAILED - {e}')

    try:
        test_moving_average_correction()
        print(f'    ✅ {test_moving_average_correction.__name__}: PASSED')
    except Exception as e:
        print(f'    ❌ {test_moving_average_correction.__name__}: FAILED - {e}')

    try:
        test_regime_stability()
        print(f'    ✅ {test_regime_stability.__name__}: PASSED')
    except Exception as e:
        print(f'    ❌ {test_regime_stability.__name__}: FAILED - {e}')

    from loop.reports.log_df import read_from_file, outcome_df, corr_df
    data = read_from_file('datasets/logreg_broad_2_3600.csv')
    outcome_df = outcome_df(data, ['solver', 'feature_to_drop', 'penalty'], type='categorical')
    corr_df = corr_df(outcome_df)

    tests = [(sfm.random, get_klines_data, True),
             (sfm.xgboost, get_klines_data, False),
             (sfm.lightgbm_reference, get_trades_data, False),
             (sfm.logreg_reference, get_klines_data, True),
             (sfm.logreg.regime_multiclass, get_klines_data, True),
             (sfm.logreg.breakout_regressor_ridge, get_klines_data, True)]

    for i, test in enumerate(tests, 1):
        print(f'\n  B2.{i} Running {test[0].__name__} with {test[1].__name__}')
        test_name = uuid.uuid4().hex[:8]
        
        try:
            uel = loop.UniversalExperimentLoop(data=test[1](),
                                             single_file_model=test[0])
            uel.run(experiment_name=test_name,
                   n_permutations=2,
                   prep_each_round=test[2])
            print(f'    ✅ {test[0].__name__}: PASSED')
        except Exception as e:
            print(f'    ❌ {test[0].__name__}: FAILED - {e}')

finally:
    cleanup_csv_files()