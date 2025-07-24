import sys

from loop.tests.utils.cleanup import cleanup_csv_files, setup_cleanup_handlers

from test_sfm import test_sfm
from test_mega_model import test_mega_model_with_live_labeling
from test_create_megamodel_predictions import test_create_megamodel_predictions
from test_confidence_filtering_system import test_calibrate_confidence_threshold
from test_confidence_filtering_system import test_apply_confidence_filtering
from test_confidence_filtering_system import test_confidence_filtering_system
from test_confidence_filtering_system import test_edge_cases
from test_quantile_model import test_quantile_model
from test_moving_average_correction_model import test_moving_average_correction
from test_regime_stability import test_regime_stability
from test_account_conviction import test_account_conviction
from test_backtest_conviction import test_backtest_conviction

tests = [
    test_sfm,
    test_mega_model_with_live_labeling,
    test_create_megamodel_predictions,
    test_calibrate_confidence_threshold,
    test_apply_confidence_filtering,
    test_confidence_filtering_system,
    test_edge_cases,
    test_quantile_model,
    test_moving_average_correction,
    test_regime_stability,
    test_account_conviction,
    test_backtest_conviction,
]

setup_cleanup_handlers()

for test in tests:
    
    try:
        test()
        print(f'    ✅ {test.__name__}: PASSED')
    
    except Exception as e:
        print(f'    ❌ {test.__name__}: FAILED - {e}')
        cleanup_csv_files()
        sys.exit(1)

cleanup_csv_files()