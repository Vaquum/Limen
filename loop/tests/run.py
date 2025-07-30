import sys
import traceback

from loop.tests.utils.cleanup import cleanup_csv_files, setup_cleanup_handlers

from loop.tests.test_sfm import test_sfm
from loop.tests.test_mega_model import test_mega_model_with_live_labeling
from loop.tests.test_create_megamodel_predictions import test_create_megamodel_predictions
from loop.tests.test_confidence_filtering_system import test_calibrate_confidence_threshold
from loop.tests.test_confidence_filtering_system import test_apply_confidence_filtering
from loop.tests.test_confidence_filtering_system import test_confidence_filtering_system
from loop.tests.test_confidence_filtering_system import test_edge_cases
from loop.tests.test_quantile_model import test_quantile_model
from loop.tests.test_moving_average_correction_model import test_moving_average_correction
from loop.tests.test_regime_stability import test_regime_stability
from loop.tests.test_account_conviction import test_account_conviction
from loop.tests.test_backtest_conviction import test_backtest_conviction

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
        traceback.print_exc()
        sys.exit(1)

cleanup_csv_files()