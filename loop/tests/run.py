import sys
import time
import traceback

from loop.tests.utils.cleanup import cleanup_csv_files, setup_cleanup_handlers

from loop.tests.test_sfm import test_sfm
from loop.tests.test_conserved_flux_renormalization import test_conserved_flux_renormalization
from loop.tests.test_confidence_filtering_system import test_calibrate_confidence_threshold
from loop.tests.test_confidence_filtering_system import test_apply_confidence_filtering
from loop.tests.test_confidence_filtering_system import test_confidence_filtering_system
from loop.tests.test_confidence_filtering_system import test_edge_cases
from loop.tests.test_moving_average_correction_model import test_moving_average_correction
from loop.tests.test_account_conviction import test_account_conviction
from loop.tests.test_backtest_conviction import test_backtest_conviction
from loop.tests.test_klines_data_maker_fields import test_klines_data_maker_fields
from loop.tests.test_polars_lazy_evaluation import (
    test_polars_lazy_evaluation_correctness,
    test_polars_lazy_evaluation_performance,
)
from loop.tests.test_large_param_space import test_large_param_space
from loop.tests.test_bars import test_volume_bars_basic, test_trade_bars_basic, test_liquidity_bars_basic

tests = [
    test_large_param_space,
    test_klines_data_maker_fields,
    test_volume_bars_basic,
    test_trade_bars_basic,
    test_liquidity_bars_basic,
    test_sfm,
    test_conserved_flux_renormalization,
    test_calibrate_confidence_threshold,
    test_apply_confidence_filtering,
    test_confidence_filtering_system,
    test_edge_cases,
    # test_moving_average_correction,
    test_account_conviction,
    test_backtest_conviction,
    test_polars_lazy_evaluation_correctness,
    test_polars_lazy_evaluation_performance,
]

setup_cleanup_handlers()

for test in tests:

    try:
        start_time = time.time()
        test()
        end_time = time.time()
        duration = end_time - start_time
        print(f'    ✅ {test.__name__}: PASSED ({duration:.3f}s)')

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f'    ❌ {test.__name__}: FAILED ({duration:.3f}s) - {e}')

        cleanup_csv_files()
        traceback.print_exc()
        sys.exit(1)

cleanup_csv_files()
