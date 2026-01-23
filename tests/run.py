import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.utils.cleanup import cleanup_csv_files
from tests.utils.cleanup import setup_cleanup_handlers

from tests.test_foundational_sfd import test_foundational_sfd
from tests.test_conserved_flux_renormalization import test_conserved_flux_renormalization
from tests.test_confidence_filtering_system import test_calibrate_confidence_threshold
from tests.test_confidence_filtering_system import test_apply_confidence_filtering
from tests.test_confidence_filtering_system import test_confidence_filtering_system
from tests.test_confidence_filtering_system import test_edge_cases
from tests.test_account_conviction import test_account_conviction
from tests.test_backtest_conviction import test_backtest_conviction
from tests.test_klines_data_maker_fields import test_klines_data_maker_fields
from tests.test_large_param_space import test_large_param_space
from tests.test_bars import test_volume_bars_basic
from tests.test_bars import test_trade_bars_basic
from tests.test_bars import test_liquidity_bars_basic
from tests.test_manifest_pre_split_random_selector import test_pre_split_random_selector
from tests.test_regime_diversified_opinion_pools import test_rdop

tests = [
    test_large_param_space,
    test_klines_data_maker_fields,
    test_volume_bars_basic,
    test_trade_bars_basic,
    test_liquidity_bars_basic,
    test_pre_split_random_selector,
    test_foundational_sfd,
    test_conserved_flux_renormalization,
    test_calibrate_confidence_threshold,
    test_apply_confidence_filtering,
    test_confidence_filtering_system,
    test_edge_cases,
    test_account_conviction,
    test_backtest_conviction,
    test_rdop,
]

setup_cleanup_handlers()

for test in tests:

    try:
        start_time = time.time()
        test()
        end_time = time.time()
        duration = end_time - start_time

    except Exception:
        end_time = time.time()
        duration = end_time - start_time

        cleanup_csv_files()
        traceback.print_exc()
        sys.exit(1)

cleanup_csv_files()
