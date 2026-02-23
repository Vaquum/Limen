import sys
import time
import traceback
import logging
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
from tests.test_param_domain import test_param_domain_init
from tests.test_param_domain import test_param_domain_init_validation
from tests.test_param_domain import test_param_domain_defensive_copy
from tests.test_param_domain import test_remove_value
from tests.test_param_domain import test_remove_value_prevents_empty
from tests.test_param_domain import test_remove_values_ge
from tests.test_param_domain import test_remove_values_ge_prevents_empty
from tests.test_param_domain import test_remove_values_le
from tests.test_param_domain import test_keep_values
from tests.test_param_domain import test_keep_values_no_overlap
from tests.test_param_domain import test_keep_between
from tests.test_param_domain import test_inject_value
from tests.test_param_domain import test_observer_notification
from tests.test_param_domain import test_observer_version_increments
from tests.test_param_domain import test_is_valid_combination
from tests.test_param_domain import test_comparison_type_error
from tests.test_param_domain import test_total_combinations_updates
from tests.test_msq import test_msq_basic_iteration
from tests.test_msq import test_msq_yielded_count
from tests.test_msq import test_priority_queue
from tests.test_msq import test_inject_validates_keys
from tests.test_msq import test_intervention_routing
from tests.test_msq import test_remove_custom
from tests.test_msq import test_filter_exhausted_error
from tests.test_msq import test_trim
from tests.test_msq import test_trim_with_priority_queue
from tests.test_msq import test_remaining_count_known
from tests.test_msq import test_remaining_count_unknown
from tests.test_msq import test_distribution
from tests.test_msq import test_intervention_log
from tests.test_msq import test_get_set_state

tests = [
    test_param_domain_init,
    test_param_domain_init_validation,
    test_param_domain_defensive_copy,
    test_remove_value,
    test_remove_value_prevents_empty,
    test_remove_values_ge,
    test_remove_values_ge_prevents_empty,
    test_remove_values_le,
    test_keep_values,
    test_keep_values_no_overlap,
    test_keep_between,
    test_inject_value,
    test_observer_notification,
    test_observer_version_increments,
    test_is_valid_combination,
    test_comparison_type_error,
    test_total_combinations_updates,
    test_msq_basic_iteration,
    test_msq_yielded_count,
    test_priority_queue,
    test_inject_validates_keys,
    test_intervention_routing,
    test_remove_custom,
    test_filter_exhausted_error,
    test_trim,
    test_trim_with_priority_queue,
    test_remaining_count_known,
    test_remaining_count_unknown,
    test_distribution,
    test_intervention_log,
    test_get_set_state,
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

setup_cleanup_handlers()

for test in tests:

    try:
        start_time = time.time()
        test()
        end_time = time.time()
        duration = end_time - start_time
        logger.info('✅ %s: PASSED (%.3fs)', test.__name__, duration)

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time

        logger.error('❌ %s: FAILED (%.3fs) - %s', test.__name__, duration, str(e))
        cleanup_csv_files()
        traceback.print_exc()
        sys.exit(1)

cleanup_csv_files()
