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
from tests.test_param_domain import test_set_state_mismatched_keys
from tests.test_param_domain import test_set_state_observer_rollback
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
from tests.test_msq import test_remaining_count
from tests.test_msq import test_distribution
from tests.test_msq import test_inject_does_not_mutate_caller_dict
from tests.test_msq import test_n_permutations_stops_iteration
from tests.test_msq import test_mismatched_domain_raises
from tests.test_msq import test_intervention_log
from tests.test_msq import test_set_state_missing_key as test_msq_set_state_missing_key
from tests.test_msq import test_get_set_state
from tests.test_msq import test_set_filter
from tests.test_msq import test_clear_filter
from tests.test_msq import test_set_filter_replaces
from tests.test_msq import test_named_filter_descriptor_roundtrip
from tests.test_msq import test_clear_filter_removes_descriptor
from tests.test_msq import test_non_descriptor_filter_warns_on_restore
from tests.test_pruning_strategy import test_cannot_instantiate_abc
from tests.test_pruning_strategy import test_active_flag
from tests.test_feedback_controller import test_should_trigger_interval
from tests.test_feedback_controller import test_trigger_pruning_strategy
from tests.test_feedback_controller import test_trigger_inactive_pruning_strategy
from tests.test_feedback_controller import test_trigger_intra_callback
from tests.test_feedback_controller import test_apply_intervention_dispatch
from tests.test_feedback_controller import test_apply_intervention_unknown_op
from tests.test_feedback_controller import test_strategy_update_called
from tests.test_feedback_controller import test_trigger_intervention_file
from tests.test_feedback_controller import test_file_polling_skips_unchanged
from tests.test_feedback_controller import test_file_polling_detects_change
from tests.test_feedback_controller import test_file_missing_no_error
from tests.test_feedback_controller import test_file_invalid_json_no_error
from tests.test_feedback_controller import test_source_isolation
from tests.test_feedback_controller import test_pruning_strategy_isolation
from tests.test_feedback_controller import test_audit_log_written
from tests.test_feedback_controller import test_set_state_missing_key as test_fc_set_state_missing_key
from tests.test_feedback_controller import test_get_set_state as test_fc_get_set_state
from tests.test_feedback_controller import test_suggestion_not_dispatched
from tests.test_feedback_controller import test_suggestion_in_audit_log
from tests.test_feedback_controller import test_mixed_interventions_and_suggestions
from tests.test_feedback_controller import test_set_filter_dispatch
from tests.test_feedback_controller import test_clear_filter_dispatch
from tests.test_checkpoint_manager import test_checkpoint_interval_validation
from tests.test_checkpoint_manager import test_should_checkpoint_interval
from tests.test_checkpoint_manager import test_compute_content_hash_dict
from tests.test_checkpoint_manager import test_initialize_fresh_creates_directory
from tests.test_checkpoint_manager import test_save_writes_checkpoint_file
from tests.test_checkpoint_manager import test_save_metadata_content
from tests.test_checkpoint_manager import test_validate_passes_when_all_match
from tests.test_checkpoint_manager import test_validate_raises_on_hash_mismatch
from tests.test_checkpoint_manager import test_validate_raises_on_strategy_mismatch
from tests.test_checkpoint_manager import test_second_checkpoint_overwrites_first
from tests.test_checkpoint_manager import test_load_raises_on_missing_checkpoint
from tests.test_checkpoint_manager import test_load_raises_on_non_dict_checkpoint
from tests.test_checkpoint_manager import test_load_raises_on_corrupt_checkpoint
from tests.test_checkpoint_manager import test_validate_raises_on_missing_checkpoint
from tests.test_checkpoint_manager import test_validate_raises_on_corrupt_checkpoint
from tests.test_checkpoint_manager import test_validate_raises_on_non_dict_checkpoint
from tests.test_checkpoint_manager import test_validate_raises_on_missing_metadata_key
from tests.test_checkpoint_manager import test_validate_raises_on_invalid_metadata_type
from tests.test_checkpoint_manager import test_validate_raises_on_missing_experiment_round
from tests.test_checkpoint_manager import test_validate_raises_on_missing_target_permutations
from tests.test_checkpoint_manager import test_validate_raises_on_missing_msq_state
from tests.test_checkpoint_manager import test_validate_raises_on_missing_domain_state
from tests.test_checkpoint_manager import test_validate_raises_on_invalid_msq_state_type
from tests.test_checkpoint_manager import test_validate_raises_on_invalid_domain_state_type
from tests.test_checkpoint_manager import test_param_domain_set_state_invalid_leaves_state_unchanged
from tests.test_checkpoint_manager import test_param_domain_get_set_state
from tests.test_checkpoint_manager import test_uel_shutdown_flag
from tests.test_checkpoint_manager import test_uel_double_signal_raises
from tests.test_checkpoint_manager import test_uel_checkpoint_and_resume
from tests.test_experiment_core_msq import test_run_with_msq_basic_flow
from tests.test_experiment_core_msq import test_run_with_msq_context_params
from tests.test_experiment_core_msq import test_run_with_msq_feedback_trigger
from tests.test_experiment_core_msq import test_run_with_msq_checkpoint_trigger
from tests.test_experiment_core_msq import test_checkpoint_saves_feedback_and_pruning_state
from tests.test_experiment_core_msq import test_run_with_msq_shutdown_resume_full_data
from tests.test_experiment_core_msq import test_shutdown_before_any_round_completes
from tests.test_experiment_core_msq import test_resume_fails_without_round_data
from tests.test_experiment_core_msq import test_preds_none_does_not_crash
from tests.test_experiment_core_msq import test_resume_truncates_stale_rounds
from tests.test_experiment_core_msq import test_truncate_round_data_unit
from tests.test_sanity_reducer import test_nan_above_threshold
from tests.test_sanity_reducer import test_min_observations_gate
from tests.test_sanity_reducer import test_early_returns
from tests.test_sanity_reducer import test_multiple_params_pruned
from tests.test_sanity_reducer import test_dedup_across_triggers
from tests.test_sanity_reducer import test_boundary_exact_threshold
from tests.test_sanity_reducer import test_zero_metric_suggestion
from tests.test_sanity_reducer import test_zero_metric_disabled_by_default
from tests.test_sanity_reducer import test_execution_timeout_suggestion
from tests.test_sanity_reducer import test_warning_suggestion
from tests.test_sanity_reducer import test_warning_no_column
from tests.test_sanity_reducer import test_suggestion_dedup
from tests.test_sanity_reducer import test_suggestion_checkpoint_roundtrip
from tests.test_correlation_reducer import test_wrong_direction_removal
from tests.test_correlation_reducer import test_low_impact_suggestion
from tests.test_correlation_reducer import test_dedup_across_triggers as test_corr_dedup_across_triggers
from tests.test_correlation_reducer import test_early_returns as test_corr_early_returns
from tests.test_correlation_reducer import test_insufficient_data_for_correlation
from tests.test_correlation_reducer import test_state_roundtrip as test_corr_state_roundtrip
from tests.test_correlation_reducer import test_set_state_missing_key as test_corr_set_state_missing_key
from tests.test_correlation_reducer import test_maximize_false
from tests.test_correlation_reducer import test_sign_stability_below_threshold
from tests.test_saturation_reducer import test_saturated_value_filter
from tests.test_saturation_reducer import test_no_saturation
from tests.test_saturation_reducer import test_early_returns as test_sat_early_returns
from tests.test_saturation_reducer import test_window_size
from tests.test_saturation_reducer import test_zero_mean_skipped
from tests.test_saturation_reducer import test_unsaturation_clears_filter
from tests.test_saturation_reducer import test_state_roundtrip as test_sat_state_roundtrip
from tests.test_saturation_reducer import test_set_state_missing_key as test_sat_set_state_missing_key
from tests.test_saturation_reducer import test_constructor_validation as test_sat_constructor_validation
from tests.test_focus_reducer import test_breakthrough_activates_focus
from tests.test_focus_reducer import test_no_breakthrough
from tests.test_focus_reducer import test_maximize_false as test_focus_maximize_false
from tests.test_focus_reducer import test_early_returns as test_focus_early_returns
from tests.test_focus_reducer import test_snap_back_after_timeout
from tests.test_focus_reducer import test_improvement_resets_timeout
from tests.test_focus_reducer import test_numeric_narrowing_range
from tests.test_focus_reducer import test_categorical_narrowing
from tests.test_focus_reducer import test_zero_center_narrowing
from tests.test_focus_reducer import test_variation_injection_count
from tests.test_focus_reducer import test_variation_injection_linspace
from tests.test_focus_reducer import test_no_injection_on_improvement
from tests.test_focus_reducer import test_mixed_param_types as test_focus_mixed_param_types
from tests.test_focus_reducer import test_improvement_updates_combo
from tests.test_focus_reducer import test_state_roundtrip as test_focus_state_roundtrip
from tests.test_focus_reducer import test_set_state_missing_key as test_focus_set_state_missing_key
from tests.test_budget_reducer import test_constructor_validation as test_budget_constructor_validation
from tests.test_budget_reducer import test_early_returns as test_budget_early_returns
from tests.test_budget_reducer import test_trimmed_flag_prevents_retrim
from tests.test_budget_reducer import test_permutation_budget_trim_random
from tests.test_budget_reducer import test_permutation_budget_check_after_pct_gate
from tests.test_budget_reducer import test_walltime_budget_trim
from tests.test_budget_reducer import test_walltime_exceeded
from tests.test_budget_reducer import test_no_trim_when_within_budget
from tests.test_budget_reducer import test_worst_first_trim
from tests.test_budget_reducer import test_worst_first_fallback_to_random
from tests.test_budget_reducer import test_remaining_none_with_permutation_budget
from tests.test_budget_reducer import test_walltime_zero_elapsed
from tests.test_budget_reducer import test_worst_first_includes_trim_fallback
from tests.test_budget_reducer import test_both_budgets_takes_min
from tests.test_budget_reducer import test_no_budget_configured
from tests.test_budget_reducer import test_state_roundtrip as test_budget_state_roundtrip
from tests.test_budget_reducer import test_set_state_missing_key as test_budget_set_state_missing_key
from tests.test_budget_reducer import test_worst_first_maximize_false
from tests.test_reference_architecture import test_xgboost_train_returns_fitted_model
from tests.test_reference_architecture import test_xgboost_evaluate_returns_all_metric_types
from tests.test_reference_architecture import test_logreg_train_evaluate_end_to_end
from tests.test_reference_architecture import test_random_binary_train_evaluate_end_to_end
from tests.test_reference_architecture import test_tabpfn_train_evaluate_end_to_end
from tests.test_reference_architecture import test_train_with_validation_data
from tests.test_reference_architecture import test_train_without_validation_data
from tests.test_manifest_prepare_data import test_price_data_for_backtest_has_ohlc_columns
from tests.test_manifest_prepare_data import test_price_data_for_backtest_row_count_matches_test
from tests.test_manifest_prepare_data import test_price_data_for_backtest_datetime_alignment
from tests.test_manifest_prepare_data import test_override_split_config
from tests.test_manifest_prepare_data import test_override_data_source_param
from tests.test_manifest_prepare_data import test_override_multiple_params
from tests.test_manifest_prepare_data import test_override_param_not_in_original_but_in_signature
from tests.test_manifest_prepare_data import test_unknown_override_key_raises
from tests.test_inline_metrics import test_inline_and_post_experiment_metrics
from tests.test_trainer import test_trainer_end_to_end
from tests.test_trainer import test_trainer_deterministic_validation

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
    test_set_state_mismatched_keys,
    test_set_state_observer_rollback,
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
    test_remaining_count,
    test_distribution,
    test_inject_does_not_mutate_caller_dict,
    test_n_permutations_stops_iteration,
    test_mismatched_domain_raises,
    test_intervention_log,
    test_msq_set_state_missing_key,
    test_get_set_state,
    test_set_filter,
    test_clear_filter,
    test_set_filter_replaces,
    test_named_filter_descriptor_roundtrip,
    test_clear_filter_removes_descriptor,
    test_non_descriptor_filter_warns_on_restore,
    test_cannot_instantiate_abc,
    test_active_flag,
    test_should_trigger_interval,
    test_trigger_pruning_strategy,
    test_trigger_inactive_pruning_strategy,
    test_trigger_intra_callback,
    test_apply_intervention_dispatch,
    test_apply_intervention_unknown_op,
    test_strategy_update_called,
    test_trigger_intervention_file,
    test_file_polling_skips_unchanged,
    test_file_polling_detects_change,
    test_file_missing_no_error,
    test_file_invalid_json_no_error,
    test_source_isolation,
    test_pruning_strategy_isolation,
    test_audit_log_written,
    test_fc_set_state_missing_key,
    test_fc_get_set_state,
    test_suggestion_not_dispatched,
    test_suggestion_in_audit_log,
    test_mixed_interventions_and_suggestions,
    test_set_filter_dispatch,
    test_clear_filter_dispatch,
    test_checkpoint_interval_validation,
    test_should_checkpoint_interval,
    test_compute_content_hash_dict,
    test_initialize_fresh_creates_directory,
    test_save_writes_checkpoint_file,
    test_save_metadata_content,
    test_validate_passes_when_all_match,
    test_validate_raises_on_hash_mismatch,
    test_validate_raises_on_strategy_mismatch,
    test_second_checkpoint_overwrites_first,
    test_load_raises_on_missing_checkpoint,
    test_load_raises_on_non_dict_checkpoint,
    test_load_raises_on_corrupt_checkpoint,
    test_validate_raises_on_missing_checkpoint,
    test_validate_raises_on_corrupt_checkpoint,
    test_validate_raises_on_non_dict_checkpoint,
    test_validate_raises_on_missing_metadata_key,
    test_validate_raises_on_invalid_metadata_type,
    test_validate_raises_on_missing_experiment_round,
    test_validate_raises_on_missing_target_permutations,
    test_validate_raises_on_missing_msq_state,
    test_validate_raises_on_missing_domain_state,
    test_validate_raises_on_invalid_msq_state_type,
    test_validate_raises_on_invalid_domain_state_type,
    test_param_domain_set_state_invalid_leaves_state_unchanged,
    test_param_domain_get_set_state,
    test_uel_shutdown_flag,
    test_uel_double_signal_raises,
    test_uel_checkpoint_and_resume,
    test_run_with_msq_basic_flow,
    test_run_with_msq_context_params,
    test_run_with_msq_feedback_trigger,
    test_run_with_msq_checkpoint_trigger,
    test_checkpoint_saves_feedback_and_pruning_state,
    test_run_with_msq_shutdown_resume_full_data,
    test_shutdown_before_any_round_completes,
    test_resume_fails_without_round_data,
    test_preds_none_does_not_crash,
    test_resume_truncates_stale_rounds,
    test_truncate_round_data_unit,
    test_nan_above_threshold,
    test_early_returns,
    test_min_observations_gate,
    test_multiple_params_pruned,
    test_dedup_across_triggers,
    test_boundary_exact_threshold,
    test_zero_metric_suggestion,
    test_zero_metric_disabled_by_default,
    test_execution_timeout_suggestion,
    test_warning_suggestion,
    test_warning_no_column,
    test_suggestion_dedup,
    test_suggestion_checkpoint_roundtrip,
    test_wrong_direction_removal,
    test_low_impact_suggestion,
    test_corr_dedup_across_triggers,
    test_corr_early_returns,
    test_insufficient_data_for_correlation,
    test_corr_state_roundtrip,
    test_corr_set_state_missing_key,
    test_maximize_false,
    test_sign_stability_below_threshold,
    test_saturated_value_filter,
    test_no_saturation,
    test_sat_early_returns,
    test_window_size,
    test_zero_mean_skipped,
    test_unsaturation_clears_filter,
    test_sat_state_roundtrip,
    test_sat_set_state_missing_key,
    test_sat_constructor_validation,
    test_breakthrough_activates_focus,
    test_no_breakthrough,
    test_focus_maximize_false,
    test_focus_early_returns,
    test_snap_back_after_timeout,
    test_improvement_resets_timeout,
    test_numeric_narrowing_range,
    test_categorical_narrowing,
    test_zero_center_narrowing,
    test_variation_injection_count,
    test_variation_injection_linspace,
    test_no_injection_on_improvement,
    test_focus_mixed_param_types,
    test_improvement_updates_combo,
    test_focus_state_roundtrip,
    test_focus_set_state_missing_key,
    test_budget_constructor_validation,
    test_budget_early_returns,
    test_trimmed_flag_prevents_retrim,
    test_permutation_budget_trim_random,
    test_permutation_budget_check_after_pct_gate,
    test_walltime_budget_trim,
    test_walltime_exceeded,
    test_no_trim_when_within_budget,
    test_worst_first_trim,
    test_worst_first_fallback_to_random,
    test_remaining_none_with_permutation_budget,
    test_walltime_zero_elapsed,
    test_worst_first_includes_trim_fallback,
    test_both_budgets_takes_min,
    test_no_budget_configured,
    test_budget_state_roundtrip,
    test_budget_set_state_missing_key,
    test_worst_first_maximize_false,
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
    test_xgboost_train_returns_fitted_model,
    test_xgboost_evaluate_returns_all_metric_types,
    test_logreg_train_evaluate_end_to_end,
    test_random_binary_train_evaluate_end_to_end,
    test_tabpfn_train_evaluate_end_to_end,
    test_train_with_validation_data,
    test_train_without_validation_data,
    test_price_data_for_backtest_has_ohlc_columns,
    test_price_data_for_backtest_row_count_matches_test,
    test_price_data_for_backtest_datetime_alignment,
    test_override_split_config,
    test_override_data_source_param,
    test_override_multiple_params,
    test_override_param_not_in_original_but_in_signature,
    test_unknown_override_key_raises,
    test_inline_and_post_experiment_metrics,
    test_trainer_end_to_end,
    test_trainer_deterministic_validation,
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
