#!/bin/sh
set -eu
# Run from the downloaded bundle root, with its matching Limen wheel installed.
# Diagnostics and lifecycle use disjoint output directories.
diagnostics() {
# region source_formats
limen run --no-progress-bar \
  compiled-research/source_csv.yaml \
  compiled-research/source_zip.yaml \
  compiled-research/source_arrow.yaml
# endregion source_formats
# region helpers
limen run --no-progress-bar \
  compiled-research/helper_indicators.yaml \
  compiled-research/helper_features.yaml \
  compiled-research/helper_regimes.yaml \
  compiled-research/helper_scalers.yaml \
  compiled-research/conditional_features.yaml
# endregion helpers
# region targets
limen run --no-progress-bar \
  compiled-research/target_emabreakout.yaml \
  compiled-research/target_forwardbreakout.yaml \
  compiled-research/target_identity.yaml \
  compiled-research/target_nextbardown.yaml \
  compiled-research/target_nextbarup.yaml \
  compiled-research/target_nextreturn.yaml \
  compiled-research/target_quantilebinary.yaml \
  compiled-research/target_randombinary.yaml \
  compiled-research/target_thresholdbinary.yaml \
  compiled-research/target_tradelinelongbinary.yaml \
  compiled-research/target_triplebarrier.yaml \
  compiled-research/target_exitquality.yaml \
  compiled-research/target_riskrewardratio.yaml
sh compiled-research/target_contracts.sh
# endregion targets
# region reducers
limen run --no-progress-bar \
  compiled-research/reducer_sanity.yaml \
  compiled-research/reducer_saturation.yaml \
  compiled-research/reducer_focus.yaml \
  compiled-research/reducer_correlation.yaml \
  compiled-research/analysis.yaml \
  compiled-research/rule_contracts.yaml
# endregion reducers
# region recording
limen run --no-progress-bar \
  compiled-research/recording_lean.yaml \
  compiled-research/recording_full.yaml \
  compiled-research/recording_outputs.yaml
# endregion recording
# region execution
limen run --no-progress-bar \
  compiled-research/execution.yaml \
  compiled-research/funded.yaml
# endregion execution
# region bars
limen run --no-progress-bar compiled-research/bars.yaml
# endregion bars
# region allocation
limen run --no-progress-bar \
  compiled-research/allocation.yaml \
  compiled-research/allocation_ml.yaml \
  compiled-research/allocation_funded.yaml
# endregion allocation
# region models
limen run --no-progress-bar \
  compiled-research/common_strategies.yaml \
  compiled-research/family_ridge_regressor.yaml \
  compiled-research/family_dlinear_regressor.yaml \
  compiled-research/family_xgboost_regressor.yaml \
  compiled-research/family_random_binary.yaml \
  compiled-research/family_direction_sizing.yaml \
  compiled-research/family_rbf_svm_binary.yaml \
  compiled-research/family_gaussian_hmm_binary.yaml \
  compiled-research/family_lstm_binary.yaml \
  compiled-research/family_residual_regressor.yaml \
  compiled-research/family_cnn1d_binary.yaml \
  compiled-research/family_pseudo_label_binary.yaml
# endregion models
}
diagnostics &
diagnostics_pid=$!
sh compiled-research/project.sh &
lifecycle_pid=$!
trap 'wait "$diagnostics_pid" "$lifecycle_pid"' 0
# region selection
limen validate compiled-research/selection.yaml
limen profile compiled-research/selection.yaml
limen run --dry-run compiled-research/selection.yaml
limen run --no-progress-bar compiled-research/selection.yaml
limen assess compiled-research/selection.yaml --results results/dev/compiled-cli
# endregion selection
# region preparation
limen run --no-progress-bar compiled-research/preparation.yaml
limen run --no-progress-bar compiled-research/research.yaml
limen assess compiled-research/research.yaml --results results/dev/compiled-research
# endregion preparation
# region search
limen run --no-progress-bar \
  compiled-research/search.yaml \
  compiled-research/search_random.yaml \
  compiled-research/search_sobol.yaml \
  compiled-research/factorized.yaml \
  compiled-research/adaptive.yaml
# endregion search
# region parallel
limen run --no-progress-bar --workers 2 compiled-research/parallel.yaml
limen run --no-progress-bar --resume results/dev/compiled-parallel --workers 2
# endregion parallel
# region recovery
limen run --no-progress-bar compiled-research/recovery.yaml
limen run --no-progress-bar --resume results/dev/compiled-recovery
# endregion recovery
# region walk_forward
limen run --no-progress-bar compiled-research/walk_forward.yaml
limen assess compiled-research/walk_forward.yaml --results results/dev/compiled-walk_forward
limen run --no-progress-bar compiled-research/acceptance.yaml
# endregion walk_forward
# region calibration
limen run --no-progress-bar compiled-research/calibration.yaml compiled-research/calibration_assessment.yaml compiled-research/calibration_isotonic.yaml
limen assess compiled-research/calibration_assessment.yaml --results results/dev/compiled-calibration-assessment
limen assess compiled-research/calibration_isotonic.yaml --results results/dev/compiled-calibration-isotonic
# endregion calibration
# region holdout
sh compiled-research/final/run.sh
sh compiled-research/final-event/run.sh
# endregion holdout

# region command_line
wait "$diagnostics_pid"
wait "$lifecycle_pid"
trap - 0
# endregion command_line
