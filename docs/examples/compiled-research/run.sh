#!/bin/sh
set -eu
# Run from the downloaded bundle root, with its matching Limen wheel installed.
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
limen run --no-progress-bar \
  compiled-research/helper_indicators.yaml \
  compiled-research/helper_features.yaml \
  compiled-research/helper_scalers.yaml \
  compiled-research/conditional_features.yaml
# endregion preparation
# region search
limen run --no-progress-bar \
  compiled-research/search.yaml \
  compiled-research/search_random.yaml \
  compiled-research/search_sobol.yaml \
  compiled-research/factorized.yaml \
  compiled-research/adaptive.yaml
# endregion search
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
limen run --no-progress-bar compiled-research/calibration.yaml compiled-research/calibration_assessment.yaml
limen assess compiled-research/calibration_assessment.yaml --results results/dev/compiled-calibration-assessment
# endregion calibration
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
  compiled-research/family_direction_sizing.yaml
# endregion models
# region holdout
sh compiled-research/final/run.sh
sh compiled-research/final-event/run.sh
# endregion holdout

# region command_line
sh compiled-research/project.sh
# endregion command_line
