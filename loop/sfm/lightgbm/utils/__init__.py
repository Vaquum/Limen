# Make utils a Python package
from loop.sfm.lightgbm.utils.regime_multiclass import (
    build_sample_dataset_for_regime_multiclass,
    add_features_to_regime_multiclass_dataset
)

from loop.sfm.lightgbm.utils.mega_model_data_sampler import (
    MegaModelDataSampler,
    run_mega_model_experiment,
    run_enhanced_megamodel_with_uel,
    integrate_enhanced_megamodel_into_workflow,
    get_best_model_from_results,
    predict_with_best_model,
    save_experiment_results
)
from loop.sfm.lightgbm.utils.create_megamodel_predictions import create_megamodel_predictions

from loop.sfm.lightgbm.utils.quantile_model_with_confidence import quantile_model_with_confidence
from loop.sfm.lightgbm.utils.moving_average_correction_model import moving_average_correction_model
from loop.sfm.lightgbm.utils.regime_stability import (
    add_stability_features,
    get_stability_features
)   

from loop.sfm.lightgbm.utils.breakout_regressor import (
    build_sample_dataset_for_breakout_regressor,
    extract_xy,
    extract_xy_polars
)

__all__ = [
    'MegaModelDataSampler',
    'run_mega_model_experiment',
    'run_enhanced_megamodel_with_uel',
    'integrate_enhanced_megamodel_into_workflow',
    'get_best_model_from_results',
    'predict_with_best_model',
    'save_experiment_results',
    'build_sample_dataset_for_regime_multiclass',
    'add_features_to_regime_multiclass_dataset',
    'create_megamodel_predictions',
    'quantile_model_with_confidence',
    'moving_average_correction_model',
    'build_sample_dataset_for_breakout_regressor',
    'extract_xy',
    'extract_xy_polars',
    'add_stability_features',
    'get_stability_features'
]
