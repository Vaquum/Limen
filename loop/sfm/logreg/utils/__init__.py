# Make utils a Python package

from loop.sfm.logreg.utils.regime_multiclass import (
    build_sample_dataset_for_regime_multiclass,
    add_features_to_regime_multiclass_dataset
)

__all__ = [
    'build_sample_dataset_for_regime_multiclass',
    'add_features_to_regime_multiclass_dataset'
]