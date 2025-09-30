"""
Enhanced Manifest System - Complete Data Prep + Model Training Pipeline

This manifest replaces the need for separate prep() and model() functions.
Everything from data processing to model training happens through method chaining.
"""

import polars as pl
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple, Union, Optional
from loop.utils.splits import split_sequential, split_data_to_prep_output

ParamValue = Union[Any, Callable[[Dict[str, Any]], Any]]
FeatureEntry = Tuple[Callable[..., pl.LazyFrame], Dict[str, ParamValue]]
FittedParamsComputationEntry = Tuple[str, Callable[..., Any], Dict[str, ParamValue]]

FittedTransformEntry = Tuple[
    List[FittedParamsComputationEntry],
    Callable[..., pl.LazyFrame],
    Dict[str, ParamValue]
]


class ModelBuilder:
    """Builder for model training configuration."""
    
    def __init__(self, manifest: 'Manifest'):
        self.manifest = manifest
    
    def set_model_class(self, model_class, **init_params) -> 'ModelBuilder':
        """Set the model class and initialization parameters."""
        self.manifest.model_class = model_class
        self.manifest.model_init_params = init_params
        return self
    
    def set_training_params(self, **params) -> 'ModelBuilder':
        """Set training-specific parameters (e.g., num_boost_round for LightGBM)."""
        self.manifest.training_params = params
        return self
    
    def set_sample_weights(self, weight_func: Callable) -> 'ModelBuilder':
        """Set function to compute sample weights from training data."""
        self.manifest.sample_weight_func = weight_func
        return self
    
    def set_prediction_transform(self, transform_func: Callable) -> 'ModelBuilder':
        """Set function to transform raw predictions (e.g., thresholding, argmax)."""
        self.manifest.prediction_transform = transform_func
        return self
    
    def set_calibration(self, method: str = 'isotonic', cv: int = 3) -> 'ModelBuilder':
        """Enable probability calibration."""
        self.manifest.calibration_config = {
            'enabled': True,
            'method': method,
            'cv': cv
        }
        return self
    
    def set_metrics_function(self, metrics_func: Callable) -> 'ModelBuilder':
        """Set function to compute evaluation metrics."""
        self.manifest.metrics_func = metrics_func
        return self
    
    def done(self) -> 'Manifest':
        """Complete model configuration and return to main manifest."""
        return self.manifest


class TargetBuilder:
    """Helper class for building target transformations with context."""

    def __init__(self, manifest: 'Manifest', target_column: str):
        self.manifest = manifest
        self.target_column = target_column
        self.manifest.target_column = target_column

    def add_fitted_transform(self, func: Callable) -> 'FittedTransformBuilder':
        return FittedTransformBuilder(self.manifest, func)

    def add_transform(self, func: Callable, **params) -> 'TargetBuilder':
        entry = ([], func, params)
        self.manifest.target_transforms.append(entry)
        return self

    def done(self) -> 'Manifest':
        return self.manifest


class FittedTransformBuilder:
    """Helper class for building fitted transforms with parameter fitting."""

    def __init__(self, manifest: 'Manifest', func: Callable):
        self.manifest = manifest
        self.func = func
        self.fitted_params: List[FittedParamsComputationEntry] = []

    def fit_param(self, name: str, compute_func: Callable, **params) -> 'FittedTransformBuilder':
        self.fitted_params.append((name, compute_func, params))
        return self

    def with_params(self, **params) -> 'TargetBuilder':
        entry = (self.fitted_params, self.func, params)
        self.manifest.target_transforms.append(entry)
        return TargetBuilder(self.manifest, self.manifest.target_column)


@dataclass
class Manifest:
    """Complete pipeline specification from raw data to trained model."""

    # Data processing configuration
    bar_formation: FeatureEntry = None
    feature_transforms: List[FeatureEntry] = field(default_factory=list)
    target_transforms: List[FittedTransformEntry] = field(default_factory=list)
    scaler: FittedTransformEntry = None
    required_bar_columns: List[str] = field(default_factory=list)
    target_column: str = None
    split_config: Tuple[int, int, int] = (8, 1, 2)
    
    # Model configuration
    model_class: Any = None
    model_init_params: Dict[str, ParamValue] = field(default_factory=dict)
    training_params: Dict[str, ParamValue] = field(default_factory=dict)
    sample_weight_func: Optional[Callable] = None
    prediction_transform: Optional[Callable] = None
    calibration_config: Dict[str, Any] = field(default_factory=dict)
    metrics_func: Optional[Callable] = None
    
    # Storage for intermediate state
    _fitted_params: Dict[str, Any] = field(default_factory=dict)
    _trained_model: Any = None

    def _add_transform(self, func: Callable, **params) -> 'Manifest':
        self.feature_transforms.append((func, params))
        return self

    def add_feature(self, func: Callable, **params) -> 'Manifest':
        """Add a feature transformation to the manifest."""
        return self._add_transform(func, **params)

    def add_indicator(self, func: Callable, **params) -> 'Manifest':
        """Add an indicator transformation to the manifest."""
        return self._add_transform(func, **params)

    def set_bar_formation(self, func: Callable, **params) -> 'Manifest':
        """Set bar formation function and parameters."""
        self.bar_formation = (func, params)
        return self

    def set_required_bar_columns(self, columns: List[str]) -> 'Manifest':
        """Set required columns after bar formation."""
        self.required_bar_columns = columns
        return self

    def set_split_config(self, train: int, val: int, test: int) -> 'Manifest':
        """Set data split configuration."""
        self.split_config = (train, val, test)
        return self

    def set_scaler(self, transform_class, param_name: str = '_scaler') -> 'Manifest':
        """Set scaler transformation using make_fitted_scaler."""
        self.scaler = make_fitted_scaler(param_name, transform_class)
        return self

    def with_target(self, target_column: str) -> TargetBuilder:
        """Start building target transformations with context."""
        return TargetBuilder(self, target_column)
    
    def with_model(self) -> ModelBuilder:
        """Start building model configuration."""
        return ModelBuilder(self)

    def execute(
        self,
        raw_data: pl.DataFrame,
        round_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute the complete pipeline: data prep → model training → predictions.
        
        Args:
            raw_data: Raw input dataset
            round_params: Parameter values for current round
            
        Returns:
            Dictionary containing all results including metrics, predictions, models, _alignment
        """
        
        # Data Preparation
        split_data = split_sequential(raw_data, self.split_config)
        datetime_bar_pairs = [_process_bars(self, split, round_params) for split in split_data]
        all_datetimes = [dt for datetimes, _ in datetime_bar_pairs for dt in datetimes]
        split_data = [bar_data for _, bar_data in datetime_bar_pairs]

        all_fitted_params = {}

        for i in range(len(split_data)):
            lazy_data = split_data[i].lazy()

            lazy_data = _apply_feature_transforms(self, lazy_data, round_params)

            data = lazy_data.collect()
            data, all_fitted_params = _apply_target_transforms(
                self, data, round_params, all_fitted_params, is_training=(i == 0)
            )
            
            data, all_fitted_params = _apply_scaler(
                self, data, round_params, all_fitted_params, is_training=(i == 0)
            )
            
            split_data[i] = data.drop_nulls()

        for i, split_df in enumerate(split_data):
            assert 'datetime' in split_df.columns, f"Split {i} missing 'datetime' column"

        if self.target_column:
            for i, split_df in enumerate(split_data):
                cols = list(split_df.columns)
                if self.target_column in cols:
                    cols.remove(self.target_column)
                    cols.append(self.target_column)
                    split_data[i] = split_df.select(cols)
                else:
                    raise ValueError(f"Split {i} missing target column '{self.target_column}'")

        cols = list(split_data[0].columns)

        data_dict = split_data_to_prep_output(split_data, cols, all_datetimes)
        
        # Add fitted parameters to data_dict
        for param_name, param_value in all_fitted_params.items():
            data_dict[param_name] = param_value

        ## TODO: Make into loop.sfm.model
        # Model Training
        if self.model_class is not None:
            round_results = self._train_and_evaluate(data_dict, round_params)
            # Merge the data_dict into round_results for logging compatibility
            # This ensures y_test, x_test, etc. are available for the logging system
            merged_results = {**data_dict, **round_results}
            return merged_results
        else:
            # Return just the prepared data if no model configured
            return data_dict

    def _train_and_evaluate(
        self, 
        data_dict: Dict[str, Any], 
        round_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Train model and compute predictions/metrics.
        
        Args:
            data_dict: Prepared data from data preparation phase
            round_params: Parameter values for current round
            
        Returns:
            Dictionary containing metrics, predictions, models, extras
        """
        ## TODO: Make into loop.sfm.model
        # Resolve model parameters
        resolved_init = _resolve_params(self.model_init_params, round_params)
        resolved_training = _resolve_params(self.training_params, round_params)
        
        # Extract data
        X_train = data_dict['x_train']
        y_train = data_dict['y_train']
        X_val = data_dict['x_val']
        y_val = data_dict['y_val']
        X_test = data_dict['x_test']
        y_test = data_dict['y_test']
        
        # Compute sample weights if configured
        sample_weights = None
        if self.sample_weight_func is not None:
            # Pass the full training split for weight computation
            train_df = data_dict.get('_train_clean')
            if train_df is not None:
                sample_weights = self.sample_weight_func(train_df, round_params)
        
        # Initialize and train model
        model = self.model_class(**resolved_init)
        
        # Handle different training interfaces
        if hasattr(model, 'fit'):
            if sample_weights is not None:
                model.fit(X_train, y_train, sample_weight=sample_weights)
            else:
                model.fit(X_train, y_train)
        else:
            # For LightGBM-style training
            import lightgbm as lgb
            train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            evals_result = {}
            model = lgb.train(
                params=resolved_init,
                train_set=train_data,
                valid_sets=[train_data, val_data],
                valid_names=['train', 'val'],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=30, verbose=False),
                    lgb.record_evaluation(evals_result)
                ],
                **resolved_training
            )
        
        # Apply calibration if configured
        if self.calibration_config.get('enabled', False):
            from sklearn.calibration import CalibratedClassifierCV
            calibrator = CalibratedClassifierCV(
                model,
                method=self.calibration_config['method'],
                cv=self.calibration_config.get('cv', 3)
            )
            calibrator.fit(X_val, y_val)
            model = calibrator
        
        # Generate predictions
        if hasattr(model, 'predict_proba'):
            y_proba_raw = model.predict_proba(X_test)
            # For binary classification, extract the positive class probabilities
            if len(y_proba_raw.shape) == 2 and y_proba_raw.shape[1] == 2:
                y_proba = y_proba_raw[:, 1]  # Take positive class probabilities
            else:
                y_proba = y_proba_raw
        elif hasattr(model, 'predict'):
            raw_pred = model.predict(X_test)
            if len(raw_pred.shape) == 2:
                y_proba = raw_pred[:, 1] if raw_pred.shape[1] == 2 else raw_pred
            else:
                y_proba = raw_pred
        else:
            y_proba = None
        
        # Transform predictions if configured
        if self.prediction_transform is not None:
            y_pred = self.prediction_transform(y_proba, round_params)
        else:
            if y_proba is not None:
                # For binary classification, threshold the probabilities
                y_pred = (y_proba >= 0.5).astype(np.int8)
            else:
                y_pred = model.predict(X_test)
        
        # Compute metrics
        if self.metrics_func is not None:
            round_results = self.metrics_func(data_dict, y_pred, y_proba)
        else:
            # Default: try to auto-detect metrics type
            from loop.metrics.binary_metrics import binary_metrics
            round_results = binary_metrics(data_dict, y_pred, y_proba)
        
        # Store artifacts
        round_results['models'] = [model]
        round_results['_preds'] = y_pred
        
        self._trained_model = model
        
        return round_results


def _apply_fitted_transform(data: pl.DataFrame, fitted_transform):
    """Apply transformation using fitted transform instance."""
    return fitted_transform.transform(data)


def make_fitted_scaler(param_name: str, transform_class):
    """Create fitted transform entry for scaling."""
    return ([
        (param_name, lambda data: transform_class(data), {})
    ],
    _apply_fitted_transform, {
        'fitted_transform': param_name
    })


def _resolve_params(params: Dict[str, Any], round_params: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve parameters using round_params."""
    resolved = {}
    for key, value in params.items():
        if isinstance(value, str):
            if value.startswith('_') or value in round_params:
                resolved[key] = round_params[value]
            elif '{' in value and '}' in value:
                resolved[key] = value.format(**round_params)
            else:
                resolved[key] = value
        else:
            resolved[key] = value
    return resolved


def _process_bars(
    manifest: Manifest,
    data: pl.DataFrame,
    round_params: Dict[str, Any]
) -> Tuple[List, pl.DataFrame]:
    """Process bar formation on data."""
    if manifest.bar_formation and round_params.get('bar_type', 'base') != 'base':
        func, base_params = manifest.bar_formation
        resolved = _resolve_params(base_params, round_params)
        lazy_data = data.lazy().pipe(func, **resolved)
        bar_data = lazy_data.collect()
        all_datetimes = bar_data['datetime'].to_list()
    else:
        all_datetimes = data['datetime'].to_list()
        bar_data = data

    available_cols = list(bar_data.columns)
    for required_col in manifest.required_bar_columns:
        assert required_col in available_cols, (
            f"Required bar column '{required_col}' not found after bar formation"
        )

    return all_datetimes, bar_data


def _apply_feature_transforms(manifest: Manifest, lazy_data, round_params: Dict[str, Any]):
    """Apply all feature transformations."""
    for func, base_params in manifest.feature_transforms:
        resolved = _resolve_params(base_params, round_params)
        lazy_data = lazy_data.pipe(func, **resolved)
    return lazy_data


def _apply_fitted_transforms(
    transform_entries: List[FittedTransformEntry],
    data: pl.DataFrame,
    round_params: Dict[str, Any],
    all_fitted_params: Dict[str, Any],
    is_training: bool
) -> Tuple[pl.DataFrame, Dict[str, Any]]:
    """Apply fitted transforms."""
    for fitted_param_computations, func, base_params in transform_entries:
        for param_name, compute_func, compute_base_params in fitted_param_computations:
            if param_name not in all_fitted_params and is_training:
                resolved = _resolve_params(compute_base_params, round_params)
                value = compute_func(data, **resolved)
                all_fitted_params[param_name] = value

        combined_round_params = {**round_params, **all_fitted_params}
        resolved = _resolve_params(base_params, combined_round_params)
        data = func(data, **resolved)

    return data, all_fitted_params


def _apply_target_transforms(
    manifest: Manifest,
    data: pl.DataFrame,
    round_params: Dict[str, Any],
    all_fitted_params: Dict[str, Any],
    is_training: bool
) -> Tuple[pl.DataFrame, Dict[str, Any]]:
    """Apply target transformations."""
    enhanced_round_params = round_params.copy()
    if manifest.target_column:
        enhanced_round_params['target_column'] = manifest.target_column

    return _apply_fitted_transforms(
        manifest.target_transforms, data, enhanced_round_params,
        all_fitted_params, is_training
    )


def _apply_scaler(
    manifest: Manifest,
    data: pl.DataFrame,
    round_params: Dict[str, Any],
    all_fitted_params: Dict[str, Any],
    is_training: bool
) -> Tuple[pl.DataFrame, Dict[str, Any]]:
    """Apply scaling transformation."""
    if manifest.scaler:
        return _apply_fitted_transforms(
            [manifest.scaler], data, round_params,
            all_fitted_params, is_training
        )
    return data, all_fitted_params