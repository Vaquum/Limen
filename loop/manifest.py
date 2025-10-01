import polars as pl

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple, Union
from loop.utils.splits import split_data_to_prep_output
from loop.utils.splits import split_sequential

ParamValue = Union[Any, Callable[[Dict[str, Any]], Any]]
FeatureEntry = Tuple[Callable[..., pl.LazyFrame], Dict[str, ParamValue]]

FittedParamsComputationEntry = Tuple[str, Callable[..., Any], Dict[str, ParamValue]]

FittedTransformEntry = Tuple[
    List[FittedParamsComputationEntry],
    Callable[..., pl.LazyFrame],
    Dict[str, ParamValue]
]


class ModelBuilder:
    '''Helper class for building model configuration.'''

    def __init__(self, manifest: 'Manifest'):
        self.manifest = manifest

    def set_model_function(self, func: Callable, **params) -> 'ModelBuilder':
        '''
        Set the model training function.
        
        Args:
            func (Callable): Model function that takes (data, **params) and returns dict
            **params: Parameter mappings for the model function
            
        Returns:
            ModelBuilder: Self for method chaining
        '''
        self.manifest.model_function = func
        self.manifest.model_params = params
        return self

    def set_metrics_function(self, func: Callable, **params) -> 'ModelBuilder':
        '''
        Set the metrics computation function.
        
        Args:
            func (Callable): Metrics function that takes (data, y_pred, y_proba, **params)
            **params: Parameter mappings for the metrics function
            
        Returns:
            ModelBuilder: Self for method chaining
        '''
        self.manifest.metrics_function = func
        self.manifest.metrics_params = params
        return self

    def done(self) -> 'Manifest':
        '''Return to manifest for further configuration.'''
        return self.manifest


class TargetBuilder:
    '''Helper class for building target transformations with context.'''

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
    '''Helper class for building fitted transforms with parameter fitting.'''

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
    '''Defines manifest for Loop experiments.'''

    bar_formation: FeatureEntry = None
    feature_transforms: List[FeatureEntry] = field(default_factory=list)
    target_transforms: List[FittedTransformEntry] = field(default_factory=list)
    scaler: FittedTransformEntry = None
    required_bar_columns: List[str] = field(default_factory=list)
    target_column: str = None
    split_config: Tuple[int, int, int] = (8, 1, 2)
    
    # New model configuration fields
    model_function: Callable = None
    model_params: Dict[str, ParamValue] = field(default_factory=dict)
    metrics_function: Callable = None
    metrics_params: Dict[str, ParamValue] = field(default_factory=dict)

    def _add_transform(self, func: Callable, **params) -> 'Manifest':
        self.feature_transforms.append((func, params))
        return self

    def add_feature(self, func: Callable, **params) -> 'Manifest':
        '''Add a feature transformation to the manifest.'''
        return self._add_transform(func, **params)

    def add_indicator(self, func: Callable, **params) -> 'Manifest':
        '''Add an indicator transformation to the manifest.'''
        return self._add_transform(func, **params)

    def set_bar_formation(self, func: Callable, **params) -> 'Manifest':
        '''Set bar formation function and parameters.'''
        self.bar_formation = (func, params)
        return self

    def set_required_bar_columns(self, columns: List[str]) -> 'Manifest':
        '''Set required columns after bar formation.'''
        self.required_bar_columns = columns
        return self

    def set_split_config(self, train: int, val: int, test: int) -> 'Manifest':
        '''Set data split configuration.'''
        self.split_config = (train, val, test)
        return self

    def set_scaler(self, transform_class, param_name: str = '_scaler') -> 'Manifest':
        '''Set scaler transformation using make_fitted_scaler.'''
        self.scaler = make_fitted_scaler(param_name, transform_class)
        return self

    def with_target(self, target_column: str) -> TargetBuilder:
        '''Start building target transformations with context.'''
        return TargetBuilder(self, target_column)

    def with_model(self) -> ModelBuilder:
        '''Start building model configuration.'''
        return ModelBuilder(self)

    def prepare_data(
        self,
        raw_data: pl.DataFrame,
        round_params: Dict[str, Any]
    ) -> dict:
        '''
        Compute final data dictionary from raw data using manifest configuration.
        '''
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

        return _finalize_to_data_dict(split_data, all_datetimes, all_fitted_params, self)

    def run_model(self, data: dict, round_params: Dict[str, Any]) -> dict:
        '''
        Execute model training and evaluation using configured functions.
        
        Args:
            data (dict): Prepared data dictionary
            round_params (Dict[str, Any]): Parameter values for current round
            
        Returns:
            dict: Results including predictions, metrics, and optional extras
        '''
        if self.model_function is None:
            raise ValueError("Model function not configured. Use .with_model().set_model_function()")

        # Resolve model parameters
        resolved_model_params = _resolve_params(self.model_params, round_params)
        
        # Run model function
        model_results = self.model_function(data, **resolved_model_params)
        
        # Extract predictions
        y_pred = model_results.get('y_pred')
        y_proba = model_results.get('y_proba')
        trained_model = model_results.get('model')
        
        # Initialize results
        round_results = {}
        
        # Compute metrics if function is configured
        if self.metrics_function is not None:
            resolved_metrics_params = _resolve_params(self.metrics_params, round_params)
            metrics_results = self.metrics_function(data, y_pred, y_proba, **resolved_metrics_params)
            round_results.update(metrics_results)
        
        # Add predictions and model to results
        round_results['_preds'] = y_pred
        if trained_model is not None:
            round_results['models'] = trained_model
        
        return round_results


def _apply_fitted_transform(data: pl.DataFrame, fitted_transform):
    '''Compute transformed data using fitted transform instance.'''
    return fitted_transform.transform(data)


def make_fitted_scaler(param_name: str, transform_class):
    '''Create fitted transform entry for scaling.'''
    
    def _fit_scaler_on_train(data):
        '''Extract x_train and create scaler instance.'''
        if isinstance(data, pl.DataFrame):
            # Called during fitted transform on split data
            # Assume last column is target, rest are features
            feature_cols = data.columns[:-1]
            x_train = data.select(feature_cols)
            return transform_class(x_train=x_train, default='standard')
        else:
            # Fallback: assume data is already x_train
            return transform_class(x_train=data, default='standard')
    
    return ([
        (param_name, _fit_scaler_on_train, {})
    ],
    _apply_fitted_transform, {
        'fitted_transform': param_name
    })


def _resolve_params(params: Dict[str, Any], round_params: Dict[str, Any]) -> Dict[str, Any]:
    '''Resolve parameters using just-in-time detection with actual round_params.'''
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
    '''Compute bar formation on data and return post-bar datetimes.'''
    if manifest.bar_formation and round_params.get('bar_type', 'base') != 'base':
        func, base_params = manifest.bar_formation
        resolved = _resolve_params(base_params, round_params)
        lazy_data = data.lazy().pipe(func, **resolved)
        bar_data = lazy_data.collect()
        all_datetimes = bar_data['datetime'].to_list()
    else:
        all_datetimes = data['datetime'].to_list()
        bar_data = data

    # Validate required columns
    available_cols = list(bar_data.columns)
    for required_col in manifest.required_bar_columns:
        assert required_col in available_cols, (
            f"Required bar column '{required_col}' not found after bar formation"
        )

    return all_datetimes, bar_data


def _apply_feature_transforms(manifest: Manifest, lazy_data, round_params: Dict[str, Any]):
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
    '''Compute fitted transforms on eager DataFrame.'''
    for fitted_param_computations, func, base_params in transform_entries:
        # Fit parameters on training data only
        for param_name, compute_func, compute_base_params in fitted_param_computations:
            if param_name not in all_fitted_params and is_training:
                resolved = _resolve_params(compute_base_params, round_params)
                value = compute_func(data, **resolved)
                all_fitted_params[param_name] = value

        # Apply transform using fitted parameters
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
    if manifest.scaler:
        return _apply_fitted_transforms(
            [manifest.scaler], data, round_params,
            all_fitted_params, is_training
        )
    return data, all_fitted_params


def _finalize_to_data_dict(
        split_data: List[pl.DataFrame],
        all_datetimes: List,
        fitted_params: Dict[str, Any],
        manifest: Manifest
) -> dict:
    # Validate all splits have datetime column
    for i, split_df in enumerate(split_data):
        assert 'datetime' in split_df.columns, f"Split {i} missing 'datetime' column"

    # Ensure target_column is last column in all splits
    if manifest.target_column:
        for i, split_df in enumerate(split_data):
            cols = list(split_df.columns)
            if manifest.target_column in cols:
                cols.remove(manifest.target_column)
                cols.append(manifest.target_column)
                split_data[i] = split_df.select(cols)
            else:
                raise ValueError(f"Split {i} missing target column '{manifest.target_column}'")

    cols = list(split_data[0].columns)
    data_dict = split_data_to_prep_output(split_data, cols, all_datetimes)

    # Add fitted parameters to data_dict
    for param_name, param_value in fitted_params.items():
        data_dict[param_name] = param_value

    return data_dict