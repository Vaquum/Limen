import polars as pl
import inspect
import importlib

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple, Union
from limen.data.utils import split_data_to_prep_output
from limen.data.utils import split_sequential

ParamValue = Union[Any, Callable[[Dict[str, Any]], Any]]
FeatureEntry = Tuple[Callable[..., pl.LazyFrame], Dict[str, ParamValue]]

FittedParamsComputationEntry = Tuple[str, Callable[..., Any], Dict[str, ParamValue]]

FittedTransformEntry = Tuple[
    List[FittedParamsComputationEntry],
    Callable[..., pl.LazyFrame],
    Dict[str, ParamValue]
]


@dataclass
class DataSourceConfig:

    '''Declarative configuration for data fetching in manifests.'''

    method: Callable
    params: Dict[str, Any] = field(default_factory=dict)


class DataSourceResolver:

    '''Resolves data source config to DataFrame.'''

    @staticmethod
    def resolve(config: DataSourceConfig) -> pl.DataFrame:

        '''
        Execute data source config and return DataFrame.

        Args:
            config: DataSourceConfig instance

        Returns:
            pl.DataFrame: Fetched data
        '''

        method = config.method
        params = config.params

        if inspect.ismethod(method) or (hasattr(method, '__self__') and method.__self__ is not None):
            result = method(**params)
            if hasattr(method.__self__, 'data'):
                return method.__self__.data
            return result

        elif inspect.isfunction(method):
            if '.' in method.__qualname__:
                module_name = method.__module__
                class_name = method.__qualname__.rsplit('.', 1)[0]

                module = importlib.import_module(module_name)
                cls = getattr(module, class_name)

                instance = cls()
                bound_method = getattr(instance, method.__name__)
                bound_method(**params)

                if hasattr(instance, 'data'):
                    return instance.data
                else:
                    raise ValueError(
                        f"Method {method.__qualname__} executed successfully but "
                        f"instance does not have 'data' attribute. Expected data source "
                        f"methods to populate instance.data"
                    )
            else:
                return method(**params)

        else:
            raise ValueError(f"Unsupported callable type: {type(method)}")


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

    data_source_config: DataSourceConfig = None
    test_data_source_config: DataSourceConfig = None
    pre_split_data_selector: FeatureEntry = None
    split_config: Tuple[int, int, int] = (8, 1, 2)
    bar_formation: FeatureEntry = None
    required_bar_columns: List[str] = field(default_factory=list)
    feature_transforms: List[FeatureEntry] = field(default_factory=list)
    target_column: str = None
    target_transforms: List[FittedTransformEntry] = field(default_factory=list)
    scaler: FittedTransformEntry = None
    data_dict_extension: Callable = None
    
    model_function: Callable = None
    model_params: Dict[str, ParamValue] = field(default_factory=dict)
    metrics_params: Dict[str, ParamValue] = field(default_factory=dict)

    def _add_transform(self, func: Callable, **params) -> 'Manifest':

        self.feature_transforms.append((func, params))

        return self

    def set_data_source(self,
                       method: Callable,
                       params: Dict[str, Any] = None) -> 'Manifest':

        '''
        Configure production data source for the manifest.

        Args:
            method (Callable): Method or function reference (e.g., HistoricalData.get_spot_klines)
            params (dict): Parameters to pass to the method

        Returns:
            Manifest: Self for method chaining
        '''

        self.data_source_config = DataSourceConfig(
            method=method,
            params=params or {}
        )

        return self

    def set_test_data_source(self,
                            method: Callable,
                            params: Dict[str, Any] = None) -> 'Manifest':

        '''
        Configure test data source for the manifest.

        Args:
            method (Callable): Function reference (e.g., HistoricalData._get_data_for_test)
            params (dict): Parameters to pass to the function

        Returns:
            Manifest: Self for method chaining
        '''

        self.test_data_source_config = DataSourceConfig(
            method=method,
            params=params or {}
        )

        return self

    def fetch_data(self) -> pl.DataFrame:

        '''Fetch data using configured data source.'''

        if self.data_source_config is None:
            raise ValueError('No data source configured')

        return DataSourceResolver.resolve(self.data_source_config)

    def fetch_test_data(self) -> pl.DataFrame:

        '''Fetch data using configured test data source.'''

        if self.test_data_source_config is None:
            raise ValueError('No test data source configured')

        return DataSourceResolver.resolve(self.test_data_source_config)

    def add_feature(self, func: Callable, **params) -> 'Manifest':

        '''
        Add feature transformation to the manifest.

        Args:
            func (Callable): Feature transformation function
            **params: Parameters for the transformation

        Returns:
            Manifest: Self for method chaining
        '''

        return self._add_transform(func, **params)

    def add_indicator(self, func: Callable, **params) -> 'Manifest':

        '''
        Add indicator transformation to the manifest.

        Args:
            func (Callable): Indicator transformation function
            **params: Parameters for the transformation

        Returns:
            Manifest: Self for method chaining
        '''

        return self._add_transform(func, **params)

    def set_pre_split_data_selector(self, func: Callable, **params) -> 'Manifest':

        '''
        Set pre-split data selector function and parameters.

        Args:
            func (Callable): Data selector function
            **params: Parameters for data selection

        Returns:
            Manifest: Self for method chaining
        '''

        self.pre_split_data_selector = (func, params)
        return self

    def set_bar_formation(self, func: Callable, **params) -> 'Manifest':

        '''
        Set bar formation function and parameters.

        Args:
            func (Callable): Bar formation function
            **params: Parameters for bar formation

        Returns:
            Manifest: Self for method chaining
        '''

        self.bar_formation = (func, params)

        return self


    def set_required_bar_columns(self, columns: List[str]) -> 'Manifest':

        '''
        Set required columns after bar formation.

        Args:
            columns (List[str]): List of required column names

        Returns:
            Manifest: Self for method chaining
        '''

        self.required_bar_columns = columns

        return self

    def set_split_config(self, train: int, val: int, test: int) -> 'Manifest':

        '''
        Set data split configuration.

        Args:
            train (int): Training split ratio
            val (int): Validation split ratio
            test (int): Test split ratio

        Returns:
            Manifest: Self for method chaining
        '''

        self.split_config = (train, val, test)

        return self

    def set_scaler(self, transform_class, param_name: str = '_scaler') -> 'Manifest':

        '''
        Set scaler transformation using make_fitted_scaler.

        Args:
            transform_class: Transform class to use for scaling
            param_name (str): Parameter name for fitted scaler

        Returns:
            Manifest: Self for method chaining
        '''

        self.scaler = make_fitted_scaler(param_name, transform_class)

        return self

    def with_target(self, target_column: str) -> TargetBuilder:

        '''
        Start building target transformations with context.

        Args:
            target_column (str): Name of target column

        Returns:
            TargetBuilder: Builder for target transformations
        '''

        return TargetBuilder(self, target_column)

    def with_model(self, model_function: Callable) -> 'Manifest':

        '''
        Configure model function for training and evaluation.

        Args:
            model_function (Callable): Model function that takes (data, **params) and returns results

        Returns:
            Manifest: Self for method chaining

        NOTE: The model function should accept data dict and return results dict with metrics and predictions.
        Parameters are auto-mapped from round_params based on model function signature.
        '''

        self.model_function = model_function

        return self

    def add_to_data_dict(self, func: Callable) -> 'Manifest':

        '''
        Configure data_dict extension function to add custom entries after data preparation.

        Args:
            func (Callable): Extension function with signature (data_dict, split_data, round_params, fitted_params) -> dict

        Returns:
            Manifest: Self for method chaining

        NOTE: The extension function receives the base data_dict and full split DataFrames.
        It should modify and return the data_dict with any additional custom entries needed by the model.
        '''

        self.data_dict_extension = func
        return self

    def compute_test_bars(self, raw_data: pl.DataFrame, round_params: Dict[str, Any]) -> pl.DataFrame:

        '''
        Compute test split bar data from raw data using manifest bar formation configuration.

        NOTE: Used by Log system to reconstruct the same test bar data that was used in training.

        Args:
            raw_data (pl.DataFrame): Raw input dataset
            round_params (Dict[str, Any]): Parameter values for current round

        Returns:
            pl.DataFrame: Bar-formed test split data
        '''

        if self.pre_split_data_selector:
            func, base_params = self.pre_split_data_selector
            resolved = _resolve_params(base_params, round_params)
            raw_data = func(raw_data, **resolved)

        split_data = split_sequential(raw_data, self.split_config)
        test_split = split_data[2]
        _, test_bar_data = _process_bars(self, test_split, round_params)

        return test_bar_data

    def prepare_data(
        self,
        raw_data: pl.DataFrame,
        round_params: Dict[str, Any]
    ) -> dict:

        '''
        Compute final data dictionary from raw data using manifest configuration.

        Args:
            raw_data (pl.DataFrame): Raw input dataset
            round_params (Dict[str, Any]): Parameter values for current round

        Returns:
            dict: Final data dictionary ready for model training
        '''

        if self.pre_split_data_selector:
            func, base_params = self.pre_split_data_selector
            resolved = _resolve_params(base_params, round_params)
            raw_data = func(raw_data, **resolved)

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

            data = data.drop_nulls()

            data, all_fitted_params = _apply_scaler(
                self, data, round_params, all_fitted_params, is_training=(i == 0)
            )

            split_data[i] = data.drop_nulls()

        return _finalize_to_data_dict(self, split_data, all_datetimes, all_fitted_params, round_params)

    def run_model(self, data: dict, round_params: Dict[str, Any]) -> dict:

        '''
        Execute model training and evaluation using configured functions.

        Args:
            data (dict): Prepared data dictionary
            round_params (Dict[str, Any]): Parameter values for current round

        Returns:
            dict: Results including predictions, metrics, and optional extras

        Raises:
            ValueError: If required model function parameters are missing from round_params

        NOTE: Auto-maps parameters from round_params to model function signature.
        Parameters in round_params override model function defaults.
        Parameters not in round_params use model function defaults.
        Required parameters (no defaults) must be in round_params.
        '''

        if self.model_function is None:
            raise ValueError('Model function not configured. Use .with_model(model_function) before run_model().')

        sig = inspect.signature(self.model_function)
        model_kwargs = {}

        for param_name, param_obj in sig.parameters.items():
            if param_name == 'data':
                continue

            if param_name in round_params:
                model_kwargs[param_name] = round_params[param_name]
            elif param_obj.default != inspect.Parameter.empty:
                model_kwargs[param_name] = param_obj.default
            else:
                raise ValueError(
                    f"Missing required parameter '{param_name}' for model function. "
                    'It must be provided in round_params.'
                )

        round_results = self.model_function(data, **model_kwargs)

        return round_results


def _apply_fitted_transform(data: pl.DataFrame, fitted_transform):

    '''
    Compute transformed data using fitted transform instance.

    Args:
        data (pl.DataFrame): Data to transform
        fitted_transform: Fitted transform instance with .transform() method

    Returns:
        pl.DataFrame: Transformed data
    '''

    return fitted_transform.transform(data)


def make_fitted_scaler(param_name: str, transform_class):

    '''
    Create fitted transform entry for scaling.

    Args:
        param_name (str): Name for the fitted parameter
        transform_class: Transform class to instantiate

    Returns:
        FittedTransformEntry: Complete fitted transform configuration
    '''

    return ([
        (param_name, lambda data: transform_class(data), {})
    ],
    _apply_fitted_transform, {
        'fitted_transform': param_name
    })


def _resolve_params(params: Dict[str, Any], round_params: Dict[str, Any]) -> Dict[str, Any]:

    '''
    Resolve parameters using just-in-time detection with actual round_params.

    Args:
        params (Dict[str, Any]): Parameter specification dictionary
        round_params (Dict[str, Any]): Round-specific parameter values

    Returns:
        Dict[str, Any]: Resolved parameter dictionary
    '''

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

    '''
    Compute bar formation on data and return post-bar datetimes.

    Args:
        manifest (Manifest): Experiment manifest containing bar formation config
        data (pl.DataFrame): Input raw dataset
        round_params (Dict[str, Any]): Parameter values for current round

    Returns:
        Tuple[List, pl.DataFrame]: Post-bar datetimes and processed data
    '''

    if manifest.bar_formation and round_params.get('bar_type', 'base') != 'base':
        func, base_params = manifest.bar_formation
        resolved = _resolve_params(base_params, round_params)
        bar_data = data.pipe(func, **resolved)
        all_datetimes = bar_data['datetime'].to_list()
    else:
        all_datetimes = data['datetime'].to_list()
        bar_data = data

    # Validate required columns are present after bar formation
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

    '''
    Compute fitted transforms on eager DataFrame.

    Args:
        transform_entries (List[FittedTransformEntry]): List of fitted transform configurations
        data (pl.DataFrame): DataFrame to apply transforms to
        round_params (Dict[str, Any]): Parameter values for current round
        all_fitted_params (Dict[str, Any]): Previously fitted parameters
        is_training (bool): Whether this is training data for fitting

    Returns:
        Tuple[pl.DataFrame, Dict[str, Any]]: Transformed data and updated fitted parameters
    '''

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
        manifest: Manifest,
        split_data: List[pl.DataFrame],
        all_datetimes: List,
        fitted_params: Dict[str, Any],
        round_params: Dict[str, Any]
) -> dict:

    # Validate all splits have datetime column
    for i, split_df in enumerate(split_data):
        assert 'datetime' in split_df.columns, f"Split {i} missing 'datetime' column"

    # Ensure target_column is last column in all splits
    if manifest.target_column:
        for i, split_df in enumerate(split_data):
            cols = list(split_df.columns)
            if manifest.target_column in cols:
                # Move target_column to end
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

    data_dict['_feature_names'] = cols

    # Apply data_dict extension if configured
    if manifest.data_dict_extension:
        data_dict = manifest.data_dict_extension(
            data_dict=data_dict,
            split_data=split_data,
            round_params=round_params,
            fitted_params=fitted_params
        )

    return data_dict
