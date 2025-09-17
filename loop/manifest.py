import polars as pl

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple, Union
from loop.utils.splits import split_data_to_prep_output, split_sequential

ParamValue = Union[Any, Callable[[Dict[str, Any]], Any]]
FeatureEntry = Tuple[Callable[..., pl.LazyFrame], Dict[str, ParamValue]]

FittedParamsComputationEntry = Tuple[str, Callable[..., Any], Dict[str, ParamValue]]

FittedTransformEntry = Tuple[
    List[FittedParamsComputationEntry],
    Callable[..., pl.LazyFrame],
    Dict[str, ParamValue]
]


@dataclass
class Manifest:

    '''
    Defines manifest for Loop experiments.

    Args:
        bar_formation (FeatureEntry): Bar formation function with parameters
        feature_transforms (List[FeatureEntry]): Simple feature transformations (lazy evaluation)
        target_transforms (List[FittedTransformEntry]): Fitted transforms for target creation
        scaler (FittedTransformEntry): Single scaler for all features
        required_bar_columns (List[str]): Columns that must be present after bar formation
        target_column (str): Name of target column to place last in feature order
        split_config (Tuple[int, int, int]): Train, validation, test split ratios
    '''

    bar_formation: FeatureEntry = None
    feature_transforms: List[FeatureEntry] = field(default_factory=list)
    target_transforms: List[FittedTransformEntry] = field(default_factory=list)
    scaler: FittedTransformEntry = None
    required_bar_columns: List[str] = field(default_factory=list)
    target_column: str = None
    split_config: Tuple[int, int, int] = (8, 1, 2)

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

        all_datetimes, bar_data = _process_bars(self, raw_data, round_params)
        split_data = split_sequential(bar_data, self.split_config)

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
        'fitted_transform': lambda p: p[param_name]
    })


def _resolve_params(params: Dict[str, ParamValue], round_params: Dict[str, Any]) -> Dict[str, Any]:

    '''
    Compute resolved parameters from specification and round values.

    Args:
        params (Dict[str, ParamValue]): Parameter specification dictionary
        round_params (Dict[str, Any]): Round-specific parameter values

    Returns:
        Dict[str, Any]: Resolved parameter dictionary
    '''

    resolved = {}
    for key, value in params.items():
        if callable(value):
            resolved[key] = value(round_params)
        else:
            resolved[key] = value
    return resolved


def _process_bars(
    manifest: Manifest,
    data: pl.DataFrame,
    round_params: Dict[str, Any]) -> Tuple[List, pl.DataFrame]:

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
        lazy_data = data.lazy().pipe(func, **resolved)
        bar_data = lazy_data.collect()
        all_datetimes = bar_data['datetime'].to_list()
    else:
        all_datetimes = data['datetime'].to_list()
        bar_data = data

    # Validate required columns are present after bar formation
    available_cols = list(bar_data.columns)
    for required_col in manifest.required_bar_columns:
        assert required_col in available_cols, \
        f"Required bar column '{required_col}' not found after bar formation"

    return all_datetimes, bar_data


def _apply_feature_transforms(manifest: Manifest, lazy_data, round_params: Dict[str, Any]):
    for func, base_params in manifest.feature_transforms:
        resolved = _resolve_params(base_params, round_params)
        lazy_data = lazy_data.pipe(func, **resolved)
    return lazy_data


def _apply_fitted_transforms(transform_entries: List[FittedTransformEntry], data: pl.DataFrame,
                            round_params: Dict[str, Any], all_fitted_params: Dict[str, Any],
                            is_training: bool) -> Tuple[pl.DataFrame, Dict[str, Any]]:

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


def _apply_target_transforms(manifest: Manifest, data: pl.DataFrame, round_params: Dict[str, Any],
                           all_fitted_params: Dict[str, Any], is_training: bool) -> Tuple[pl.DataFrame, Dict[str, Any]]:
    # Create enhanced params without modifying original round_params
    enhanced_round_params = round_params.copy()
    if manifest.target_column:
        enhanced_round_params['target_column'] = manifest.target_column

    return _apply_fitted_transforms(manifest.target_transforms, data, enhanced_round_params,
                                   all_fitted_params, is_training)


def _apply_scaler(manifest: Manifest, data: pl.DataFrame, round_params: Dict[str, Any],
                 all_fitted_params: Dict[str, Any], is_training: bool) -> Tuple[pl.DataFrame, Dict[str, Any]]:
    if manifest.scaler:
        return _apply_fitted_transforms([manifest.scaler], data, round_params,
                                       all_fitted_params, is_training)
    return data, all_fitted_params


def _finalize_to_data_dict(split_data: List[pl.DataFrame], all_datetimes: List,
                          fitted_params: Dict[str, Any], manifest: Manifest) -> dict:
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

    return data_dict


