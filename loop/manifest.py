import polars as pl

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple, Union

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
    Compute universal experiment manifest for Loop experiments.

    Args:
        bar_formation (FeatureEntry): Bar formation function with parameters
        ordered_transformations (List[Union[FeatureEntry, FittedTransformEntry]]): Ordered list of transformations
        required_bar_columns (List[str]): Columns that must be present after bar formation
        target_column (Optional[str]): Name of target column to place last in feature order
        split_config (Tuple[int, int, int]): Train, validation, test split ratios

    Returns:
        Manifest: Configured experiment manifest instance
    '''

    bar_formation: FeatureEntry = None
    ordered_transformations: List[Union[FeatureEntry, FittedTransformEntry]] = field(default_factory=list)
    required_bar_columns: List[str] = field(default_factory=list)
    target_column: str = None
    split_config: Tuple[int, int, int] = (8, 1, 2)


def resolve_params(params: Dict[str, ParamValue], round_params: Dict[str, Any]) -> Dict[str, Any]:

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


def process_bars(
    manifest: Manifest,
    data: pl.DataFrame,
    round_params: Dict[str, Any]) -> Tuple[List, pl.DataFrame]:

    '''
    Apply bar formation to data and return post-bar datetimes.

    Args:
        manifest (Manifest): Experiment manifest containing bar formation config
        data (pl.DataFrame): Input raw dataset
        round_params (Dict[str, Any]): Parameter values for current round

    Returns:
        Tuple[List, pl.DataFrame]: Post-bar datetimes and processed data
    '''
    if manifest.bar_formation and round_params.get('bar_type', 'base') != 'base':
        func, base_params = manifest.bar_formation
        resolved = resolve_params(base_params, round_params)
        lazy_data = data.lazy().pipe(func, **resolved)
        bar_data = lazy_data.collect()
        all_datetimes = bar_data['datetime'].to_list()
    else:
        all_datetimes = data['datetime'].to_list()
        bar_data = data

    # Validate required columns are present after bar formation
    available_cols = list(bar_data.columns)
    for required_col in manifest.required_bar_columns:
        assert required_col in available_cols, f"Required bar column '{required_col}' not found after bar formation"

    return all_datetimes, bar_data


def process_manifest(
    manifest: Manifest,
    split_data: List[pl.DataFrame],
    round_params: Dict[str, Any]
) -> Tuple[List[pl.DataFrame], Dict[str, Any]]:

    '''
    Compute data transformations according to manifest specification.

    Args:
        manifest (Manifest): Experiment manifest containing pipeline configuration
        split_data (List[pl.DataFrame]): Input split datasets
        round_params (Dict[str, Any]): Parameter values for current round

    Returns:
        Tuple[List[pl.DataFrame], Dict[str, Any]]: Processed split data and fitted parameters
    '''

    all_fitted_params = {}

    for i in range(len(split_data)):
        lazy_data = split_data[i].lazy()

        # Process transformations in order
        for transform_entry in manifest.ordered_transformations:
            if len(transform_entry) == 2:  # Simple transform
                func, base_params = transform_entry
                resolved = resolve_params(base_params, round_params)
                lazy_data = lazy_data.pipe(func, **resolved)

            elif len(transform_entry) == 3:  # Fit transform
                fitted_param_computations, func, base_params = transform_entry

                for param_name, compute_func, compute_base_params in fitted_param_computations:
                    if param_name not in all_fitted_params:
                        if i == 0:  # Only compute on training data
                            current_data = lazy_data.collect()
                            resolved = resolve_params(compute_base_params, round_params)
                            value = compute_func(current_data, **resolved)
                            all_fitted_params[param_name] = value
                            lazy_data = current_data.lazy()

                # Merge fitted params into round_params for parameter resolution
                combined_round_params = {**round_params, **all_fitted_params}
                resolved = resolve_params(base_params, combined_round_params)
                lazy_data = lazy_data.pipe(func, **resolved)
            else:
                raise ValueError(f"Transform entry must have 2 or 3 elements, got {len(transform_entry)}")

        collected_data = lazy_data.collect()
        split_data[i] = collected_data.drop_nulls(collected_data.columns)

    return split_data, all_fitted_params
