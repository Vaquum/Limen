import polars as pl

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple, Union

ParamValue = Union[Any, Callable[[Dict[str, Any]], Any]]
FeatureEntry = Tuple[Callable[..., pl.LazyFrame], Dict[str, ParamValue]]


@dataclass
class Manifest:

    '''
    Compute universal experiment manifest for Loop experiments.

    Args:
        name (str): Unique identifier for the experiment manifest
        description (str): Human-readable description of the experiment purpose
        bar_formation (FeatureEntry): Bar formation function with parameters
        indicators (List[FeatureEntry]): List of indicator functions with their parameters
        features (List[FeatureEntry]): List of feature functions with their parameters
        transformations (List[FeatureEntry]): List of transformation functions with parameters
        required_columns (List[str]): Base columns that must be present (e.g., datetime, OHLCV)
        target_column (Optional[str]): Name of target column to place last in feature order
        split_config (Tuple[int, int, int]): Train, validation, test split ratios

    Returns:
        Manifest: Configured experiment manifest instance
    '''

    name: str
    description: str
    bar_formation: FeatureEntry = None
    indicators: List[FeatureEntry] = field(default_factory=list)
    features: List[FeatureEntry] = field(default_factory=list)
    transformations: List[FeatureEntry] = field(default_factory=list)
    required_columns: List[str] = field(default_factory=list)
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
        return all_datetimes, bar_data
    else:
        all_datetimes = data['datetime'].to_list()
        return all_datetimes, data


def process_manifest(
    manifest: Manifest,
    data: pl.DataFrame,
    round_params: Dict[str, Any]) -> pl.DataFrame:

    '''
    Compute data transformations according to manifest specification.

    Args:
        manifest (Manifest): Experiment manifest containing pipeline configuration
        data (pl.DataFrame): Input klines dataset
        round_params (Dict[str, Any]): Parameter values for current round

    Returns:
        pl.DataFrame: Processed data with indicators and features applied
    '''

    lazy_data = data.lazy()

    for func, base_params in manifest.indicators:
        resolved = resolve_params(base_params, round_params)
        lazy_data = lazy_data.pipe(func, **resolved)

    for func, base_params in manifest.features:
        resolved = resolve_params(base_params, round_params)
        lazy_data = lazy_data.pipe(func, **resolved)

    for func, base_params in manifest.transformations:
        resolved = resolve_params(base_params, round_params)
        lazy_data = lazy_data.pipe(func, **resolved)

    # Auto-filter all null/NaN values from all columns (except datetime)
    collected_data = lazy_data.collect()

    # Get all columns except datetime for null filtering
    filter_cols = [col for col in collected_data.columns if col != 'datetime']

    if filter_cols:
        # Use drop_nulls on specific columns to preserve datetime
        collected_data = collected_data.drop_nulls(filter_cols)

    return collected_data
