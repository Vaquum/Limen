import polars as pl

from collections.abc import Sequence
from itertools import accumulate


def split_sequential(data: pl.DataFrame, ratios: Sequence[int]) -> list[pl.DataFrame]:
    '''
    Compute sequential data splits with proportional lengths based on ratios.

    Args:
        data (pl.DataFrame): Polars DataFrame to split sequentially
        ratios (Sequence[int]): Sequence of positive integers defining split proportions

    Returns:
        List[pl.DataFrame]: List of DataFrames partitioned sequentially without losing or duplicating rows
    '''

    total = data.height
    if total == 0:
        return [pl.DataFrame() for _ in ratios]

    total_ratio = sum(ratios)

    sizes: list[int] = []
    cumulative = 0

    for r in ratios[:-1]:
        chunk_size = int(total * r / total_ratio)
        sizes.append(chunk_size)
        cumulative += chunk_size

    sizes.append(total - cumulative)

    out: list[pl.DataFrame] = []
    start = 0
    for size in sizes:
        out.append(data.slice(start, size))
        start += size

    return out


def split_random(data: pl.DataFrame, ratios: Sequence[int], seed: int | None = None) -> list[pl.DataFrame]:
    '''
    Compute random data splits with proportional lengths based on ratios.

    Args:
        data (pl.DataFrame): Polars DataFrame to split randomly
        ratios (Sequence[int]): Sequence of positive integers defining split proportions
        seed (int): Seed for random number generator

    Returns:
        List[pl.DataFrame]: List of randomly shuffled DataFrames with proportional sizes
    '''

    total = data.height
    total_ratio = sum(ratios)
    bounds = [int(total * c / total_ratio) for c in accumulate(ratios)]
    starts = [0, *bounds[:-1]]

    return [data.sample(fraction=1.0, seed=seed, shuffle=True).slice(start, end - start) for start, end in zip(starts, bounds, strict=True)]


def _apply_confidence_gate(
    split_df: pl.DataFrame,
    confidence_col: str,
    target_col: str,
    confidence_threshold: float,
    gated_value: int | float,
) -> pl.DataFrame:
    '''
    Compute target gating using confidence threshold.

    Args:
        split_df (pl.DataFrame): Split DataFrame with confidence and target columns
        confidence_col (str): Name of confidence column
        target_col (str): Name of target column to gate
        confidence_threshold (float): Confidence threshold below which target is gated
        gated_value (int | float): Value assigned when confidence is below threshold

    Returns:
        pl.DataFrame: DataFrame with gated target column
    '''

    if confidence_col not in split_df.columns:
        raise ValueError(
            f"confidence_col '{confidence_col}' not found in split data"
        )

    if target_col not in split_df.columns:
        raise ValueError(
            f"gated_target_col '{target_col}' not found in split data"
        )

    return split_df.with_columns(
        pl.when(pl.col(confidence_col) < confidence_threshold)
        .then(pl.lit(gated_value))
        .otherwise(pl.col(target_col))
        .alias(target_col)
    )


def split_data_to_prep_output(split_data: list,
                              cols: list,
                              all_datetimes: list,
                              confidence_col: str | None = None,
                              confidence_threshold: float | None = None,
                              gated_target_col: str | None = None,
                              gated_value: int | float = 0,
                              gate_splits: tuple[str, ...] = ('test',)) -> dict:
    '''
    Compute data preparation output dictionary from split data and column names.

    Args:
        split_data (list): List of three DataFrames representing train, validation, and test splits
        cols (list): Column names where the last column is the target variable
        all_datetimes (list): List of all datetimes
        confidence_col (str | None): Optional confidence column used for gating
        confidence_threshold (float | None): Threshold below which target is gated
        gated_target_col (str | None): Target column to gate; defaults to final target column
        gated_value (int | float): Value to assign when confidence is below threshold
        gate_splits (tuple[str, ...]): Splits to gate. Supported values: train, val, test

    Returns:
        dict: Dictionary with train, validation, and test features and targets
    '''

    split_map = {'train': 0, 'val': 1, 'test': 2}
    invalid_splits = [
        split_name for split_name in gate_splits if split_name not in split_map]
    if invalid_splits:
        raise ValueError(
            f"Unsupported gate_splits entries: {invalid_splits}. Supported: {list(split_map.keys())}"
        )

    if (confidence_col is None) ^ (confidence_threshold is None):
        raise ValueError(
            'confidence_col and confidence_threshold must both be provided for confidence gating'
        )

    if confidence_threshold is not None and not isinstance(confidence_threshold, int | float):
        raise ValueError('confidence_threshold must be numeric')

    remaining_datetimes = split_data[0]['datetime'].to_list()
    remaining_datetimes += split_data[1]['datetime'].to_list()
    remaining_datetimes += split_data[2]['datetime'].to_list()

    first_test_datetime = split_data[2]['datetime'].min()
    last_test_datetime = split_data[2]['datetime'].max()

    split_data[0] = split_data[0].drop('datetime')
    split_data[1] = split_data[1].drop('datetime')
    split_data[2] = split_data[2].drop('datetime')

    output_cols = cols.copy()
    if 'datetime' in output_cols:
        output_cols.remove('datetime')
    else:
        raise ValueError(
            'SFDs must contain `datetime` in data up to when it enters `split_data_to_prep_output` in sfd.prep')

    target_col = gated_target_col or output_cols[-1]

    if target_col != output_cols[-1]:
        raise ValueError(
            f"gated_target_col must match the final target column '{output_cols[-1]}'"
        )

    if confidence_col is not None and confidence_threshold is not None:
        for split_name in gate_splits:
            split_idx = split_map[split_name]
            split_data[split_idx] = _apply_confidence_gate(
                split_df=split_data[split_idx],
                confidence_col=confidence_col,
                target_col=target_col,
                confidence_threshold=float(confidence_threshold),
                gated_value=gated_value,
            )

    data_dict = {'x_train': split_data[0][output_cols[:-1]],
                 'y_train': split_data[0][output_cols[-1]],
                 'x_val': split_data[1][output_cols[:-1]],
                 'y_val': split_data[1][output_cols[-1]],
                 'x_test': split_data[2][output_cols[:-1]],
                 'y_test': split_data[2][output_cols[-1]]}

    data_dict['_alignment'] = {}

    data_dict['_alignment']['missing_datetimes'] = sorted(set(all_datetimes) - set(remaining_datetimes))
    data_dict['_alignment']['first_test_datetime'] = first_test_datetime
    data_dict['_alignment']['last_test_datetime'] = last_test_datetime

    return data_dict
