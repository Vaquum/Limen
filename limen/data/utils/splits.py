import polars as pl

from collections.abc import Sequence
from datetime import date
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


def split_by_dates(
    data: pl.DataFrame,
    train_start: date, train_end: date,
    val_start: date, val_end: date,
    test_start: date, test_end: date,
) -> list[pl.DataFrame]:

    '''
    Split a datetime-indexed DataFrame into train/val/test by half-open
    date windows `[start, end)`.

    Each window selects its rows independently. No row from outside all
    three windows enters any split, and no row appears in more than one
    split when windows are non-overlapping (the ordering contract that
    `Manifest.set_split_dates` enforces). Use when the train / val /
    test boundaries must be specific dates rather than proportions of
    the input row count — paired with `Manifest.set_split_dates`, this
    is the splitter the manifest pipeline uses on its date path.

    Args:
        data (pl.DataFrame): Input data; must have a `datetime` column
        train_start (date | datetime): Train window start (inclusive)
        train_end   (date | datetime): Train window end (exclusive)
        val_start   (date | datetime): Val window start (inclusive)
        val_end     (date | datetime): Val window end (exclusive)
        test_start  (date | datetime): Test window start (inclusive)
        test_end    (date | datetime): Test window end (exclusive)

    Returns:
        list[pl.DataFrame]: three DataFrames in train, val, test order

    Raises:
        TypeError: If any bound is not a `date` or `datetime` instance
    '''

    bounds = (train_start, train_end, val_start, val_end, test_start, test_end)
    for name, value in zip(
        ('train_start', 'train_end', 'val_start', 'val_end', 'test_start', 'test_end'),
        bounds,
        strict=True,
    ):
        if not isinstance(value, date):
            raise TypeError(
                f'splits {name} must be a date or datetime instance, '
                f'got {type(value).__name__}: {value!r}'
            )

    return [
        data.filter((pl.col('datetime') >= train_start) & (pl.col('datetime') < train_end)),
        data.filter((pl.col('datetime') >= val_start)   & (pl.col('datetime') < val_end)),
        data.filter((pl.col('datetime') >= test_start)  & (pl.col('datetime') < test_end)),
    ]


def _compute_alignment(split_data: list, all_datetimes: list) -> dict:
    remaining = (split_data[0]['datetime'].to_list()
                 + split_data[1]['datetime'].to_list()
                 + split_data[2]['datetime'].to_list())
    return {
        'missing_datetimes': sorted(set(all_datetimes) - set(remaining)),
        'first_test_datetime': split_data[2]['datetime'].min(),
        'last_test_datetime': split_data[2]['datetime'].max(),
    }


def split_data_to_prep_output(split_data: list,
                              cols: list,
                              all_datetimes: list) -> dict:

    '''
    Compute data preparation output dictionary from split data and column names.

    Args:
        split_data (list): List of three DataFrames representing train, validation, and test splits
        cols (list): Column names where the last column is the target variable
        all_datetimes (list): List of all datetimes

    Returns:
        dict: Dictionary with train, validation, and test features and targets
    '''

    alignment = _compute_alignment(split_data, all_datetimes)

    split_data[0] = split_data[0].drop('datetime')
    split_data[1] = split_data[1].drop('datetime')
    split_data[2] = split_data[2].drop('datetime')

    if 'datetime' in cols:
        cols.remove('datetime')
    else:
        raise ValueError('SFDs must contain `datetime` in data up to when it enters `split_data_to_prep_output` in sfd.prep')

    data_dict = {'x_train': split_data[0][cols[:-1]],
                 'y_train': split_data[0][cols[-1]],
                 'x_val': split_data[1][cols[:-1]],
                 'y_val': split_data[1][cols[-1]],
                 'x_test': split_data[2][cols[:-1]],
                 'y_test': split_data[2][cols[-1]]}

    data_dict['_alignment'] = alignment

    return data_dict


def split_data_to_rule_based_prep_output(split_data: list, all_datetimes: list) -> dict:

    '''
    Compute data preparation output dictionary for rule-based strategies.

    Args:
        split_data (list): List of three DataFrames representing train, validation, and test splits
        all_datetimes (list): List of all datetimes

    Returns:
        dict: Dictionary with train, val, test full DataFrames and _alignment metadata
    '''

    alignment = _compute_alignment(split_data, all_datetimes)

    return {
        'train': split_data[0].drop('datetime'),
        'val': split_data[1].drop('datetime'),
        'test': split_data[2].drop('datetime'),
        '_alignment': alignment,
    }
