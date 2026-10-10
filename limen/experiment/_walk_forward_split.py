from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import polars as pl

from limen.data.utils import split_walk_forward

__all__ = ['WalkForwardConfig', 'read_walk_forward_config', 'resolve_walk_forward_split', 'validate_fold_splits']


@dataclass(frozen=True)
class WalkForwardConfig:
    n_folds: int
    test_bars: int
    purge_bars: int
    embargo_bars: int
    anchored: bool

    def __post_init__(self) -> None:
        for name, value, minimum in (
            ('n_folds', self.n_folds, 2), ('test_bars', self.test_bars, 1),
            ('purge_bars', self.purge_bars, 0), ('embargo_bars', self.embargo_bars, 0),
        ):
            if type(value) is not int or value < minimum:
                raise ValueError(f'split_walk_forward {name} must be a literal integer >= {minimum}')
        if type(self.anchored) is not bool:
            raise ValueError('split_walk_forward anchored must be a literal boolean')

    def as_dict(self) -> dict[str, int | bool]:
        return {
            'n_folds': self.n_folds, 'test_bars': self.test_bars,
            'purge_bars': self.purge_bars, 'embargo_bars': self.embargo_bars,
            'anchored': self.anchored,
        }


def read_walk_forward_config(value: object) -> WalkForwardConfig:
    if not isinstance(value, Mapping):
        raise ValueError('split_walk_forward must be a mapping')
    config = cast(Mapping[str, object], value)
    if set(config) != {'n_folds', 'test_bars', 'purge_bars', 'embargo_bars', 'anchored'}:
        raise ValueError('split_walk_forward requires exactly n_folds, test_bars, purge_bars, embargo_bars and anchored')
    for name in ('n_folds', 'test_bars', 'purge_bars', 'embargo_bars'):
        if type(config[name]) is not int:
            raise ValueError(f'split_walk_forward {name} must be a literal integer')
    if type(config['anchored']) is not bool:
        raise ValueError('split_walk_forward anchored must be a literal boolean')
    return WalkForwardConfig(
        n_folds=cast(int, config['n_folds']), test_bars=cast(int, config['test_bars']),
        purge_bars=cast(int, config['purge_bars']), embargo_bars=cast(int, config['embargo_bars']),
        anchored=config['anchored'],
    )


def validate_fold_splits(splits: list[pl.DataFrame], *, require_validation: bool) -> None:
    if splits[0].is_empty():
        raise ValueError('split_walk_forward fold has no fit rows')
    if require_validation and splits[1].is_empty():
        raise ValueError('split_walk_forward fold requires non-empty validation rows')
    if splits[2].is_empty():
        raise ValueError('split_walk_forward fold has no test rows')


def resolve_walk_forward_split(
    raw_data: pl.DataFrame, config: WalkForwardConfig, fold: int | None,
    ratios: tuple[int, int, int], *, require_validation: bool,
) -> list[pl.DataFrame]:
    if not isinstance(fold, int) or isinstance(fold, bool) or not 0 <= fold < config.n_folds:
        raise ValueError('split_walk_forward requires a valid fold selected by the experiment loop')
    if 'datetime' not in raw_data.columns:
        raise ValueError('split_walk_forward requires a datetime column')
    datetimes = raw_data.get_column('datetime')
    if datetimes.null_count() or datetimes.n_unique() != raw_data.height or not datetimes.is_sorted():
        raise ValueError('split_walk_forward requires unique, strictly increasing datetimes without nulls')
    folds = split_walk_forward(
        raw_data, n_folds=config.n_folds, test_bars=config.test_bars,
        purge_bars=config.purge_bars, embargo_bars=config.embargo_bars, anchored=config.anchored,
    )
    pool, test = folds[fold]
    train_ratio, val_ratio, _ = ratios
    if train_ratio <= 0 or val_ratio < 0:
        raise ValueError('split_walk_forward requires a positive train ratio and non-negative validation ratio')
    validation_start = pool.height * train_ratio // (train_ratio + val_ratio)
    validation = pool.slice(validation_start)
    fit_end = validation_start - config.purge_bars if validation.height else validation_start
    splits = [pool.slice(0, max(0, fit_end)), validation, test]
    validate_fold_splits(splits, require_validation=require_validation)
    return splits
