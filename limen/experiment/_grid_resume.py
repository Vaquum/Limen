from collections.abc import Mapping
from typing import cast
from pathlib import Path

import polars as pl

from limen.experiment.param_domain import ParamDomain
from limen.experiment.param_search.grid_strategy import GridStrategy
from limen.experiment.param_search._grid_permutation import _SHUFFLE_VERSION, domain_digest
from limen.experiment.param_search.search_strategy import SearchStrategy


def add_grid_metadata(strategy: SearchStrategy | None, metadata: dict[str, object]) -> None:
    if isinstance(strategy, GridStrategy) and strategy.get_state().get('permutation_version') == _SHUFFLE_VERSION:
        metadata['grid_shuffle_version'] = _SHUFFLE_VERSION


def validate_grid_resume(strategy: SearchStrategy, domain: ParamDomain,
                         checkpoint: Mapping[str, object], metadata: Mapping[str, object],
                         *, factorize: bool) -> None:
    if not isinstance(strategy, GridStrategy):
        return
    msq_state = cast(Mapping[str, object], checkpoint['msq_state'])
    saved = cast(Mapping[str, object], msq_state['strategy_state'])
    current = strategy.get_state()
    version = saved.get('permutation_version', 1 if saved.get('shuffle') else None)
    recorded = metadata.get('grid_shuffle_version')
    if recorded is not None or version == _SHUFFLE_VERSION or (factorize and current.get('permutation_version') == _SHUFFLE_VERSION):
        if type(recorded) is not int or recorded != _SHUFFLE_VERSION or type(version) is not int or version != _SHUFFLE_VERSION or saved.get('shuffle') is not True:
            raise ValueError('Cannot resume: grid shuffle metadata and checkpoint version disagree')
        if current.get('permutation_version') != _SHUFFLE_VERSION or current['seed'] != saved.get('seed'):
            raise ValueError('Cannot resume with a different grid shuffle seed or version')
    if factorize:
        if saved.get('shuffle') and (version != _SHUFFLE_VERSION or current.get('permutation_version') != _SHUFFLE_VERSION):
            raise ValueError('factorize requires version 2 seeded grid shuffle')
        saved_domain = cast(Mapping[str, list[object]], checkpoint['domain_state'])
        if domain_digest(domain.params) != domain_digest(saved_domain):
            raise ValueError('Cannot resume factorize with a different ordered parameter domain')
        if version == _SHUFFLE_VERSION and saved.get('domain_digest') != domain_digest(domain.params):
            raise ValueError('Cannot resume: grid shuffle domain digest disagrees with declaration')


def validate_factorized_grid(strategy: SearchStrategy) -> None:
    state = strategy.get_state()
    if state.get('shuffle') and (state.get('permutation_version') != _SHUFFLE_VERSION or type(state.get('seed')) is not int):
        raise ValueError('factorize requires version 2 seeded grid shuffle after restore')


def complete_failed_header(path: Path, columns: list[str] | None, keys: list[str],
                           log: pl.DataFrame | None) -> list[str] | None:
    if columns and set(keys) - set(columns) and log is not None and log['strict_mode_error'].null_count() == 0:
        previous = pl.read_csv(path, infer_schema=False)
        columns = list(dict.fromkeys([*keys, *columns]))
        previous.select(pl.col(key) if key in previous.columns else pl.lit(None).alias(key) for key in columns).write_csv(path)
    return columns
