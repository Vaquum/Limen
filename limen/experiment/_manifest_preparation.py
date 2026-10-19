from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Protocol, TypeVar, cast
import polars as pl

from limen.backtest.trade_contract import TradePolicy
from limen.experiment._prepare_trade_context import FrozenTradeSources, PreparedTradeContext, prepare_trade_context, source_interval
from limen.experiment._prepare_backtest_data import prepare_backtest_data
from limen.experiment._resolve_backtest_config import configured_barriers
from limen.experiment._resolve_trade_policy import BacktestConfig, SourceConfig

_M = TypeVar('_M')


class SetupManifest(Protocol):
    @property
    def backtest_config(self) -> BacktestConfig | None: ...
    @property
    def data_source_config(self) -> SourceConfig | None: ...
    def resolve_trade_policy(self, round_params: Mapping[str, object]) -> TradePolicy | None: ...


def finish_prepare_setup(manifest: SetupManifest, raw_splits: list[pl.DataFrame],
                         pairs: list[tuple[list[datetime] | list[int], pl.DataFrame]],
                         round_params: dict[str, object]) -> tuple[list[pl.DataFrame], list[datetime] | list[int], pl.DataFrame | None, list[pl.DataFrame] | None, PreparedTradeContext | None]:
    all_datetimes = cast(list[datetime] | list[int], [dt for datetimes, _ in pairs for dt in datetimes])
    splits = [frame for _, frame in pairs]
    policy = manifest.resolve_trade_policy(round_params)
    frozen = cast(FrozenTradeSources | None, getattr(manifest, '_frozen_trade_sources', None))
    if callable(frozen):
        frozen = frozen(manifest.backtest_config, round_params)
    trade = None
    if policy is not None and manifest.backtest_config is not None:
        interval = source_interval(manifest.data_source_config)
        trade = prepare_trade_context(manifest.backtest_config, policy, raw_splits, splits, round_params, interval_seconds=interval, frozen_sources=frozen)
        splits = [split.drop([name for name in ('start_ns', 'end_ns', 'open_available_at_ns', 'available_at_ns') if name in split.columns]) for split in splits]
    prepared = prepare_backtest_data(splits, all_datetimes, configured=policy is None and configured_barriers(manifest.backtest_config))
    return (*prepared, trade)


def finish_sensor_input(manifest: _M, data: pl.DataFrame, lookback: int,
                        params: dict[str, object], fitted: dict[str, object], decisions: pl.DataFrame | None, *,
                        scale: Callable[[_M, pl.DataFrame, dict[str, object], dict[str, object], bool], tuple[pl.DataFrame, dict[str, object]]],
                        compress: Callable[[_M, pl.DataFrame, dict[str, object], dict[str, object]], pl.DataFrame]) -> tuple[pl.DataFrame, int]:
    data, _ = scale(manifest, data, params, fitted, False)
    data = data.fill_nan(None)
    data = compress(manifest, data, params, fitted)
    if decisions is not None:
        data = data.join(decisions, on='datetime', how='left', maintain_order='left')
    return data, lookback
