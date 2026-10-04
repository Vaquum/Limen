from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Protocol, cast

import pandas as pd
import polars as pl

from limen.backtest.backtest_snapshot import backtest_snapshot

if TYPE_CHECKING:
    from limen.experiment._resolve_backtest_config import BacktestConfig as _BacktestConfig


class _ReplayManifest(Protocol):
    backtest_config: _BacktestConfig | None

    def resolve_backtest_config(self, round_params: Mapping[str, object]) -> dict[str, float | None]: ...
    def compute_test_bars(self, raw_data: pl.DataFrame, round_params: dict[str, object]) -> pl.DataFrame: ...
    def prepare_data(self, raw_data: pl.DataFrame, round_params: dict[str, object]) -> dict[str, object]: ...


class _ReplayLog(Protocol):
    data: pl.DataFrame
    round_params: list[dict[str, object]]
    preds: list[object]
    _alignment: list[dict[str, object]]

    def permutation_prediction_performance(self, round_id: int) -> pd.DataFrame: ...


def snapshot_backtest_round(
    log: _ReplayLog, round_id: int, normalize: Callable[[pd.DataFrame], pd.DataFrame],
) -> dict[str, float]:
    from limen.experiment._backtest_provenance import replay_prices as _replay_prices, validate_witness as _validate_witness
    from limen.experiment._resolve_backtest_config import configured_barriers as _configured_barriers
    from limen.sfd.reference_architecture._backtest_evaluation import execution_options as _execution_options

    manifest = cast(_ReplayManifest | None, getattr(log, 'manifest', None))
    resolved = manifest.resolve_backtest_config(log.round_params[round_id]) if manifest is not None else {}
    if manifest is not None and _configured_barriers(manifest.backtest_config):
        alignment = getattr(log, '_alignment', None)
        if not isinstance(alignment, list):
            raise ValueError('backtest provenance is required for configured replay')
        alignment_rows = cast(list[Mapping[str, object]], alignment)
        if len(alignment_rows) <= round_id:
            raise ValueError('backtest provenance is required for configured replay')
        witness = _validate_witness(alignment_rows[round_id].get('_backtest_provenance'))
        params = log.round_params[round_id]
        source = manifest.compute_test_bars(log.data, params)
        predictions = log.preds[round_id]
        prices = _replay_prices(source, witness, len(cast(list[object], predictions)))
        prepared = manifest.prepare_data(log.data, dict(params))
        fresh = _validate_witness(prepared.get('_backtest_provenance'))
        if fresh != witness:
            raise ValueError('backtest source identity changed during replay preparation')
        perf = pd.DataFrame({'predictions': predictions, 'actuals': prepared['y_test']})
        for col in ('open', 'high', 'low', 'close'):
            perf[col] = prices[col].to_numpy()
        perf['price_change'] = perf['close'] - perf['open']
    else:
        perf = log.permutation_prediction_performance(round_id)
    perf = normalize(perf)
    columns = {col: perf[col].to_numpy() for col in ('predictions', 'open', 'high', 'low', 'close', 'price_change') if col in perf}
    return backtest_snapshot(columns, execution_lag_bars=1, **_execution_options(resolved))


__all__ = ['snapshot_backtest_round']
