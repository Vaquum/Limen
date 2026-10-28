from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Protocol, cast

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

    def permutation_prediction_performance(self, round_id: int) -> pl.DataFrame: ...


def snapshot_backtest_round(
    log: _ReplayLog, round_id: int, normalize: Callable[[pl.DataFrame], pl.DataFrame],
) -> dict[str, float]:
    from limen.experiment._backtest_provenance import replay_prices as _replay_prices, validate_witness as _validate_witness
    from limen.experiment._resolve_backtest_config import configured_barriers as _configured_barriers
    from limen.sfd.reference_architecture._backtest_evaluation import execution_options as _execution_options

    from limen.experiment._exit_rules import snapshot_exits, verify_replay

    manifest = cast(_ReplayManifest | None, getattr(log, 'manifest', None))
    alignment_rows = cast(list[dict[str, object]], getattr(log, '_alignment', []))
    if round_id < len(alignment_rows) and 'trade_contract' in alignment_rows[round_id]:
        from limen.backtest.execution_events import with_predictions
        from limen.backtest.trade_contract import JsonValue, TradeInputs, TradePolicy, contract_digest, export_trade_contract
        from limen.backtest.trade_execution import trade_execution

        alignment = alignment_rows[round_id]
        inputs, policy = alignment.get('_trade_inputs'), alignment.get('_trade_policy')
        if manifest is None or not isinstance(inputs, TradeInputs) or not isinstance(policy, TradePolicy):
            raise ValueError('File-only/resumed configured Log replay requires original bound source context')
        fresh = manifest.prepare_data(log.data, dict(log.round_params[round_id]))
        verify_replay(alignment, fresh)
        digest = contract_digest(export_trade_contract(policy, inputs))
        if digest != alignment.get('trade_contract_digest') or digest != fresh.get('trade_contract_digest') or digest != contract_digest(cast(Mapping[str, JsonValue], alignment['trade_contract'])):
            raise ValueError('Trade rules or source identity changed during replay')
        return dict(trade_execution(with_predictions(inputs, log.preds[round_id]), policy).metrics)
    from limen.experiment._dynamic_barriers import dynamic_barriers
    if manifest is not None and dynamic_barriers(manifest.backtest_config):
        raise ValueError('Dynamic Log replay requires original bound distance context')
    resolved = manifest.resolve_backtest_config(log.round_params[round_id]) if manifest is not None else {}
    prepared: dict[str, object] = {}
    exits_enabled = getattr(manifest, 'exit_rules_config', None) is not None or (round_id < len(alignment_rows) and 'exit_rules' in alignment_rows[round_id])
    if manifest is None and exits_enabled:
        raise ValueError('Configured exit replay requires original bound source context')
    if manifest is not None and (_configured_barriers(manifest.backtest_config) or exits_enabled):
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
        if exits_enabled:
            verify_replay(alignment_rows[round_id], prepared)
        fresh = _validate_witness(prepared.get('_backtest_provenance'))
        if fresh != witness:
            raise ValueError('backtest source identity changed during replay preparation')
        perf = pl.DataFrame({'predictions': predictions, 'actuals': prepared.get('y_test', [None] * len(cast(list[object], predictions)))}).with_columns(
            prices.select('open', 'high', 'low', 'close'),
        ).with_columns((pl.col('close') - pl.col('open')).alias('price_change'))
    else:
        perf = log.permutation_prediction_performance(round_id)
    perf = normalize(perf)
    columns = {col: perf[col].to_numpy() for col in ('predictions', 'open', 'high', 'low', 'close', 'price_change') if col in perf.columns}
    if exits_enabled:
        columns['exit'] = pl.Series(snapshot_exits(prepared)).to_numpy()
    return backtest_snapshot(columns, execution_lag_bars=1, exit_col='exit' if exits_enabled else None, **_execution_options(resolved))


__all__ = ['snapshot_backtest_round']
