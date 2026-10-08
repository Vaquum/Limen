from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from limen.backtest.funding_adapter import prepare_funding
from limen.backtest.funding_presets import PRESETS
from limen.backtest.trade_contract import NANOSECONDS, contract_digest, export_trade_contract, source_binding
from limen.backtest.trade_execution import trade_execution
from limen.experiment._resolve_trade_policy import BacktestConfig, FundingConfig, ProductConfig, resolve_funding, resolve_trade_policy
from limen.experiment.manifest_core import DataSourceConfig, MLManifest
from limen.sfd.reference_architecture._backtest_evaluation import compute_backtest
from tests.test_target_exposure import _case



def _funded(inputs, policy, *, mechanism='discrete', interval=3600.0, rate=0.001):
    params = {'mechanism': mechanism, 'rate': rate, 'rate_unit': 'decimal', 'rate_basis_seconds': interval,
              'settlement_interval_seconds': interval, 'settlement_phase_utc_seconds': inputs.partition_start_ns / NANOSECONDS,
              'cash_settlement_interval_seconds': interval, 'cash_settlement_phase_utc_seconds': inputs.partition_start_ns / NANOSECONDS,
              'valuation': 'execution_proxy', 'currency': 'USDT', 'approximation': 'scenario'}
    funding = resolve_funding(FundingConfig(params=params), {})
    assert funding is not None
    events = prepare_funding(funding, None, inputs.partition_start_ns, inputs.partition_end_ns)
    binding = source_binding(events, 'declared scenario funding', inputs.partition_start_ns, inputs.partition_end_ns, 1, 'scenario funding support')
    return replace(inputs, funding_events=events, sources=(*inputs.sources, binding)), replace(policy, funding=funding)


def test_preset_mechanics_and_rate_override():
    for name, preset in PRESETS.items():
        assert preset.calibration['status'] == 'scenario_assumption'
        assert preset.calibration['window_start_utc'] == '2025-10-08T00:00:00Z'
        assert preset.calibration['window_end_utc'] == '2026-10-08T00:00:00Z'
        default = resolve_funding(FundingConfig(preset=name), {})
        assert default is not None and default.approximation == 'scenario'
        assert default.params['rate'] == pytest.approx(0.0000035 if name == 'hyperliquid_btc' else 0.000028)
        assert default.rate_basis_seconds == (3600 if name == 'hyperliquid_btc' else 28800)
        assert default.calibration == preset.calibration
        events = prepare_funding(default, None, 0, 86400 * NANOSECONDS)
        assert events.filter(pl.col('kind') != 'cash_settlement')['rate_decimal'].unique().to_list() == [default.params['rate']]
        policy = resolve_funding(FundingConfig(preset=name, params={'rate': '{scenario_rate}'}), {'scenario_rate': -0.0002})
        assert policy is not None and policy.params['rate'] == -0.0002
        assert policy.calibration == preset.calibration
    with pytest.raises(ValueError, match='Unknown funding parameters'):
        resolve_funding(FundingConfig(preset='binance_btcusdt', params={'typo_rate': 0.1}), {})


def test_history_units_valuation_and_coverage():
    # Explicit scenario rates exercise ingestion; market times/valuations are
    # unchanged recorded observations, not claimed historical funding quotes.
    inputs, trade = _case([0.5, 0.5, 0.5])
    times = inputs.observations['available_at_ns'].to_list()
    history = pl.DataFrame({'event_id': ['scenario:0', 'scenario:1', 'scenario:2'], 'settlement_at': times,
        'rate_decimal': [0.001] * 3, 'valuation_price': inputs.observations['open'],
        'schedule_start': times, 'schedule_end': [*times[1:], times[-1] + 1],
        'settlement_interval_seconds': [86400.0] * 3,
        'settlement_phase_utc_seconds': [time / NANOSECONDS for time in times]})
    config = FundingConfig(preset='binance_btcusdt', data_source=DataSourceConfig(lambda: history))
    policy = resolve_funding(config, {})
    assert policy is not None and 'rate' not in policy.params
    events = prepare_funding(policy, history, times[0], times[-1])
    assert events['rate_decimal'].to_list() == history['rate_decimal'].to_list()
    assert events['valuation_price'].to_list() == history['valuation_price'].to_list()
    with pytest.raises(ValueError, match='conflicts'):
        resolve_funding(replace(config, params={'rate': 0.001}), {})
    with pytest.raises(ValueError, match=r'coverage|gap'):
        prepare_funding(policy, history[[0, 2]], times[0], times[-1])
    with pytest.raises(ValueError, match='valuation'):
        prepare_funding(policy, history.with_columns(pl.lit(None).alias('valuation_price')), times[0], times[-1])
    inputs, trade = _case([0.5, 0.5])
    funded, resolved = _funded(inputs, trade, rate=10.0)
    decimal = trade_execution(funded, resolved).metrics['funding_pnl']
    assert resolved.funding is not None
    bps_policy = replace(resolved.funding, params={**resolved.funding.params, 'rate': 100000.0, 'rate_unit': 'bps'})
    bps_events = prepare_funding(bps_policy, None, inputs.partition_start_ns, inputs.partition_end_ns)
    assert bps_events['rate_decimal'].equals(funded.funding_events['rate_decimal'])
    assert decimal == 0.0  # no settlement after entry in this recorded window


def test_signed_inventory_intervals_and_schedules():
    for side in (1, -1):
        inputs, policy = _case([side * 0.5, side * 0.5, side * 0.75, 0.0], fee_bps=0.0, slip_bps=0.0)
        funded, policy = _funded(inputs, policy)
        result = trade_execution(funded, policy)
        quantity = side * np.floor(0.5 * 10000 / inputs.observations['open'][0] / 1e-12) * 1e-12
        # At the 01:00 boundary the first entry is still held. The 00:49:40
        # recorded observation is the latest causal scenario valuation.
        first_held = result.funding.filter(pl.col('quantity') != 0).row(0, named=True)
        expected = -quantity * inputs.observations['open'][1] * 0.001
        assert first_held['recognized_delta'] == pytest.approx(expected)
        assert result.states['quantity'][0] == result.states['quantity'][1]
        assert result.fills.height == 3
        assert result.funding['recognized_delta'].sum() == pytest.approx(result.metrics['funding_pnl'])
        assert np.sign(result.metrics['funding_pnl']) == -side
    inputs, policy = _case([0.5, 0.5, 0.0], fee_bps=0.0, slip_bps=0.0)
    funded, policy = _funded(inputs, policy, mechanism='continuous')
    result = trade_execution(funded, policy)
    q = result.fills['quantity_delta'][0]
    prices, times = inputs.observations['open'].to_list(), inputs.observations['available_at_ns'].to_list()
    expected = -q * 0.001 * sum(price * (right - left) / NANOSECONDS / 3600 for price, left, right in zip(prices, times, times[1:], strict=False))
    assert result.metrics['funding_pnl'] == pytest.approx(expected)
    assert result.metrics['ending_equity'] == pytest.approx(10000 + q * (prices[2] - prices[0]) + expected)
    transfers = result.funding.filter(pl.col('cash_delta') != 0)
    assert transfers['recognized_delta'].sum() == 0.0


def _sized_native(data, exposure=0.5):
    return {'_preds': np.full(len(data['x_test']), exposure), '_prediction_mode': 'target_exposure'}


def test_context_is_shared_and_private():
    source = pl.read_parquet(Path(__file__).parent / 'fixtures/spot_15m_20250101_20250531.parquet').head(60)
    calls = []

    def execution_source():
        calls.append('execution')
        from limen.experiment._prepare_trade_context import normalize_observations
        return normalize_observations(source, interval_seconds=900.0)

    from limen.targets import IdentityTarget
    manifest = MLManifest().set_data_source(lambda: source, params={'klines_size': 900.0})
    manifest.set_split_config(6, 2, 2)
    manifest.with_target_label(target_name='close', target_class=IdentityTarget)
    manifest.with_reference_architecture(_sized_native)
    product = ProductConfig('linear_perpetual', 'BTCUSDT', 'BTC', 'USDT', 1e-9, 0.0)
    funding = FundingConfig(preset='binance_btcusdt', params={'rate': 0.0001})
    manifest.set_backtest_config(prediction_mode='target_exposure', product=product, funding=funding, execution_data_source=DataSourceConfig(execution_source))
    data = manifest.prepare_data(source, {})
    assert calls == ['execution']
    result = manifest.run_model(data, {})
    assert data['trade_contract_digest'] == contract_digest(export_trade_contract(data['_trade_policy'], data['_trade_inputs']))
    inline = compute_backtest(result['_preds'], data)
    assert all(result[key] == pytest.approx(value) for key, value in inline.items())
    assert not any('funding' in col or col.endswith('_ns') for col in data['x_train'].columns)
    from types import SimpleNamespace
    from limen.log._snapshot_backtest_round import snapshot_backtest_round
    log = SimpleNamespace(manifest=manifest, data=source, round_params=[{}], preds=[result['_preds']], _alignment=[data['_alignment']])
    replay = snapshot_backtest_round(log, 0, lambda frame: pytest.fail('signed replay must preserve sizing'))
    assert all(result[f'backtest_{key}'] == pytest.approx(value) for key, value in replay.items())
    assert calls == ['execution', 'execution']
    manifest.backtest_config = BacktestConfig()
    disabled = manifest.prepare_data(source, {})
    assert '_trade_inputs' not in disabled
    assert calls == ['execution', 'execution']


def test_rule_based_splits_use_event_economics():
    from limen.experiment.manifest_core import RuleBasedManifest
    from limen.sfd.reference_architecture.rule_based import rule_based

    source = pl.read_parquet(Path(__file__).parent / 'fixtures/spot_15m_20250101_20250531.parquet').head(80)
    manifest = RuleBasedManifest().set_data_source(lambda: source, params={'klines_size': 900})
    manifest.set_split_config(6, 2, 2)
    manifest.with_strategy([{'id': 'entry', 'type': 'threshold', 'column': 'close', 'operator': '>', 'value': 0}], entry='entry')
    manifest.with_reference_architecture(rule_based)
    manifest.set_backtest_config(product=ProductConfig('linear_perpetual', 'BTCUSDT', 'BTC', 'USDT', 1e-9, 0), funding=FundingConfig(preset='binance_btcusdt', params={'rate': 0.0001}), max_holding_seconds=1800)
    data = manifest.prepare_data(source, {})
    result = manifest.run_model(data, {})
    for split in ('train', 'val', 'test'):
        assert result[f'completed_trades_{split}'] == 1
        assert result[f'num_executed_trades_{split}'] == 1
        assert result[f'funding_pnl_{split}'] <= 0
    assert result['ending_equity_test'] == pytest.approx(result['backtest_ending_equity'])


def test_new_options_are_resolved_or_rejected():
    assert resolve_trade_policy(BacktestConfig(), {}) is None
    for option in ({'execution_lag_seconds': 1}, {'max_price_gap_seconds': 1}, {'flat_threshold': 0.01}):
        with pytest.raises(ValueError, match='product metadata'):
            resolve_trade_policy(BacktestConfig(**option), {})
    product = ProductConfig('linear_perpetual', 'BTCUSDT', 'BTC', 'USDT', '{step}', 0)
    policy = resolve_trade_policy(BacktestConfig(product=product, max_holding_seconds='{holding}', fee_bps='{fee}'), {'step': 0.001, 'holding': 120, 'fee': 2})
    assert policy is not None and policy.max_holding_seconds == 120 and policy.fee_bps == 2
    assert policy.product.quantity_step == 0.001
    from limen.yaml._backtest_spec import check_backtest_spec
    errors = []
    check_backtest_spec({'sfd': {'manifest': {'backtest': {'prediction_mode': 'target_exposure', 'product': {'kind': 'invalid'}, 'max_holding_seconds': '{holding}', 'execution_data_source': {'method': 'fetch', 'ignored': True}}}, 'params': {'holding': [-1, float('inf')]}}}, errors)
    assert len(errors) >= 4
