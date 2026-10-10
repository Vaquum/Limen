from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from limen.backtest.execution_events import execution_events, with_predictions
from limen.backtest.funding_presets import BASELINE_RATE_8H
from limen.backtest.trade_contract import NANOSECONDS
from limen.backtest.trade_execution import trade_execution
from limen.calibration import grid_threshold_optimizer
from limen.data import HistoricalData
from limen.experiment._objective import prepare_objective, score_objective
from limen.experiment._resolve_trade_policy import FundingConfig, ProductConfig
from limen.experiment.manifest_core import DataSourceConfig, MLManifest
from limen.sfd.reference_architecture.logreg_binary import logreg_binary
from limen.targets import NextBarUpTarget


def _funded_validation(interpretation):
    recorded = pl.read_parquet(Path(__file__).parent / 'fixtures/spot_15m_20250101_20250531.parquet').head(128)
    anchors = recorded[[0, 32, 64, 96]]
    # Funding rates are declared scenario assumptions; clocks and valuations
    # remain recorded market observations, not invented historical quotes.
    history = pl.DataFrame({
        'event_id': ['scenario:0', 'scenario:1', 'scenario:2'],
        'start': anchors['datetime'][:-1], 'end': anchors['datetime'][1:],
        'valuation_price': anchors['open'][:-1],
        'rate_decimal': [BASELINE_RATE_8H] * 3,
        'rate_basis_seconds': [28800.0] * 3,
    })
    funding = FundingConfig(data_source=DataSourceConfig(lambda: history), params={
        'mechanism': 'continuous', 'rate_unit': 'decimal', 'rate_basis_seconds': 28800.0,
        'cash_settlement_interval_seconds': 86400.0, 'valuation': 'mark',
        'currency': 'USDT', 'approximation': 'scenario', 'history_interpretation': interpretation,
    })
    source = recorded.head(60)
    manifest = MLManifest().set_data_source(lambda: source, params={'kline_size': 900})
    manifest.set_split_config(6, 2, 2).with_target_label('next_up', NextBarUpTarget)
    manifest.with_reference_architecture(logreg_binary).set_objective()
    manifest.set_backtest_config(product=ProductConfig('linear_perpetual', 'BTCUSDT', 'BTC', 'USDT', 1e-9, 0.0),
                                 fee_bps=0.0, slip_bps=0.0, funding=funding)
    return manifest, manifest.prepare_data(source, {})


def test_quoted_funding_covers_partial_validation_interval():
    manifest, data = _funded_validation('quoted')
    original = data['_trade_context'].partitions[1]
    assert original.funding_events.filter((pl.col('kind') == 'accrual') & (pl.col('start_ns') < original.partition_end_ns)
                                         & (pl.col('end_ns') > original.partition_end_ns)).height == 1
    scorer = prepare_objective(data, manifest.architecture_function, manifest.prediction_calibration_config)
    accrual = scorer.inputs.funding_events.filter(pl.col('kind') == 'accrual')
    assert accrual['start_ns'].min() == original.partition_start_ns
    assert accrual['end_ns'].max() == original.partition_end_ns
    assert accrual['rate_basis_seconds'].to_list() == [28800.0]
    assert original.funding_events['end_ns'].max() > original.partition_end_ns
    predictions = np.ones(scorer.inputs.signals.height, dtype=np.int8)
    ledger = trade_execution(with_predictions(scorer.inputs, predictions), scorer.policy)
    quantity = ledger.fills['quantity_delta'][0]
    first_fill = ledger.fills['time_ns'][0]
    expected = -quantity * accrual['valuation_price'][0] * BASELINE_RATE_8H * (original.partition_end_ns - first_fill) / NANOSECONDS / 28800.0
    assert ledger.metrics['funding_pnl'] == pytest.approx(expected)
    assert scorer(data['y_val'], predictions) == ledger.metrics['total_return']
    assert ledger.funding['end_ns'].max() == original.partition_end_ns
    assert scorer.inputs.sources[-1].checksum != original.sources[-1].checksum


def test_realized_integrated_funding_cannot_borrow_test_interval():
    manifest, data = _funded_validation('integrated')
    with pytest.raises(ValueError, match='held-out evidence'):
        prepare_objective(data, manifest.architecture_function, manifest.prediction_calibration_config)


@pytest.mark.parametrize('execution_interval', (900, 3600))
@pytest.mark.parametrize('direction', ('maximize', 'minimize'))
def test_validation_keeps_causal_interval_open_without_future_ohlc(monkeypatch, execution_interval, direction):
    recorded = pl.read_parquet(Path(__file__).parent / 'fixtures/spot_15m_20250101_20250531.parquet').head(128)
    monkeypatch.setattr('limen.data.historical_data._read_any_file', lambda *args, **kwargs: recorded)
    source = recorded.head(59)
    manifest = MLManifest().set_data_source(lambda: source, params={'kline_size': 900})
    manifest.set_split_config(6, 2, 2).with_target_label('next_up', NextBarUpTarget)
    manifest.with_reference_architecture(logreg_binary).set_objective(direction=direction)
    manifest.with_calibration().threshold_function(grid_threshold_optimizer, threshold_min=0.2,
                                                   threshold_max=0.8, threshold_step=0.2).done()
    manifest.set_backtest_config(product=ProductConfig('cash_spot', 'BTCUSDT', 'BTC', 'USDT', 1e-9, 0.0),
                                 fee_bps=0.0, slip_bps=0.0,
                                 execution_data_source=DataSourceConfig(HistoricalData.get_spot_klines,
                                                                      {'kline_size': execution_interval}))
    data = manifest.prepare_data(source, {})
    context = data['_trade_context']
    validation = context.partitions[1]
    end = validation.partition_end_ns
    scorer = prepare_objective(data, manifest.architecture_function, manifest.prediction_calibration_config)
    straddling = scorer.inputs.observations.filter(pl.col('available_at_ns') > end)
    held_out = validation.observations.filter(pl.col('start_ns') >= end)
    assert scorer.inputs.observations.filter(pl.col('start_ns') >= end).is_empty()
    if execution_interval == 3600:
        assert straddling.height == 1
        interval = straddling.row(0, named=True)
        events = [event for event in execution_events(scorer.inputs, scorer.policy)
                  if event.source_row_id == interval['row_id']]
        assert [(event.time_ns, event.observation_phase) for event in events] == [(interval['start_ns'], 'open')]
        replacement = recorded.filter(recorded['datetime'].dt.epoch('ns') == interval['start_ns']).row(0, named=True)
        assert replacement['open'] == interval['open']
        assert any(replacement[key] != interval[key] for key in ('high', 'low', 'close'))
    else:
        assert held_out.height == 1
        assert straddling.is_empty()
        replacement = recorded.row(0, named=True)
    # Substitute actual recorded OHLC for the unavailable close and test-owned
    # row. The interval's causal opening price and all clocks stay unchanged.
    changed_observations = validation.observations.with_columns(*[
        pl.when(pl.col('start_ns') >= end).then(recorded[key][0])
        .when((pl.col('available_at_ns') > end) & pl.lit(key != 'open')).then(replacement[key])
        .otherwise(pl.col(key)).alias(key)
        for key in ('open', 'high', 'low', 'close')
    ])
    changed = dict(data)
    changed['_trade_context'] = replace(context, partitions=(context.partitions[0],
                                                           replace(validation, observations=changed_observations),
                                                           context.partitions[2]))
    changed_scorer = prepare_objective(changed, manifest.architecture_function, manifest.prediction_calibration_config)
    assert changed_scorer.inputs.observations.filter(pl.col('start_ns') >= end).is_empty()
    predictions = np.ones(scorer.inputs.signals.height, dtype=np.int8)
    baseline = trade_execution(with_predictions(scorer.inputs, predictions), scorer.policy)
    perturbed = trade_execution(with_predictions(changed_scorer.inputs, predictions), changed_scorer.policy)
    assert baseline.fills.height > 0
    assert baseline.metrics == perturbed.metrics
    result = manifest.run_model(data, {'solver': 'liblinear', 'max_iter': 1000})
    probabilities = result['_model'].predict({'x_test': data['x_val']})['_probs']
    selected = grid_threshold_optimizer(data['y_val'], probabilities, threshold_min=0.2,
                                       threshold_max=0.8, threshold_step=0.2,
                                       metric=changed_scorer, _objective_maximize=direction == 'maximize')
    assert selected == (result['optimal_threshold'], result['val_backtest_total_return'])
    assert score_objective(changed, result, changed_scorer) == result['val_backtest_total_return']
    assert data['_trade_context'] is context
    assert data['_trade_inputs'] is context.partitions[2]
    if execution_interval == 3600:
        unavailable = validation.observations.with_columns(
            pl.when(pl.col('available_at_ns') > end).then(pl.col('available_at_ns'))
            .otherwise(pl.col('open_available_at_ns')).alias('open_available_at_ns')
        )
        changed['_trade_context'] = replace(context, partitions=(context.partitions[0],
                                                               replace(validation, observations=unavailable),
                                                               context.partitions[2]))
        with pytest.raises(ValueError, match='does not cover'):
            prepare_objective(changed, manifest.architecture_function, manifest.prediction_calibration_config)
