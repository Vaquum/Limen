from pathlib import Path

import numpy as np
import polars as pl
import pytest

from limen.backtest.execution_events import with_predictions
from limen.backtest.funding_presets import BASELINE_RATE_8H
from limen.backtest.trade_contract import NANOSECONDS
from limen.backtest.trade_execution import trade_execution
from limen.experiment._objective import prepare_objective
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
