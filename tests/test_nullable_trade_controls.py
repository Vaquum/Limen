from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from limen.experiment._resolve_trade_policy import BacktestConfig, ProductConfig, TRADE_NUMBERS, resolve_trade_policy
from limen.experiment.manifest_core import MLManifest
from limen.targets import IdentityTarget
from limen.yaml._backtest_spec import check_backtest_spec


OPTIONAL = ('max_holding_seconds', 'timer_interval_seconds', 'max_price_gap_seconds')
PRODUCT = ProductConfig('cash_spot', 'BTCUSDT', 'BTC', 'USDT', 1e-9, 0)


@pytest.mark.parametrize('field', OPTIONAL)
def test_nullable_search_agrees_with_validation(field):
    errors = []
    check_backtest_spec({'sfd': {'manifest': {'backtest': {'product': asdict(PRODUCT), field: '{candidate}'}}, 'params': {'candidate': [None, 1800]}}}, errors)
    assert not errors
    config = BacktestConfig(product=PRODUCT)
    for value in (None, 1800):
        expected = resolve_trade_policy(replace(config, **{field: value}), {})
        for reference in ('candidate', '{candidate}'):
            assert resolve_trade_policy(replace(config, **{field: reference}), {'candidate': value}) == expected


def target_exposure(data):
    return {'_preds': np.full(len(data['x_test']), 0.5), '_prediction_mode': 'target_exposure'}


@pytest.mark.parametrize('field', OPTIONAL)
def test_nullable_round_matches_disabled_execution(field):
    source = pl.read_parquet(Path(__file__).parent / 'fixtures/spot_15m_20250101_20250531.parquet').head(120)
    manifest = MLManifest().set_data_source(lambda: source, params={'klines_size': 900})
    manifest.set_split_config(6, 2, 2).with_target_label('close', IdentityTarget).with_reference_architecture(target_exposure)
    manifest.set_backtest_config(prediction_mode='target_exposure', product=PRODUCT)
    literal = manifest.prepare_data(source, {})
    baseline = manifest.run_model(literal, {})
    setattr(manifest.backtest_config, field, '{candidate}')
    params = {'candidate': None}
    searched = manifest.prepare_data(source, params)
    result = manifest.run_model(searched, params)
    assert searched['trade_contract_digest'] == literal['trade_contract_digest']
    assert searched['_trade_ledger'].fills.equals(literal['_trade_ledger'].fills)
    assert result['backtest_open_trades'] == baseline['backtest_open_trades']


@pytest.mark.parametrize('field', OPTIONAL)
@pytest.mark.parametrize('value', (0, -1, float('inf'), float('nan'), True))
def test_nullable_controls_reject_invalid_candidates(field, value):
    with pytest.raises(ValueError):
        resolve_trade_policy(replace(BacktestConfig(product=PRODUCT), **{field: '{candidate}'}), {'candidate': value})


@pytest.mark.parametrize('field', tuple(name for name in TRADE_NUMBERS if name not in OPTIONAL))
def test_required_controls_reject_null_candidates(field):
    with pytest.raises(ValueError, match='finite number'):
        resolve_trade_policy(replace(BacktestConfig(product=PRODUCT), **{field: '{candidate}'}), {'candidate': None})
