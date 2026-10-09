from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from ruamel.yaml import YAML

from limen.cli.commands.run import run_experiment
from limen.data import HistoricalData
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
        for reference in ('candidate', '{candidate}', lambda params: params['candidate']):
            assert resolve_trade_policy(replace(config, **{field: reference}), {'candidate': value}) == expected


def target_exposure(data):
    return {'_preds': np.full(len(data['x_test']), 0.5), '_prediction_mode': 'target_exposure'}


def recorded_yaml_source(klines_size=900, start_date_limit=None, end_date_limit=None):
    assert klines_size == 900
    return pl.read_parquet(Path(__file__).parent / 'fixtures/spot_15m_20250101_20250531.parquet').head(288)


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
    ids = ('fill_id', 'intent_id', 'episode_id')
    assert searched['_trade_ledger'].fills.drop(ids).equals(literal['_trade_ledger'].fills.drop(ids))
    assert result['backtest_open_trades'] == baseline['backtest_open_trades']
    if field == 'max_price_gap_seconds':
        with pytest.raises(ValueError, match='Execution price gap exceeds declared precision'):
            manifest.prepare_data(source, {'candidate': 600})
    elif field == 'max_holding_seconds':
        positive = manifest.prepare_data(source, {'candidate': 1800})
        assert manifest.run_model(positive, {'candidate': 1800})['backtest_open_trades'] == 0


@pytest.mark.parametrize('field', OPTIONAL)
@pytest.mark.parametrize('value', (0, -1, float('inf'), float('nan'), True, False, 'null'))
def test_nullable_controls_reject_invalid_candidates(field, value):
    with pytest.raises(ValueError):
        resolve_trade_policy(replace(BacktestConfig(product=PRODUCT), **{field: '{candidate}'}), {'candidate': value})


@pytest.mark.parametrize('field', TRADE_NUMBERS)
def test_required_controls_reject_null_candidates(field):
    errors = []
    check_backtest_spec({'sfd': {'manifest': {'backtest': {'product': asdict(PRODUCT), field: '{candidate}'}}, 'params': {'candidate': [None]}}}, errors)
    assert bool(errors) == (field not in OPTIONAL)
    config = replace(BacktestConfig(product=PRODUCT), **{field: '{candidate}'})
    if errors:
        with pytest.raises(ValueError, match='finite number'):
            resolve_trade_policy(config, {'candidate': None})
    else:
        assert getattr(resolve_trade_policy(config, {'candidate': None}), field) is None


@pytest.mark.parametrize('field', OPTIONAL)
@pytest.mark.parametrize('reference', ('missing', '{missing}'))
def test_nullable_controls_reject_missing_references(field, reference):
    with pytest.raises(ValueError):
        resolve_trade_policy(replace(BacktestConfig(product=PRODUCT), **{field: reference}), {})


@pytest.mark.parametrize('field', OPTIONAL)
@pytest.mark.parametrize('candidates', ([None, 1800], [1800, None]))
def test_nullable_yaml_search_completes_both_rounds(field, candidates, monkeypatch, tmp_path):
    monkeypatch.setattr(HistoricalData, 'get_spot_klines', staticmethod(recorded_yaml_source))
    manifest = {
        'type': 'rule_based',
        'data_source': {'method': 'limen.data.HistoricalData.get_spot_klines', 'params': {'klines_size': 900}},
        'split_dates': {'train_start': '2025-01-01', 'train_end': '2025-01-02', 'val_start': '2025-01-02', 'val_end': '2025-01-03', 'test_start': '2025-01-03', 'test_end': '2025-01-04'},
        'strategy': {'conditions': [{'id': 'entry', 'name': 'entry', 'type': 'threshold', 'column': 'close', 'operator': '>', 'value': 0}], 'entry': 'entry'},
        'reference_architecture': 'limen.sfd.reference_architecture.rule_based',
        'backtest': {'product': asdict(PRODUCT), field: '{candidate}'},
    }
    config = {'schema_version': '1.0', 'metadata': {'name': 'nullable_search', 'mode': 'development'}, 'sfd': {'manifest': manifest, 'params': {'candidate': candidates}}, 'uel': {'n_permutations': 2, 'search_strategy': {'type': 'grid'}, 'output_format': 'csv'}}
    path = tmp_path / 'search.yaml'
    with path.open('w') as stream:
        YAML().dump(config, stream)
    assert run_experiment(path, results_base=tmp_path, progress_bar=False)
    records = [json.loads(line) for line in next(tmp_path.rglob('round_data.jsonl')).read_text().splitlines()]
    assert len(records) == 2
    assert [record['trade_contract']['policy'][field] for record in records] == candidates
