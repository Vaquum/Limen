from pathlib import Path

import numpy as np
import polars as pl
import pytest

from limen.experiment.manifest_core import FundingConfig, MLManifest, ProductConfig
from limen.sfd.reference_architecture.direction_sizing import direction_sizing
from limen.targets import TradeOutcomeTarget


def recorded_source(klines_size=900, limit=120):
    return pl.read_parquet(Path(__file__).parent / 'fixtures/spot_15m_20250101_20250531.parquet').head(limit)


def native_manifest():
    manifest = MLManifest().set_data_source(recorded_source, params={'klines_size':900})
    manifest.set_split_config(6,2,2)
    manifest.with_target_label('outcome', TradeOutcomeTarget)
    manifest.with_reference_architecture(direction_sizing)
    manifest.set_backtest_config(prediction_mode='target_exposure', product=ProductConfig('linear_perpetual','BTCUSDT','BTC','USDT',1e-9,0), max_holding_seconds=1800, funding=FundingConfig(preset='binance_btcusdt', params={'rate':0.0001}))
    return manifest


class ConstantComponent:
    def __init__(self, value):
        self.value = value

    def fit(self, x, y):
        assert len(x) == len(y) > 0
        self.targets = np.asarray(y).copy()
        return self

    def predict(self, x):
        return np.full(len(x), self.value, dtype=np.float64)


def direction_factory(*, seed, sign=1):
    return ConstantComponent(sign)


def sizing_factory(*, seed, size=0.25):
    return ConstantComponent(size)


direction_factory.deterministic = True
sizing_factory.deterministic = True


def component_params(**kwargs):
    return {'direction_factory':direction_factory,'sizing_factory':sizing_factory,'min_train_samples':2,**kwargs}


def test_native_components_and_tunable_parameters():
    manifest = native_manifest()
    source = recorded_source()
    data = manifest.prepare_data(source, {})
    for sign, size in ((1,.25),(-1,.25),(-1,.75)):
        result = manifest.run_model(data, component_params(direction_params={'sign':sign}, sizing_params={'size':size}))
        assert np.all(result['_preds'] == sign * size)
        assert len(result['_preds']) == len(data['x_test'])
        assert result['_prediction_mode'] == 'target_exposure'
        assert result['_model'].deterministic
    with pytest.raises(ValueError, match='feature'):
        manifest.run_model(data, component_params(direction_features=['missing']))
