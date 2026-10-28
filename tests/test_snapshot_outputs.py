from pathlib import Path

import numpy as np
import polars as pl
import pytest

from limen.backtest.backtest_snapshot import BACKTEST_SNAPSHOT_COLUMNS, backtest_snapshot
from limen.log._experiment_backtest_results import _prepare_snapshot_backtest_input


@pytest.fixture(scope='module')
def prices():
    return pl.read_parquet(Path(__file__).parent / 'fixtures/spot_1h_20240101_20240116.parquet').head(32).select(
        'open', 'close', (pl.col('close') - pl.col('open')).alias('price_change'),
    )


def _same_ledger(actual, expected):
    assert list(actual) == list(expected) == BACKTEST_SNAPSHOT_COLUMNS
    for key, value in expected.items():
        assert actual[key] == pytest.approx(value, nan_ok=True), key


@pytest.mark.parametrize('strings', [False, True])
def test_multiclass_explicit_mapping_matches_binary_control(prices, strings):
    change = prices['price_change'].to_numpy()
    identities = np.where(change > np.median(change), 1, np.where(change < 0, 2, 0))
    if strings:
        identities = np.asarray(['flat', 'long', 'other'])[identities]
        mapping = {'flat': 0, 'long': 1, 'other': 0, 'unobserved': 1}
        expected = identities == 'long'
    else:
        mapping = {0: 0, 1: 1, 2: 0, 9: 1}
        expected = identities == 1
    columns = {**prices.to_dict(as_series=False), 'predictions': identities}
    before = identities.copy()
    _same_ledger(backtest_snapshot(columns, class_mapping=mapping), backtest_snapshot({**columns, 'predictions': expected}))
    np.testing.assert_array_equal(identities, before)


@pytest.mark.parametrize('label', [0, 1])
def test_multilabel_explicit_column_matches_binary_control(prices, label):
    labels = np.column_stack((prices['price_change'].to_numpy() > 0, prices['close'].to_numpy() > prices['close'].median()))
    before = labels.copy()
    columns = {**prices.to_dict(as_series=False), 'predictions': labels}
    _same_ledger(backtest_snapshot(columns, label_index=label), backtest_snapshot({**columns, 'predictions': labels[:, label]}))
    np.testing.assert_array_equal(labels, before)


@pytest.mark.parametrize('labels', [False, True])
def test_explicit_log_input_uses_the_same_interpretation(prices, labels):
    signal = (prices['price_change'] > 0).cast(pl.Int64)
    if labels:
        values = np.column_stack((signal.to_numpy(), 1 - signal.to_numpy())).tolist()
        options = {'label_index': 1}
        expected = (1 - signal).to_list()
    else:
        values = ['up' if value else 'flat' for value in signal]
        options = {'class_mapping': {'up': 1, 'flat': 0}}
        expected = signal.to_list()
    frame = prices.with_columns(pl.Series('predictions', values), pl.Series('actuals', values))
    original = frame.clone()
    prepared = _prepare_snapshot_backtest_input(frame, **options)
    assert prepared['predictions'].to_list() == prepared['actuals'].to_list() == expected
    assert prepared.select(prices.columns).equals(prices)
    assert frame.equals(original)
    _same_ledger(backtest_snapshot(frame.to_dict(as_series=False), **options), backtest_snapshot(prepared.to_dict(as_series=False)))


@pytest.mark.parametrize('mapping', [{}, {0: 0}, {0: 0, 1: 2}, {0: 0, 1: True}, {False: 0, 1: 1}, {0.5: 0, 1: 1}, ['invalid']])
def test_invalid_class_mapping_refuses(prices, mapping):
    with pytest.raises(ValueError, match='class_mapping'):
        backtest_snapshot({**prices.to_dict(as_series=False), 'predictions': [1] * prices.height}, class_mapping=mapping)


@pytest.mark.parametrize('identities', [[None], [np.nan], [0.5], [1.0], [True], [[1]], []])
def test_ambiguous_class_identities_refuse(prices, identities):
    with pytest.raises(ValueError, match='snapshot class'):
        backtest_snapshot({**prices.head(1).to_dict(as_series=False), 'predictions': identities}, class_mapping={0: 0, 1: 1})


@pytest.mark.parametrize('index', [-1, True, 0.5, 2])
def test_invalid_label_index_refuses(prices, index):
    with pytest.raises(ValueError, match='label_index'):
        backtest_snapshot({**prices.head(1).to_dict(as_series=False), 'predictions': [[0, 1]]}, label_index=index)


@pytest.mark.parametrize('labels', [[], [[]], [1], [[0, 1], [1]], [[0, 2]], [[0, np.nan]], [[0, None]], [['0', '1']], [[0, 0.5]]])
def test_ambiguous_multilabel_inputs_refuse(prices, labels):
    with pytest.raises(ValueError, match='snapshot'):
        backtest_snapshot({**prices.head(1).to_dict(as_series=False), 'predictions': labels}, label_index=0)


def test_interpretation_conflicts_and_unknown_actuals_refuse(prices):
    columns = {**prices.to_dict(as_series=False), 'predictions': [[1, 0]] * prices.height}
    with pytest.raises(ValueError, match='mutually exclusive'):
        backtest_snapshot(columns, class_mapping={0: 0, 1: 1}, label_index=0)
    frame = prices.with_columns(pl.lit(1).alias('predictions'), pl.lit(2).alias('actuals'))
    with pytest.raises(ValueError, match='no decision for class 2'):
        _prepare_snapshot_backtest_input(frame, class_mapping={1: 1})
    for options in ({'class_mapping': {1: 1}}, {'label_index': 0}):
        with pytest.raises(ValueError, match='directional interpretation'):
            _prepare_snapshot_backtest_input(frame, prediction_kind='directional', **options)
