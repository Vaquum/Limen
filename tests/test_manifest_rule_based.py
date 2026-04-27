import polars as pl

from limen.experiment import Manifest


def _make_raw_data() -> pl.DataFrame:
    n = 40
    return pl.DataFrame({
        'datetime': pl.datetime_range(
            start=pl.datetime(2025, 1, 1),
            end=pl.datetime(2025, 1, 1) + pl.duration(hours=n - 1),
            interval='1h',
            eager=True,
        ),
        'rsi_14': [20.0, 35.0, 25.0, 40.0] * 10,
        'close':  [100.0, 110.0, 95.0, 105.0] * 10,
        'sma_50': [105.0] * n,
    })


_CONDITIONS = [
    {'id': 'rsi_low', 'type': 'threshold', 'column': 'rsi_14', 'operator': '<', 'value': 30},
    {'id': 'above_sma', 'type': 'relative', 'column': 'close', 'operator': '>', 'other_column': 'sma_50'},
    {'id': 'entry_signal', 'operator': 'and', 'operands': ['rsi_low', 'above_sma']},
]


def test_with_strategy_sets_sentinel_and_returns_self() -> None:
    m = Manifest()
    result = m.with_strategy(_CONDITIONS, entry='entry_signal')

    assert result is m
    assert m._rule_based is not None
    assert m._rule_based.entry == 'entry_signal'
    assert m._rule_based.conditions == _CONDITIONS


def test_prepare_data_rule_based_returns_split_dataframes() -> None:
    m = Manifest().with_strategy(_CONDITIONS, entry='entry_signal')
    data = m.prepare_data(_make_raw_data(), {})

    assert set(data.keys()) >= {'train', 'val', 'test', '_alignment', 'strategy'}
    for split in ('train', 'val', 'test'):
        assert isinstance(data[split], pl.DataFrame)
        assert 'datetime' not in data[split].columns


def test_prepare_data_rule_based_adds_predicate_columns() -> None:
    m = Manifest().with_strategy(_CONDITIONS, entry='entry_signal')
    data = m.prepare_data(_make_raw_data(), {})

    for split in ('train', 'val', 'test'):
        assert 'rsi_low' in data[split].columns
        assert 'above_sma' in data[split].columns
        assert data[split]['rsi_low'].dtype == pl.Boolean


def test_prepare_data_rule_based_attaches_strategy_config() -> None:
    m = Manifest().with_strategy(_CONDITIONS, entry='entry_signal')
    data = m.prepare_data(_make_raw_data(), {})

    assert data['strategy']['entry'] == 'entry_signal'
    assert data['strategy']['conditions'] == _CONDITIONS


def test_prepare_data_rule_based_compound_condition_not_added_as_column() -> None:
    m = Manifest().with_strategy(_CONDITIONS, entry='entry_signal')
    data = m.prepare_data(_make_raw_data(), {})

    for split in ('train', 'val', 'test'):
        assert 'entry_signal' not in data[split].columns


def test_prepare_data_rule_based_rejects_scaler() -> None:
    from limen.scalers.robust_scaler import RobustScaler
    m = Manifest().with_strategy(_CONDITIONS, entry='entry_signal').set_scaler(RobustScaler)
    try:
        m.prepare_data(_make_raw_data(), {})
        assert False, 'Expected ValueError'
    except ValueError as e:
        assert 'Scalers cannot be used' in str(e)


def test_prepare_data_rule_based_rejects_ablation() -> None:
    m = (Manifest()
         .with_strategy(_CONDITIONS, entry='entry_signal')
         .set_feature_ablation(drop_count_key='drop_n', seed_key='seed'))
    try:
        m.prepare_data(_make_raw_data(), {})
        assert False, 'Expected ValueError'
    except ValueError as e:
        assert 'Feature ablation cannot be used' in str(e)


def test_prepare_data_rule_based_rejects_condition_id_colliding_with_column() -> None:
    colliding_conditions = [
        {'id': 'rsi_14', 'type': 'threshold', 'column': 'rsi_14', 'operator': '<', 'value': 30},
        {'id': 'entry_signal', 'operator': 'and', 'operands': ['rsi_14']},
    ]
    m = Manifest().with_strategy(colliding_conditions, entry='entry_signal')
    try:
        m.prepare_data(_make_raw_data(), {})
        assert False, 'Expected ValueError for column name collision'
    except ValueError as e:
        assert 'collide' in str(e).lower()
