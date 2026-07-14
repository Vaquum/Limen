import hashlib
import math
from datetime import date
from pathlib import Path
from typing import Any, cast

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from limen.data import HistoricalData
from limen.experiment import RuleBasedManifest
from limen.features import dollar_bar_crash_reversal
from limen.sfd.foundational_sfd import dollar_bar_crash_reversal as sfd
from limen.sfd.reference_architecture.rule_based import RuleBasedStrategy
from limen.sfd.reference_architecture.rule_based import rule_based


_FIXTURE_PATH = (
    Path(__file__).parent
    / 'fixtures'
    / 'dollar_bar_crash_reversal_15m.parquet'
)
_FIXTURE_SHA256 = (
    'e8618f7d5a2278ad5250cc82ccf1acde9b5bf7de39a367d21466d6d7cb80378c'
)
_OUTPUT_COLUMN = 'dollar_bar_crash_reversal_position'


def _fixture() -> pl.DataFrame:
    return pl.read_parquet(_FIXTURE_PATH)


def _candidate(
    data: pl.DataFrame | pl.LazyFrame,
) -> pl.DataFrame | pl.LazyFrame:
    return dollar_bar_crash_reversal(
        data,
        momentum_threshold_bps=-575.0,
        flow_z_threshold=-0.5,
        hold_minutes=60,
    )


def test_dollar_bar_crash_reversal_preserves_frame_type_and_schema() -> None:
    source = _fixture()

    eager = _candidate(source)
    lazy = _candidate(source.lazy())

    assert isinstance(eager, pl.DataFrame)
    assert isinstance(lazy, pl.LazyFrame)
    assert eager.columns == [*source.columns, _OUTPUT_COLUMN]
    assert eager.schema[_OUTPUT_COLUMN] == pl.Int8
    assert not any(column.startswith('_dcr_') for column in eager.columns)
    assert_frame_equal(eager, lazy.collect())
    assert eager[_OUTPUT_COLUMN].sum() == 221


def test_dollar_bar_crash_reversal_parameter_grid_changes_the_signal() -> None:
    source = _fixture()
    cases = [
        (-600.0, -0.5, 60, 220),
        (-575.0, 0.0, 60, 170),
        (-575.0, -0.5, 30, 173),
        (-575.0, -0.5, 120, 305),
    ]

    for momentum, flow_z, hold, expected_active_rows in cases:
        result = dollar_bar_crash_reversal(
            source,
            momentum_threshold_bps=momentum,
            flow_z_threshold=flow_z,
            hold_minutes=hold,
        )
        assert isinstance(result, pl.DataFrame)
        assert result[_OUTPUT_COLUMN].sum() == expected_active_rows


@pytest.mark.parametrize('invalid', [True, 'bad', float('nan'), float('inf')])
def test_dollar_bar_crash_reversal_rejects_invalid_thresholds(
    invalid: Any,
) -> None:
    source = _fixture()

    with pytest.raises((TypeError, ValueError), match='momentum_threshold_bps'):
        _ = dollar_bar_crash_reversal(source, invalid, -0.5, 60)
    with pytest.raises((TypeError, ValueError), match='flow_z_threshold'):
        _ = dollar_bar_crash_reversal(source, -575.0, invalid, 60)


@pytest.mark.parametrize('invalid', [True, 0, -1, 1.5])
def test_dollar_bar_crash_reversal_rejects_invalid_hold(invalid: Any) -> None:
    with pytest.raises((TypeError, ValueError), match='hold_minutes'):
        _ = dollar_bar_crash_reversal(_fixture(), -575.0, -0.5, invalid)


def test_dollar_bar_crash_reversal_validates_frame_contract() -> None:
    source = _fixture()

    with pytest.raises(TypeError, match='DataFrame or LazyFrame'):
        _ = dollar_bar_crash_reversal(
            cast(Any, object()), -575.0, -0.5, 60
        )
    with pytest.raises(ValueError, match='missing required columns'):
        _ = dollar_bar_crash_reversal(
            source.drop('maker_liquidity'),
            -575.0,
            -0.5,
            60,
        )
    with pytest.raises(TypeError, match='datetime must be a Datetime'):
        _ = dollar_bar_crash_reversal(
            source.with_columns(pl.col('datetime').dt.epoch().alias('datetime')),
            -575.0,
            -0.5,
            60,
        )


def test_dollar_bar_crash_reversal_requires_one_later_same_day_row() -> None:
    source = _fixture()
    final_index = source.height - 1
    mutated = (
        source
        .with_row_index('_row')
        .with_columns(
            pl.when(pl.col('_row') == final_index)
            .then(pl.lit(1.0))
            .otherwise(pl.col('close'))
            .alias('close'),
            pl.when(pl.col('_row') == final_index)
            .then(pl.lit(0.0))
            .otherwise(pl.col('maker_liquidity'))
            .alias('maker_liquidity'),
        )
        .drop('_row')
    )

    unavailable = dollar_bar_crash_reversal(mutated, -5_000.0, -1e9, 1)
    appended = mutated.vstack(
        mutated.tail(1).with_columns(
            (pl.col('datetime') + pl.duration(minutes=1)).alias('datetime')
        )
    )
    available = dollar_bar_crash_reversal(appended, -5_000.0, -1e9, 1)

    assert isinstance(unavailable, pl.DataFrame)
    assert isinstance(available, pl.DataFrame)
    assert unavailable[-1, _OUTPUT_COLUMN] == 0
    assert available[-2, _OUTPUT_COLUMN] == 1


def test_dollar_bar_crash_reversal_is_prior_only_before_availability_boundary() -> None:
    source = _fixture()
    prefix_rows = 3_000

    prefix = _candidate(source.head(prefix_rows))
    full = _candidate(source)

    assert isinstance(prefix, pl.DataFrame)
    assert isinstance(full, pl.DataFrame)
    assert_frame_equal(
        prefix.head(prefix_rows - 1).select(_OUTPUT_COLUMN),
        full.head(prefix_rows - 1).select(_OUTPUT_COLUMN),
    )


def test_dollar_bar_crash_reversal_sfd_contract() -> None:
    grid = sfd.params()
    configured = sfd.manifest()

    assert grid == {
        'momentum_threshold_bps': [-525.0, -550.0, -575.0, -600.0],
        'flow_z_threshold': [-1.0, -0.5, 0.0, 0.5],
        'hold_minutes': [30, 45, 60, 90, 120],
    }
    assert len(grid) == 3
    assert math.prod(len(values) for values in grid.values()) == 80
    assert isinstance(configured, RuleBasedManifest)
    assert configured.data_source_config is not None
    assert (
        configured.data_source_config.method
        is HistoricalData.get_spot_dollar_klines
    )
    assert configured.data_source_config.params == {
        'dollar_bar_size': 15_000_000,
        'start_date_limit': '2020-02-01',
        'end_date_limit': '2026-07-10',
    }
    assert configured.required_bar_columns == [
        'datetime',
        'open',
        'close',
        'liquidity_sum',
        'maker_liquidity',
    ]
    assert configured.split_dates == (
        date(2020, 2, 1),
        date(2024, 1, 1),
        date(2024, 1, 1),
        date(2025, 1, 1),
        date(2025, 1, 1),
        date(2026, 7, 10),
    )
    assert len(configured.feature_transforms) == 1
    assert configured.feature_transforms[0].func is dollar_bar_crash_reversal
    assert configured.feature_transforms[0].params == {
        'momentum_threshold_bps': 'momentum_threshold_bps',
        'flow_z_threshold': 'flow_z_threshold',
        'hold_minutes': 'hold_minutes',
    }
    assert configured.strategy is not None
    assert configured.strategy.entry == 'crash_reversal_position'
    assert configured.strategy.conditions == [
        {
            'id': 'crash_reversal_position',
            'name': 'Crash-reversal position is active',
            'type': 'threshold',
            'column': _OUTPUT_COLUMN,
            'operator': '>',
            'value': 0,
        }
    ]
    assert configured.backtest_config is not None
    assert configured.backtest_config.fee_bps == 10.0
    assert configured.backtest_config.slip_bps == 5.0
    assert configured.architecture_function is rule_based


def test_dollar_bar_crash_reversal_real_fixture_edge() -> None:
    assert hashlib.sha256(_FIXTURE_PATH.read_bytes()).hexdigest() == _FIXTURE_SHA256

    featured = _candidate(_fixture())
    assert isinstance(featured, pl.DataFrame)
    evaluation_frame = featured.with_columns(
        (pl.col(_OUTPUT_COLUMN) > 0).alias('crash_reversal_position')
    )
    data: dict[str, Any] = {
        'train': evaluation_frame,
        'val': evaluation_frame,
        'test': evaluation_frame,
        'strategy': {
            'conditions': [
                {
                    'id': 'crash_reversal_position',
                    'type': 'threshold',
                    'column': _OUTPUT_COLUMN,
                    'operator': '>',
                    'value': 0,
                },
            ],
            'entry': 'crash_reversal_position',
        },
        '_alignment': {},
        'backtest_fee_bps': 10.0,
        'backtest_slip_bps': 5.0,
    }

    metrics = RuleBasedStrategy().evaluate(data)

    assert featured[_OUTPUT_COLUMN].sum() == 221
    assert metrics['num_trades_test'] == 5
    assert metrics['pnl_per_trade_bps_test'] == 148.9
    assert metrics['pnl_per_trade_bps_test'] > 60.0
