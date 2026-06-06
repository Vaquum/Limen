import math

import polars as pl
import pytest

from limen.experiment import MLManifest
from limen.experiment import Manifest
from limen.targets import EmaBreakoutTarget
from limen.targets import ExitQualityTarget
from limen.targets import ForwardBreakoutTarget
from limen.targets import ForwardVolNormalizedReturnTarget
from limen.targets import IdentityTarget
from limen.targets import NextBarDownTarget
from limen.targets import NextBarUpTarget
from limen.targets import NextReturnTarget
from limen.targets import QuantileBinaryTarget
from limen.targets import RandomBinaryTarget
from limen.targets import RiskRewardRatioTarget
from limen.targets import ThresholdBinaryTarget
from limen.targets import TripleBarrierTarget
from limen.targets import VolNormalizedReturnTarget


def _close_series(values: list[float]) -> pl.DataFrame:
    return pl.DataFrame({'close': values})


def _roc_series(values: list[float]) -> pl.DataFrame:
    return pl.DataFrame({'roc_1': values})


def _constant_vol_bars(n_rows: int = 8,
                       log_return: float = 0.01,
                       range_scale: float = 1.0) -> pl.DataFrame:
    rows = []
    price = 100.0
    log_range = abs(log_return) * math.sqrt(4.0 * math.log(2.0)) * range_scale
    for _ in range(n_rows):
        open_price = price
        close_price = open_price * math.exp(log_return)
        rows.append({
            'open': open_price,
            'high': open_price * math.exp(log_range),
            'low': open_price,
            'close': close_price,
        })
        price = close_price
    return pl.DataFrame(rows)


def _with_hourly_datetime(data: pl.DataFrame) -> pl.DataFrame:
    return data.with_columns(
        pl.datetime_range(
            start=pl.datetime(2025, 1, 1, 0, 0, 0),
            end=pl.datetime(2025, 1, 1, data.height - 1, 0, 0),
            interval='1h',
            eager=True,
        ).alias('datetime')
    ).select(['datetime', *data.columns])


def test_quantile_binary_fits_cutoff_on_train() -> None:
    train = _roc_series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    t = QuantileBinaryTarget(train, 'flag', source_column='roc_1', quantile=0.3)
    assert abs(t.cutoff - 0.7) < 1e-6


def test_quantile_binary_transform_labels_above_cutoff() -> None:
    train = _roc_series([0.1, 0.2, 0.3, 0.4, 0.5])
    t = QuantileBinaryTarget(train, 'flag', source_column='roc_1', quantile=0.4)
    data = _roc_series([0.1, 0.4, 0.9])
    result = t.transform(data, shift=0)
    labels = result['flag'].to_list()
    assert labels[2] == 1
    assert labels[0] == 0


def test_quantile_binary_shift_applied_when_nonzero() -> None:
    train = _roc_series([0.1, 0.9])
    t = QuantileBinaryTarget(train, 'flag', source_column='roc_1', quantile=0.4)
    data = _roc_series([0.9, 0.1, 0.9])
    unshifted = t.transform(data, shift=0)['flag'].to_list()
    shifted = t.transform(data, shift=-1)['flag'].to_list()
    # shift(-1) moves labels one position back; last entry becomes null → None
    assert unshifted[0] == shifted[1]
    assert shifted[-1] is None


def test_forward_breakout_labels_above_threshold() -> None:
    close = [100.0, 103.0, 100.0, 101.0, 100.0]
    data = _close_series(close)
    t = ForwardBreakoutTarget(data, 'breakout')
    result = t.transform(data, forward_periods=1, threshold=0.02, shift=0)
    assert result['breakout'][0] == 1
    assert result['breakout'][1] == 0


def test_forward_breakout_shift_applied() -> None:
    # Bar 2 produces a breakout (2.5% > 2%). After shift(-1) it should appear at bar 1.
    close = [100.0, 100.0, 100.0, 102.5, 100.0, 100.0]
    data = _close_series(close)
    t = ForwardBreakoutTarget(data, 'breakout')
    unshifted = t.transform(data, forward_periods=1, threshold=0.02, shift=0)
    shifted = t.transform(data, forward_periods=1, threshold=0.02, shift=-1)
    # shift(-1): shifted[i] == unshifted[i+1]
    assert unshifted['breakout'][2] == 1
    assert shifted['breakout'][1] == 1


def test_threshold_binary_labels_above_threshold() -> None:
    data = pl.DataFrame({'rsi': [20.0, 50.0, 80.0]})
    t = ThresholdBinaryTarget(data, 'rsi_high', source_column='rsi', threshold=60.0)
    result = t.transform(data, shift=0)
    assert result['rsi_high'].to_list() == [0, 0, 1]


def test_threshold_binary_threshold_is_fixed() -> None:
    train = pl.DataFrame({'rsi': [10.0, 20.0]})
    val = pl.DataFrame({'rsi': [70.0, 90.0]})
    t = ThresholdBinaryTarget(train, 'rsi_high', source_column='rsi', threshold=60.0)
    result = t.transform(val, shift=0)
    assert result['rsi_high'].to_list() == [1, 1]


def test_threshold_binary_shift_applied() -> None:
    data = pl.DataFrame({'v': [1.0, 5.0, 1.0, 1.0]})
    t = ThresholdBinaryTarget(data, 'flag', source_column='v', threshold=3.0)
    unshifted = t.transform(data, shift=0)
    shifted = t.transform(data, shift=-1)
    # shift(-1): label at bar 1 (v=5.0 > 3.0, label=1) moves to bar 0
    assert unshifted['flag'][1] == shifted['flag'][0]


def test_next_return_computes_percentage_return() -> None:
    data = _close_series([100.0, 110.0, 99.0])
    t = NextReturnTarget(data, 'ret')
    result = t.transform(data, periods=1, scale=100.0)
    assert abs(result['ret'][0] - 10.0) < 1e-6
    assert abs(result['ret'][1] - (-10.0)) < 1e-6


def test_next_return_respects_periods() -> None:
    data = _close_series([100.0, 100.0, 120.0])
    t = NextReturnTarget(data, 'ret')
    result = t.transform(data, periods=2, scale=100.0)
    assert abs(result['ret'][0] - 20.0) < 1e-6


def test_next_return_respects_scale() -> None:
    data = _close_series([100.0, 101.0, 100.0])
    t = NextReturnTarget(data, 'ret')
    result_pct = t.transform(data, periods=1, scale=100.0)
    result_raw = t.transform(data, periods=1, scale=1.0)
    assert abs(result_pct['ret'][0] - result_raw['ret'][0] * 100.0) < 1e-9


def test_next_bar_up_labels_next_close_above_current_close() -> None:
    data = _close_series([100.0, 101.0, 100.0, 100.0])
    t = NextBarUpTarget(data, 'next_bar_up')
    result = t.transform(data)
    assert result['next_bar_up'].to_list() == [1, 0, 0, None]


def test_next_bar_down_labels_next_close_below_current_close() -> None:
    data = _close_series([100.0, 101.0, 100.0, 100.0])
    t = NextBarDownTarget(data, 'next_bar_down')
    result = t.transform(data)
    assert result['next_bar_down'].to_list() == [0, 1, 0, None]


def test_vol_normalized_return_matches_unit_parkinson_ratio() -> None:
    data = _constant_vol_bars(n_rows=6, log_return=0.01)
    t = VolNormalizedReturnTarget(data, 'vol_normalized_return', halflife=2, min_periods=2)
    result = t.transform(data)

    assert result['vol_normalized_return'].to_list()[:2] == [None, None]
    assert result['vol_normalized_return'].to_list()[2:] == pytest.approx([1.0, 1.0, 1.0, 1.0])


def test_vol_normalized_return_uses_prior_sigma_for_current_return() -> None:
    data = _constant_vol_bars(n_rows=6, log_return=0.01)
    mutated_rows = data.to_dicts()
    mutated_rows[2]['high'] *= 100.0
    mutated = pl.DataFrame(mutated_rows)

    t = VolNormalizedReturnTarget(data, 'vol_normalized_return', halflife=2, min_periods=2)
    baseline = t.transform(data)['vol_normalized_return'].to_list()
    changed = t.transform(mutated)['vol_normalized_return'].to_list()

    assert changed[2] == pytest.approx(baseline[2])
    assert changed[3] != pytest.approx(baseline[3])


def test_vol_normalized_return_drops_dirty_ohlc_bars() -> None:
    data = _constant_vol_bars(n_rows=6, log_return=0.01)
    dirty_rows = data.to_dicts()
    dirty_rows[2]['high'] = max(dirty_rows[2]['open'], dirty_rows[2]['close']) - 0.01
    dirty_rows[4]['low'] = min(dirty_rows[4]['open'], dirty_rows[4]['close']) + 0.01
    dirty = pl.DataFrame(dirty_rows)

    t = VolNormalizedReturnTarget(data, 'vol_normalized_return', halflife=2, min_periods=1)
    result = t.transform(dirty)

    assert result.height == data.height - 2


def test_vol_normalized_return_rejects_bad_training_sigma_ratio() -> None:
    data = _constant_vol_bars(n_rows=6, log_return=0.01, range_scale=2.0)

    with pytest.raises(ValueError, match='outside'):
        VolNormalizedReturnTarget(data, 'vol_normalized_return', halflife=2, min_periods=2)


def test_forward_vol_normalized_return_uses_current_sigma_and_future_return() -> None:
    data = _constant_vol_bars(n_rows=6, log_return=0.01)
    t = ForwardVolNormalizedReturnTarget(data, 'forward_vol_normalized_return', halflife=2, min_periods=2)
    result = t.transform(data)

    assert result['forward_vol_normalized_return'].to_list()[:1] == [None]
    assert result['forward_vol_normalized_return'].to_list()[1:-1] == pytest.approx([1.0, 1.0, 1.0, 1.0])
    assert result['forward_vol_normalized_return'].to_list()[-1] is None


def test_forward_vol_normalized_return_absolute_and_multi_periods() -> None:
    data = _constant_vol_bars(n_rows=6, log_return=0.01)
    t = ForwardVolNormalizedReturnTarget(
        data,
        'forward_abs_vol_normalized_return',
        periods=2,
        absolute=True,
        halflife=2,
        min_periods=2,
    )
    result = t.transform(data)

    assert result['forward_abs_vol_normalized_return'].to_list()[1:-2] == pytest.approx([2.0, 2.0, 2.0])
    assert result['forward_abs_vol_normalized_return'].to_list()[-2:] == [None, None]


def test_canonical_decoder_outcomes_run_through_manifest() -> None:
    raw_up = _with_hourly_datetime(_constant_vol_bars(n_rows=12, log_return=0.01))
    raw_down = _with_hourly_datetime(_constant_vol_bars(n_rows=12, log_return=-0.01))

    up = (
        MLManifest()
        .set_split_config(6, 3, 3)
        .with_target_label('next_bar_up', NextBarUpTarget)
    ).prepare_data(raw_up, {'bar_type': 'base'})
    down = (
        MLManifest()
        .set_split_config(6, 3, 3)
        .with_target_label('next_bar_down', NextBarDownTarget)
    ).prepare_data(raw_down, {'bar_type': 'base'})
    normalized = (
        MLManifest()
        .set_split_config(6, 3, 3)
        .with_target_label(
            'vol_normalized_return',
            VolNormalizedReturnTarget,
            fit_params={'halflife': 2, 'min_periods': 1},
        )
    ).prepare_data(raw_up, {'bar_type': 'base'})

    assert up['y_train'].to_list() == [1, 1, 1, 1, 1]
    assert down['y_train'].to_list() == [1, 1, 1, 1, 1]
    assert normalized['y_train'].to_list() == pytest.approx([1.0, 1.0, 1.0, 1.0, 1.0])


def _make_split_data() -> list[pl.DataFrame]:
    roc = [float(i) / 10 for i in range(20)]
    close = [100.0 + i for i in range(20)]
    df = pl.DataFrame({'datetime': pl.Series(range(20)).cast(pl.Int64).cast(pl.Datetime), 'roc_1': roc, 'close': close})
    return [df[:10], df[10:15], df[15:]]


def test_with_target_label_sets_target_class_config() -> None:
    m = Manifest().with_target_label('qflag', QuantileBinaryTarget,
                                     fit_params={'source_column': 'roc_1', 'quantile': 0.3})
    assert m.target_class_config is not None
    assert m.target_class_config.target_class is QuantileBinaryTarget
    assert m.target_class_config.fit_params['source_column'] == 'roc_1'
    assert m.target_class_config.fit_params['quantile'] == 0.3


def test_with_target_label_applies_transform_to_all_splits() -> None:
    splits = _make_split_data()
    m = Manifest()
    m.with_target_label('qflag', QuantileBinaryTarget,
                         fit_params={'source_column': 'roc_1', 'quantile': 0.3})

    from limen.experiment.manifest_core import _apply_class_based_target
    all_fitted: dict = {}
    for i, split in enumerate(splits):
        result, all_fitted = _apply_class_based_target(m, split, {}, all_fitted, is_training=(i == 0))
        assert 'qflag' in result.columns


def test_with_target_label_fits_only_on_train() -> None:
    splits = _make_split_data()
    m = Manifest()
    m.with_target_label('qflag', QuantileBinaryTarget,
                         fit_params={'source_column': 'roc_1', 'quantile': 0.3})

    from limen.experiment.manifest_core import _apply_class_based_target
    all_fitted: dict = {}
    splits[0], all_fitted = _apply_class_based_target(m, splits[0], {}, all_fitted, is_training=True)
    instance_after_train = all_fitted['_target_cls_qflag']
    cutoff = instance_after_train.cutoff

    splits[1], all_fitted = _apply_class_based_target(m, splits[1], {}, all_fitted, is_training=False)
    # Instance not re-fitted — cutoff must be unchanged
    assert all_fitted['_target_cls_qflag'].cutoff == cutoff


def test_random_binary_target_adds_column_with_binary_values() -> None:
    data = pl.DataFrame({'close': [1.0, 2.0, 3.0, 4.0, 5.0]})
    t = RandomBinaryTarget(data, 'noise')
    result = t.transform(data)
    assert 'noise' in result.columns
    assert set(result['noise'].to_list()).issubset({0, 1})
    assert result.height == data.height


def test_identity_target_returns_data_unchanged() -> None:
    data = pl.DataFrame({'feature': [1.0, 2.0, 3.0], 'label': [0, 1, 0]})
    t = IdentityTarget(data, 'label')
    result = t.transform(data)
    assert result.equals(data)


def test_identity_target_raises_when_column_missing_on_init() -> None:
    data = pl.DataFrame({'feature': [1.0, 2.0]})
    try:
        IdentityTarget(data, 'label')
        assert False, 'expected ValueError'
    except ValueError as e:
        assert 'label' in str(e)


def test_identity_target_raises_when_column_missing_on_transform() -> None:
    train = pl.DataFrame({'feature': [1.0, 2.0], 'label': [0, 1]})
    t = IdentityTarget(train, 'label')
    data_without_label = pl.DataFrame({'feature': [3.0, 4.0]})
    try:
        t.transform(data_without_label)
        assert False, 'expected ValueError'
    except ValueError as e:
        assert 'label' in str(e)


def test_ema_breakout_target_labels_future_moves_above_ema_threshold() -> None:
    data = pl.DataFrame({'close': [10.0, 10.0, 20.0, 20.0]})
    t = EmaBreakoutTarget(data, 'breakout_ema')
    result = t.transform(data, target_col='close', ema_span=2, breakout_delta=0.1, breakout_horizon=1)

    assert result['breakout_ema'].to_list() == [0, 1, 1, None]


def test_exit_quality_target_distinguishes_good_bad_and_neutral_exits() -> None:
    data = pl.DataFrame(
        {
            'exit_reason': ['target_hit', 'stop_loss', 'timeout', 'timeout'],
            'exit_net_return': [0.2, -0.4, -0.1, 0.1],
        }
    )
    t = ExitQualityTarget(data, 'exit_quality')
    result = t.transform(data)

    assert result['exit_quality'].to_list() == pytest.approx([1.0, 0.2, 0.2, 0.5])
    assert 'exit_reason' not in result.columns
    assert 'exit_net_return' not in result.columns


def test_risk_reward_ratio_target_uses_absolute_drawdown_with_epsilon_guard() -> None:
    data = pl.DataFrame(
        {
            'capturable_breakout': [0.5, 1.0],
            'max_drawdown': [-0.1, 0.0],
        }
    )
    t = RiskRewardRatioTarget(data, 'risk_reward_ratio')
    result = t.transform(data)

    assert result['risk_reward_ratio'].to_list() == pytest.approx([0.5 / 0.101, 1000.0])
    assert 'capturable_breakout' not in result.columns
    assert 'max_drawdown' not in result.columns


def _triple_barrier_close() -> list[float]:
    return [100.0, 101.0, 100.0, 101.0, 100.0]


def test_triple_barrier_labels_upper_touch_first() -> None:
    data = _close_series([*_triple_barrier_close(), 200.0, 200.0, 200.0])
    t = TripleBarrierTarget(data, 'tb', span=2, min_periods=2, max_horizon=3)
    result = t.transform(data)['tb'].to_list()
    assert result[4] == 1


def test_triple_barrier_labels_lower_touch_first() -> None:
    data = _close_series([*_triple_barrier_close(), 50.0, 50.0, 50.0])
    t = TripleBarrierTarget(data, 'tb', span=2, min_periods=2, max_horizon=3)
    result = t.transform(data)['tb'].to_list()
    assert result[4] == -1


def test_triple_barrier_vertical_barrier_labels_zero() -> None:
    data = _close_series([*_triple_barrier_close(), 100.0, 100.0, 100.0])
    t = TripleBarrierTarget(data, 'tb', span=2, min_periods=2, max_horizon=3)
    result = t.transform(data)['tb'].to_list()
    assert result[4] == 0


def test_triple_barrier_nulls_warmup_and_truncated_horizon() -> None:
    data = _close_series([*_triple_barrier_close(), 100.0, 100.0, 100.0])
    t = TripleBarrierTarget(data, 'tb', span=2, min_periods=2, max_horizon=3)
    result = t.transform(data)['tb'].to_list()
    assert result[0] is None
    assert result[1] is None
    assert result[5] is None
    assert result[6] is None
    assert result[7] is None


def test_triple_barrier_lower_multiple_widens_stop() -> None:
    data = _close_series([*_triple_barrier_close(), 97.0, 110.0, 110.0])
    narrow = TripleBarrierTarget(data, 'tb', span=2, min_periods=2, max_horizon=3,
                                 lower_multiple=1.0).transform(data)['tb'].to_list()
    wide = TripleBarrierTarget(data, 'tb', span=2, min_periods=2, max_horizon=3,
                               lower_multiple=100.0).transform(data)['tb'].to_list()
    assert narrow[4] == -1
    assert wide[4] == 1


def test_triple_barrier_rejects_nonpositive_params() -> None:
    data = _close_series(_triple_barrier_close())
    with pytest.raises(ValueError, match='upper_multiple'):
        TripleBarrierTarget(data, 'tb', upper_multiple=0.0)
    with pytest.raises(ValueError, match='max_horizon'):
        TripleBarrierTarget(data, 'tb', max_horizon=0)
