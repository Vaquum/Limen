import ast
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn import metrics

from limen.backtest.long_flat_strategy import long_flat_strategy
from limen.cohort.sfc.top_n import select
from limen.data.historical_data import _aggregate_spot_dollar_klines, _aggregate_spot_klines
from limen.experiment.reducer.filter_types import _build_sample, FILTER_SAMPLE, SAMPLE_HASH_RULE
from tests.test_msq import _make_msq
from limen.features.entry_score_microstructure import entry_score_microstructure
from limen.features.fractional_diff import find_min_d
from limen.features.momentum_confirmation import momentum_confirmation
from limen.features.momentum_weight import momentum_weight
from limen.log.log import Log
from limen.log._experiment_backtest_results import _prepare_snapshot_backtest_input
from limen.log._experiment_parameter_correlation import experiment_parameter_correlation
from limen.log._permutation_confusion_metrics import confusion_mean_return_pct, permutation_confusion_metrics
from limen.metrics.binary_metrics import binary_metrics
from limen.metrics.continuous_metrics import continuous_metrics
from limen.metrics.multiclass_metrics import multiclass_metrics
from limen.metrics.rule_based_metrics import rule_based_metrics
from limen.metrics.safe_ovr_auc import safe_ovr_auc
from limen.sfd.reference_architecture.random_binary import RandomBinary
from limen.sfd.reference_architecture.rule_based import _compounded_trade_pnl_summary

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def bars():
    return pl.read_parquet(ROOT / 'tests/fixtures/spot_1h_20240101_20241231.parquet')


def test_recorded_metrics_preserve_precision(bars, tmp_path):
    actual = (bars['close'] > bars['open']).to_numpy().astype(int)
    predictions = np.roll(actual, 1)
    probabilities = bars['close'].rank().to_numpy() / (bars.height + 1)
    result = binary_metrics({'y_test': actual}, predictions, probabilities)
    assert result == {
        'precision': metrics.precision_score(actual, predictions),
        'recall': metrics.recall_score(actual, predictions),
        'fpr': metrics.confusion_matrix(actual, predictions)[0, 1] / (actual == 0).sum(),
        'auc': metrics.roc_auc_score(actual, probabilities),
        'accuracy': metrics.accuracy_score(actual, predictions),
    }
    assert any(value != round(value, 3) for value in result.values())
    prices = bars['close'].to_numpy()
    regression = continuous_metrics({'y_test': prices[1:]}, prices[:-1])
    assert regression == {
        'bias': float(np.mean(prices[:-1] - prices[1:])),
        'mae': metrics.mean_absolute_error(prices[1:], prices[:-1]),
        'rmse': metrics.root_mean_squared_error(prices[1:], prices[:-1]),
        'r2': metrics.r2_score(prices[1:], prices[:-1]),
        'mape': metrics.mean_absolute_percentage_error(prices[1:], prices[:-1]) * 100,
    }
    returns = (bars['close'] / bars['open'] - 1).to_numpy()
    classes = np.searchsorted(np.quantile(returns, [1 / 3, 2 / 3]), returns)
    predicted_classes = np.roll(classes, 1)
    # One-hot probabilities are predictions derived from recorded price classes.
    class_probabilities = np.eye(3)[predicted_classes]
    multi = multiclass_metrics({'y_test': classes}, predicted_classes, class_probabilities)
    assert multi == {
        'precision': metrics.precision_score(classes, predicted_classes, average='macro'),
        'recall': metrics.recall_score(classes, predicted_classes, average='macro'),
        'auc': safe_ovr_auc(classes, class_probabilities),
        'accuracy': metrics.accuracy_score(classes, predicted_classes),
    }
    # One changed classification must survive CSV and ranking despite a rounded tie.
    imperfect = actual.copy()
    imperfect[0] = 1 - imperfect[0]
    scores = [binary_metrics({'y_test': actual}, pred, probabilities)['accuracy']
              for pred in (imperfect, actual)]
    assert scores[0] < scores[1] and round(scores[0], 3) == round(scores[1], 3)
    path = tmp_path / 'results.csv'
    pl.DataFrame({'id': [0, 1], 'accuracy': scores}).write_csv(path)
    saved = pl.read_csv(path)
    assert saved['accuracy'].to_list() == scores
    assert select({'results': saved}, column='accuracy', n=2) == [1, 0]


def test_recorded_returns_survive_log_csv_exactly(bars, tmp_path):
    returns = ((bars['close'] - bars['open']) / bars['open']).to_numpy()
    path = tmp_path / 'results.csv'
    pl.DataFrame({'id': np.arange(len(returns)), 'return': returns}).write_csv(path)
    restored = Log(file_path=str(path)).experiment_log['return'].to_numpy()
    np.testing.assert_array_equal(restored, returns)


def test_recorded_confusion_and_rule_metrics_preserve_precision(bars):
    actual = (bars['close'] > bars['open']).to_numpy().astype(int)
    predictions = np.roll(actual, 1)
    opens, closes = bars['open'].to_numpy(), bars['close'].to_numpy()
    model = RandomBinary()
    confusion = model._compute_confusion(predictions, actual)
    assert confusion['confusion_precision'] == metrics.precision_score(actual, predictions)
    assert confusion['confusion_recall'] == metrics.recall_score(actual, predictions)
    means = confusion_mean_return_pct(predictions, actual, opens, closes - opens)
    future_returns = (closes[1:] - opens[1:]) / opens[1:] * 100
    for name, pred_value, actual_value in [('tp', 1, 1), ('fp', 1, 0), ('tn', 0, 0), ('fn', 0, 1)]:
        mask = (predictions[:-1] == pred_value) & (actual[:-1] == actual_value)
        assert means[f'{name}_mean_return_pct'] == float(future_returns[mask].mean())
    frame = pl.DataFrame({'predictions': predictions, 'actuals': actual,
                          'open': opens, 'price_change': closes - opens})
    log = SimpleNamespace(permutation_prediction_performance=lambda _: frame.clone())
    report = permutation_confusion_metrics(log, 'price_change', 0, outlier_quantiles=(0, 1)).row(0, named=True)
    assert report['precision_pct'] == metrics.precision_score(actual, predictions) * 100
    assert report['recall_pct'] == metrics.recall_score(actual, predictions) * 100
    assert report['pred_pos_rate_pct'] == predictions.mean() * 100
    assert report['actual_pos_rate_pct'] == actual.mean() * 100
    positions = {'train': predictions, 'val': actual, 'test': np.roll(actual, 2)}
    backtests = {split: {'drawdown_bps_p5': float(bars['std'][i])} for i, split in enumerate(positions)}
    rules = rule_based_metrics(positions, backtests)
    assert rules['position_rate_train'] == predictions.mean()
    assert rules['drawdown_std_bps'] == np.std([value['drawdown_bps_p5'] for value in backtests.values()], ddof=1)
    execution = long_flat_strategy(predictions, opens, closes, closes - opens, fee_bps=0, slip_bps=0)
    in_market = execution.pos > 0
    starts = np.flatnonzero(in_market & ~np.r_[False, in_market[:-1]])
    ends = np.flatnonzero(in_market & ~np.r_[in_market[1:], False])
    pnl, count = _compounded_trade_pnl_summary(execution, 1.0)
    trades = [np.prod(1 + execution.net[start:end + 1]) - 1 for start, end in zip(starts, ends, strict=True)]
    assert count == len(trades)
    assert pnl == float(np.mean(trades)) * 10000


def test_recorded_aggregates_preserve_precision(bars):
    assert _aggregate_spot_klines(bars, 3600).equals(bars)
    aggregate = _aggregate_spot_klines(bars, 7200)
    pairs = bars.head(2)
    for name in ('volume', 'liquidity_sum', 'maker_liquidity'):
        assert aggregate[name][0] == pairs[name].sum()
    expected_mean = (pairs['mean'] * pairs['no_of_trades']).sum() / pairs['no_of_trades'].sum()
    assert aggregate['mean'][0] == expected_mean
    dollar = _aggregate_spot_dollar_klines(bars, 100_000_000, None)
    assert _aggregate_spot_dollar_klines(dollar, 100_000_000, 100_000_000).equals(dollar)
    combined = _aggregate_spot_dollar_klines(dollar, 200_000_000, 100_000_000)
    assert combined['volume'].sum() == pytest.approx(dollar['volume'].sum())
    assert any(aggregate[name][0] != round(aggregate[name][0], decimals)
               for name, decimals in [('mean', 5), ('std', 6), ('volume', 9), ('liquidity_sum', 1)])


def test_recorded_weighted_features_retain_float64(bars):
    weight = float(bars['maker_ratio'][0])
    weighted = momentum_weight(bars, period=1, weight_multiplier=weight, base_weight=0)
    assert weighted['momentum_weight'].dtype == pl.Float64
    positive = bars['close'].pct_change(1) > 0
    assert weighted.filter(positive)['momentum_weight'][0] == weight
    confirmed = momentum_confirmation(bars, short_period=1, long_period=3, short_weight=weight)
    assert confirmed['momentum_score'].dtype == pl.Float64
    mixed = (confirmed['momentum_1'] > 0) & (confirmed['momentum_3'] <= 0)
    assert confirmed.filter(mixed)['momentum_score'][0] == weight
    scored = entry_score_microstructure(
        bars.with_columns(pl.lit('low').alias('volatility_regime')),
        entry_position_weight_low_vol=0, entry_volume_weight_low_vol=0,
        entry_spread_weight_low_vol=0, entry_momentum_weight_low_vol=weight,
    )
    assert scored.filter(pl.col('micro_momentum') > 0)['entry_score'].drop_nulls()[0] == weight


@pytest.mark.parametrize('dtype', [pl.Float32, pl.Float64])
@pytest.mark.parametrize('regime', ['base', 'low', 'high'])
@pytest.mark.parametrize('component', ['position', 'momentum', 'volume', 'spread'])
def test_every_entry_component_weights_in_float64(bars, dtype, regime, component):
    source = bars.with_columns(pl.col(pl.Float64).cast(dtype), pl.lit(regime).alias('volatility_regime'))
    weight = float(bars['maker_ratio'][0])
    suffix = 'base' if regime == 'base' else f'{regime}_vol'
    kwargs = {f'entry_{name}_weight_{suffix}': weight if name == component else 0.0
              for name in ('position', 'momentum', 'volume', 'spread')}
    scored = entry_score_microstructure(source, **kwargs)
    spread = scored['spread_percent'].cast(pl.Float64)
    components = {
        'position': 1 - scored['position_in_candle'].cast(pl.Float64),
        'momentum': (scored['micro_momentum'] > 0).cast(pl.Float64),
        'volume': scored['volume_spike'].cast(pl.Float64).clip(0.5, 1.5) / 1.5,
        'spread': 1 - (spread / spread.rolling_mean(48)).clip(0, 2),
    }
    valid = scored['entry_score'].is_not_null()
    assert valid.any()
    assert scored['entry_score'].dtype == pl.Float64
    np.testing.assert_array_equal(scored['entry_score'].filter(valid).to_numpy(),
                                  (components[component] * weight).filter(valid).to_numpy())


def test_recorded_cohort_percentages_stay_distinct(bars):
    data = pl.DataFrame({'metric': bars['close'], 'parameter': bars['volume']})
    report = experiment_parameter_correlation(SimpleNamespace(experiment_log=data), 'metric',
                                               heads=[10, 11], min_n=2, n_boot=2)
    assert report['cohort_pct'].unique(maintain_order=True).to_list() == [1000 / len(data), 1100 / len(data)]
    assert report.select('cohort_pct', 'feature').unique().height == report.height


def test_sampling_preserves_sub_basis_point_fractions(bars, monkeypatch):
    module = import_module('limen.experiment.reducer.filter_types')
    # Fix the hash at its actual minimum to probe the old zero-threshold collapse.
    monkeypatch.setattr(module.hashlib, 'blake2b', lambda *args, **kwargs: SimpleNamespace(digest=lambda: bytes(8)))
    fraction = 1 / bars.height ** 2
    combo = {'param': float(bars['mean'][0])}
    reject = _build_sample({'param': 'param', 'value': combo['param'], 'fraction': fraction})
    assert not reject(combo)
    assert _build_sample({'param': 'param', 'value': combo['param'], 'fraction': 0.0})(combo)
    assert not _build_sample({'param': 'param', 'value': combo['param'], 'fraction': 1.0})(combo)


def test_sampling_checkpoint_versions_selection_rule(bars):
    values = bars['mean'].head(8).to_list()
    original, _ = _make_msq({'p': values})
    params = {'param': 'p', 'value': values[0], 'fraction': 0.5}
    original.set_filter('sample', _build_sample(params), filter_type=FILTER_SAMPLE, filter_params=params)
    state = original.get_state()
    assert state['named_filter_descriptors']['sample'][1]['hash_rule'] == SAMPLE_HASH_RULE
    state['intervention_log'][-1]['filter_params']['fraction'] = 0.25
    assert original.get_state()['named_filter_descriptors']['sample'][1]['fraction'] == 0.5
    restored, _ = _make_msq({'p': values})
    restored.set_state(state)
    assert list(restored) == list(original)
    del state['named_filter_descriptors']['sample'][1]['hash_rule']
    with pytest.raises(ValueError, match='use the original package version'):
        restored.set_state(state)


def test_fractional_search_preserves_declared_grid(bars, monkeypatch):
    module = import_module('limen.features.fractional_diff')
    adf_module = import_module('limen.utils.adf_test')
    visited = []
    def record(data, d, cols, threshold):
        visited.append(d)
        return data.with_columns(pl.col(cols[0]).alias(f'{cols[0]}_fracdiff'))
    monkeypatch.setattr(module, 'fractional_diff', record)
    monkeypatch.setattr(adf_module, 'adf_test', lambda *args, **kwargs: SimpleNamespace(stationary=len(visited) == 3))
    assert find_min_d(bars.head(24), 'close', d_start=0.1, d_end=0.3, step=0.1) == 0.3
    assert visited == [0.1, 0.2, 0.3]
    visited.clear()
    start = float(bars['maker_ratio'][0])
    end = start + 2e-11
    found = find_min_d(bars.head(24), 'close', d_start=start, d_end=end, step=1e-11)
    assert len(visited) == 3
    assert found == end
    assert len(set(visited)) == 3


def test_near_integer_recorded_scores_remain_directional(bars):
    returns = (bars['close'] / bars['open'] - 1).to_numpy()
    classes = np.searchsorted(np.quantile(returns, [1 / 3, 2 / 3]), returns)
    scores = classes + 1 + np.abs(returns) * 1e-6
    frame = pd.DataFrame({'predictions': scores})
    prepared = _prepare_snapshot_backtest_input(frame, prediction_kind='directional')
    assert (prepared['predictions'] == 1).all()


def test_package_rounding_inventory():
    # Only explicitly discrete nanoseconds and integer parameter domains may round.
    allowed = {
        'backtest/execution_events.py', 'backtest/funding_adapter.py',
        'backtest/trade_execution.py', 'experiment/_prepare_trade_context.py',
        'targets/trade_outcome.py', 'experiment/reducer/focus_reducer.py',
        'experiment/_holdout_report.py',
    }
    found = set()
    for path in (ROOT / 'limen').rglob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ''
            if name not in {'round', 'around', 'rint'}:
                continue
            relative = path.relative_to(ROOT / 'limen').as_posix()
            assert relative in allowed, f'Lossy measurement rounding: {relative}:{node.lineno}'
            assert len(node.args) == 1 and not node.keywords, f'Decimal rounding: {relative}:{node.lineno}'
            if relative != 'experiment/reducer/focus_reducer.py':
                assert 'NANOSECONDS' in ast.unparse(node.args[0])
            found.add(relative)
    assert found == allowed
