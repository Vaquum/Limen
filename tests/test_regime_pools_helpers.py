from types import SimpleNamespace

import numpy as np
import pandas as pd
import polars as pl
import pytest

import limen.cohort.regime_pools as regime_pools_module
from limen.cohort.regime_pools import AggregationStrategy
from limen.cohort.regime_pools import DEFAULT_PERF_COLS
from limen.cohort.regime_pools import OfflineDiversification
from limen.cohort.regime_pools import OfflineFilter
from limen.cohort.regime_pools import OfflineRegime
from limen.cohort.regime_pools import OnlineAggregation
from limen.cohort.regime_pools import OnlineModelLoader
from limen.cohort.regime_pools import RegimeDiversifiedOpinionPools


def _make_perf_frame(n_rows: int = 5) -> pl.DataFrame:
    rows = []

    for row_idx in range(n_rows):
        row = {
            col: float(row_idx + col_idx + 1)
            for col_idx, col in enumerate(DEFAULT_PERF_COLS)
        }
        row['alpha'] = row_idx
        row['x_name'] = f'model-{row_idx}'
        row['n_kept'] = row_idx + 1
        row['id'] = row_idx
        row['cluster'] = row_idx % 2
        rows.append(row)

    return pl.DataFrame(rows)


def test_offline_filter_sanity_filter_drops_rows_with_null_performance_values() -> None:
    df = _make_perf_frame().with_columns(
        pl.when(pl.arange(0, pl.len()) == 2)
        .then(None)
        .otherwise(pl.col('precision_pct'))
        .alias('precision_pct')
    )

    filtered = OfflineFilter().sanity_filter(df)

    assert filtered.height == df.height - 1
    assert filtered['precision_pct'].null_count() == 0


def test_offline_filter_outlier_filter_removes_extreme_iqr_outlier() -> None:
    df = _make_perf_frame().with_columns(
        pl.when(pl.arange(0, pl.len()) == (pl.len() - 1))
        .then(10_000.0)
        .otherwise(pl.col('pred_pos_rate_pct'))
        .alias('pred_pos_rate_pct')
    )

    filtered = OfflineFilter(iqr_multiplier=1.5).outlier_filter(df)

    assert filtered.height == df.height - 1
    assert filtered['pred_pos_rate_pct'].max() < 10_000.0


def test_offline_regime_cluster_models_returns_single_cluster_for_one_sample() -> None:
    labels = OfflineRegime(random_state=0).cluster_models(_make_perf_frame(1), k=3)

    assert labels.tolist() == [0]


def test_offline_regime_cluster_models_limits_cluster_count_to_sample_size() -> None:
    labels = OfflineRegime(random_state=0).cluster_models(_make_perf_frame(4), k=10)

    assert len(labels) == 4
    assert set(labels.tolist()).issubset(set(range(4)))


def test_offline_diversification_returns_input_when_already_at_target_count() -> None:
    df = _make_perf_frame(2)

    selected = OfflineDiversification().pca_performance_selection(df, target_count=2)

    assert selected.equals(df)


def test_offline_diversification_selects_unique_models_up_to_target_count() -> None:
    df = _make_perf_frame(5)

    selected = OfflineDiversification().pca_performance_selection(
        df,
        target_count=3,
        n_clusters=2,
        random_state=0,
    )

    assert selected.height == 3
    assert len(set(selected['alpha'].to_list())) == 3


def test_online_model_loader_extract_model_params_excludes_perf_and_metadata_columns() -> None:
    loader = OnlineModelLoader(SimpleNamespace(), None)
    regime_df = _make_perf_frame(2)

    params = loader.extract_model_params(regime_df)

    assert params == [{'alpha': [0]}, {'alpha': [1]}]


def test_online_model_loader_merge_prediction_dataframes_joins_on_price_keys() -> None:
    loader = OnlineModelLoader(SimpleNamespace(), None)
    first = pd.DataFrame({
        'predictions': [0, 1],
        'open': [1, 2],
        'close': [2, 3],
        'price_change': [1, 1],
    })
    second = pd.DataFrame({
        'predictions': [1, 1],
        'open': [1, 2],
        'close': [2, 3],
        'price_change': [1, 1],
    })

    merged = loader.merge_prediction_dataframes([first, second])

    assert merged.columns == [
        'predictions',
        'open',
        'close',
        'price_change',
        'predictions_right',
    ]
    assert merged.height == 2


def test_aggregation_strategy_supports_mean_median_majority_vote_and_fallback() -> None:
    pred_arrays = np.array([
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0],
    ])

    assert AggregationStrategy().aggregate(pred_arrays, 'mean').tolist() == pytest.approx(
        [0.5, 1.0, 0.5]
    )
    assert AggregationStrategy().aggregate(pred_arrays, 'median').tolist() == pytest.approx(
        [0.5, 1.0, 0.5]
    )
    assert AggregationStrategy(threshold=0.75).aggregate(
        pred_arrays,
        'majority_vote',
    ).tolist() == pytest.approx([0.0, 1.0, 0.0])
    assert AggregationStrategy().aggregate(
        pred_arrays,
        'unknown',
    ).tolist() == pytest.approx([0.5, 1.0, 0.5])


def test_online_aggregation_run_regime_experiments_skips_failed_models_and_drops_prediction_columns() -> None:
    aggregation = OnlineAggregation(SimpleNamespace(), manifest=None)
    merge_loader = OnlineModelLoader(SimpleNamespace(), None)
    regime_df = pl.DataFrame({'alpha': [1, 2, 3]})
    experiment_frames = [
        pd.DataFrame({
            'predictions': [0.0, 1.0],
            'open': [1, 2],
            'close': [2, 3],
            'price_change': [1, 1],
        }),
        None,
        pd.DataFrame({
            'predictions': [1.0, 1.0],
            'open': [1, 2],
            'close': [2, 3],
            'price_change': [1, 1],
        }),
    ]

    aggregation.model_loader.extract_model_params = lambda _: [{'alpha': [1]}, {'alpha': [2]}, {'alpha': [3]}]
    aggregation.model_loader.run_single_model_experiment = (
        lambda data, params, regime_id, model_id: experiment_frames[model_id]
    )
    aggregation.model_loader.merge_prediction_dataframes = merge_loader.merge_prediction_dataframes

    aggregated, merged = aggregation.run_regime_experiments(
        data=pd.DataFrame({'feature': [1, 2]}),
        regime_id=0,
        regime_df=regime_df,
        aggregation_method='mean',
    )

    assert aggregated.tolist() == pytest.approx([0.5, 1.0])
    assert merged.columns == ['open', 'close', 'price_change']
    assert merged.height == 2


def test_rdop_offline_pipeline_handles_all_null_metrics_as_single_regime() -> None:
    confusion_metrics = pd.DataFrame({
        col: [None]
        for col in DEFAULT_PERF_COLS
    })
    confusion_metrics['alpha'] = [1]

    rdop = RegimeDiversifiedOpinionPools(SimpleNamespace(), random_state=0)

    result = rdop.offline_pipeline(confusion_metrics=confusion_metrics, k_regimes=3)

    assert rdop.n_regimes == 1
    assert list(result) == [0]
    assert result[0]['regime'].to_list() == [0]


def test_rdop_online_pipeline_returns_empty_frame_when_no_regime_produces_predictions() -> None:
    class _EmptyAggregation:
        def __init__(self, sfd, manifest=None, aggregation_threshold=None) -> None:
            pass

        def run_regime_experiments(self, data, regime_id, regime_df, aggregation_method):
            return np.array([]), pl.DataFrame()

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(regime_pools_module, 'OnlineAggregation', _EmptyAggregation)

        rdop = RegimeDiversifiedOpinionPools(SimpleNamespace())
        rdop.regime_pools = {0: pl.DataFrame({'alpha': [1]})}

        result = rdop.online_pipeline(pd.DataFrame({'feature': [1, 2]}))

    assert result.is_empty()


def test_rdop_online_pipeline_attaches_prediction_columns_for_each_regime() -> None:
    class _StubAggregation:
        def __init__(self, sfd, manifest=None, aggregation_threshold=None) -> None:
            self._base = pl.DataFrame({
                'open': [1, 2],
                'close': [2, 3],
                'price_change': [1, 1],
            })

        def run_regime_experiments(self, data, regime_id, regime_df, aggregation_method):
            return np.asarray([float(regime_id), 1.0]), self._base

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(regime_pools_module, 'OnlineAggregation', _StubAggregation)

        rdop = RegimeDiversifiedOpinionPools(SimpleNamespace())
        rdop.regime_pools = {
            0: pl.DataFrame({'alpha': [1]}),
            1: pl.DataFrame({'alpha': [2]}),
        }

        result = rdop.online_pipeline(pd.DataFrame({'feature': [1, 2]}))

    assert result.columns == [
        'open',
        'close',
        'price_change',
        'regime_0_prediction',
        'regime_1_prediction',
    ]
    assert result['regime_0_prediction'].to_list() == pytest.approx([0.0, 1.0])
    assert result['regime_1_prediction'].to_list() == pytest.approx([1.0, 1.0])
