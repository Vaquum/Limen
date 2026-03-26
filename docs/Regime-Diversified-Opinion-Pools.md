# Regime Diversified Opinion Pools

Regime Diversified Opinion Pools (RDOP) is Limen's cohorting layer for selecting and aggregating diversified model pools across performance regimes.

RDOP sits downstream from experiment runs. It consumes experiment analytics, groups strong permutations into regime-specific pools, and then produces one aggregated prediction series per stored regime.

## What RDOP Does Today

RDOP currently has two explicit phases:

1. `offline_pipeline()` filters, clusters, and diversifies experiment permutations into regime pools.
2. `online_pipeline()` reruns the selected permutations on fresh data and adds one aggregated prediction column per regime.

## What RDOP Does Not Do Today

The current implementation does not:

- choose a single active regime automatically
- collapse regime outputs into one final decision series
- provide a built-in cache layer
- provide downstream trade decisioning

Those decisions belong outside Limen.

## `RegimeDiversifiedOpinionPools.__init__`

Create an RDOP instance tied to a single SFD.

### Args

| Parameter | Type | Description |
|-----------|------|-------------|
| `sfd` | `SingleFileDecoder` | SFD used to retrain and rerun selected permutations |
| `random_state` | `int \| None` | Random state for clustering and diversification |

### Notes

- If the SFD is manifest-based, RDOP extracts the manifest during initialization.
- Offline and online phases should use the same SFD family so that selected permutations can be replayed consistently.

## Offline Pipeline

### `RegimeDiversifiedOpinionPools.offline_pipeline`

Cluster experiment results into regime-specific model pools.

### Args

| Parameter | Type | Description |
|-----------|------|-------------|
| `confusion_metrics` | `pd.DataFrame` | Experiment confusion-metrics table, typically `uel.experiment_confusion_metrics` |
| `perf_cols` | `list[str] \| None` | Performance columns used for filtering and clustering |
| `target_count` | `int` | Target number of selected models per regime |
| `k_regimes` | `int` | Requested number of regime clusters |
| `iqr_multiplier` | `float` | IQR multiplier for outlier filtering |
| `n_pca_components` | `int \| None` | PCA dimension count for diversification |
| `n_pca_clusters` | `int` | PCA-space cluster count for diversification |

### Returns

`dict[int, pl.DataFrame]`

Each key is a regime id, and each value is the selected permutation table for that regime.

### Current Processing Flow

1. sanity-filter invalid rows
2. outlier-filter extreme rows
3. cluster remaining rows into regimes
4. diversify each regime pool through PCA-space selection

### Fallback Behavior

- If sanity filtering removes everything, RDOP falls back to a single regime `0`.
- If outlier filtering removes everything, RDOP falls back to the sanity-filtered set.

## Online Pipeline

### `RegimeDiversifiedOpinionPools.online_pipeline`

Rerun the stored regime pools on fresh data and aggregate predictions within each regime.

### Args

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | `pd.DataFrame` | Fresh prediction data |
| `aggregation_method` | `str` | One of `mean`, `median`, or `majority_vote` |
| `aggregation_threshold` | `float \| None` | Optional threshold used during aggregation |

### Returns

`pl.DataFrame`

The returned frame contains the replayed data plus one aggregated prediction column per stored regime:

- `regime_0_prediction`
- `regime_1_prediction`
- ...

### Aggregation Methods

- `mean`: average regime prediction
- `median`: median regime prediction
- `majority_vote`: binary majority vote across models in the regime

## Example Workflow

```python
import limen
from limen import sfd

uel = limen.UniversalExperimentLoop(
    data=train_data,
    sfd=sfd.foundational_sfd.logreg_binary,
)

uel.run(
    experiment_name='logreg_training',
    n_permutations=1000,
    prep_each_round=True,
)

rdop = limen.RegimeDiversifiedOpinionPools(sfd.foundational_sfd.logreg_binary)

regime_pools = rdop.offline_pipeline(
    uel.experiment_confusion_metrics,
    k_regimes=5,
    target_count=25,
)

production_predictions = rdop.online_pipeline(
    data=live_kline_data,
    aggregation_method='median',
)
```

## Interpretation

RDOP is best understood as a cohort-construction layer:

- `Experiment` finds promising permutations
- `RDOP` groups and diversifies them
- downstream systems decide how regime-level outputs should be validated and acted on
