# Regime Diversified Opinion Pools

Regime Diversified Opinion Pools (RDOP) is Limen's cohorting layer for grouping strong experiment rounds into diversified regime-specific pools and replaying those pools on fresh data.

It sits downstream from experiment runs and upstream from any final decision system.

## What RDOP Does Today

RDOP currently has two explicit phases:

1. `offline_pipeline()` filters, clusters, and diversifies experiment permutations into regime pools
2. `online_pipeline()` reruns the selected pools and adds one aggregated prediction column per stored regime

The output is still a set of regime-level prediction series. RDOP does not choose the winning regime for you and it does not produce a final trading decision.

## What RDOP Does Not Do Today

The current implementation does not:

- choose one active regime automatically
- collapse all regime outputs into one final signal
- perform downstream risk validation or capital checks
- provide a built-in caching or orchestration layer

Those concerns belong downstream from Limen.

## Typical Workflow

```python
import limen

rdop = limen.RegimeDiversifiedOpinionPools(
    limen.sfd.foundational_sfd.logreg_binary
)

regime_pools = rdop.offline_pipeline(
    uel.experiment_confusion_metrics,
    k_regimes=5,
    target_count=25,
)

regime_predictions = rdop.online_pipeline(
    data=fresh_data,
    aggregation_method='median',
)
```

In that flow:

- the experiment supplies the benchmark table
- RDOP turns that table into per-regime model pools
- RDOP then replays those pools on fresh compatible data

## Constructor

### `RegimeDiversifiedOpinionPools(sfd, random_state=42)`

| Argument | Meaning |
|---|---|
| `sfd` | SFD used to replay selected rounds |
| `random_state` | clustering/diversification random state |

If the SFD is manifest-driven, RDOP extracts the manifest automatically and uses the same experiment family during online replay.

## Offline Pipeline

### `offline_pipeline(confusion_metrics, ...)`

The offline phase starts from an experiment confusion table, usually:

```python
uel.experiment_confusion_metrics
```

and produces:

```python
dict[int, pl.DataFrame]
```

where each key is a regime id and each value is the selected permutation table for that regime.

### Current processing flow

The current offline path does four things:

1. sanity-filter rows with invalid benchmark values
2. remove outliers through IQR filtering
3. cluster the remaining rows into regimes
4. diversify each regime pool through PCA-space medoid-style selection

### Main arguments

| Argument | Meaning |
|---|---|
| `confusion_metrics` | experiment-wide benchmark table |
| `perf_cols` | columns used for filtering and clustering |
| `target_count` | target number of selected models per regime |
| `k_regimes` | requested number of regime clusters |
| `iqr_multiplier` | outlier-filter strength |
| `n_pca_components` | optional PCA dimension count |
| `n_pca_clusters` | diversification cluster count |

### Fallback behavior

RDOP has simple fallback behavior when filtering becomes too aggressive:

- if sanity filtering removes everything, RDOP falls back to a single regime `0`
- if outlier filtering removes everything, RDOP falls back to the sanity-filtered set

## Online Pipeline

### `online_pipeline(data, aggregation_method='mean', aggregation_threshold=None)`

The online phase reruns the selected pools and aggregates the model predictions inside each regime.

The returned frame contains replayed price data plus one aggregated prediction column per stored regime:

- `regime_0_prediction`
- `regime_1_prediction`
- `regime_2_prediction`
- ...

### Aggregation methods

Current built-in aggregation methods are:

- `mean`
- `median`
- `majority_vote`

`aggregation_threshold` acts as an optional cutoff on top of those aggregated outputs.

## Input Expectations

`online_pipeline()` passes the provided dataframe directly into `UniversalExperimentLoop`, so the input must be compatible with the SFD you used to build the pools.

In practice that usually means:

- a `pl.DataFrame` or compatible dataframe object
- the same kind of market-data columns the underlying SFD expects
- the same general prep contract used during the original experiment family

## Important Alignment Caveat

The current online implementation assumes that the per-regime replayed outputs align to the same row count when regime prediction columns are added to the final result frame.

In other words, RDOP works best when the selected regimes replay onto the same aligned surface. If different regime pools produce different retained row counts after replay, the current implementation can fail instead of reconciling them for you.

That is one reason to think of RDOP today as a research-layer cohort tool rather than a full downstream decision system.

## How To Think About RDOP

RDOP is best understood as:

- `Experiment` finds promising rounds
- `Benchmark` helps characterize them
- `RDOP` groups and diversifies them
- downstream systems decide how to validate and act on the resulting regime outputs

It is not a substitute for downstream decision logic.

## Read Next

- Continue to [Trainer](Trainer.md) if you want to promote selected rounds individually into reusable sensors.
- Continue to [Log](Log.md) and [Benchmark](Benchmark.md) if you need the analysis surfaces RDOP consumes upstream.
