# `limen.cohort`

> Select and aggregate a diversified pool of models across market regimes using clustering and ensemble opinion pooling.

## Responsibilities

Owns the offline model-selection pipeline (filter → cluster → diversify) and the online prediction-aggregation pipeline (re-train selected models, merge predictions, aggregate).
Does **not** own raw data fetching, individual model training logic, or the base experiment loop — it orchestrates `UniversalExperimentLoop` and `Manifest` for each regime.

## Key concepts

- **RegimeDiversifiedOpinionPools (RDOP)** – top-level class; combines offline and online pipelines.
- **Offline pipeline** – filters outliers from a confusion-metrics DataFrame, clusters models by performance profile (KMeans), then picks a diverse representative subset per regime via PCA + medoid selection.
- **Online pipeline** – re-runs the selected models on fresh data, collects per-model predictions, then reduces them to a single prediction series per regime via a configurable aggregation strategy.
- **Regime** – a KMeans cluster of models with similar performance characteristics, identified during offline analysis.
- **Opinion pooling** – combining multiple model predictions via mean, median, or majority-vote aggregation.

## Entry points

| What | Where | When you'd call it |
|------|-------|--------------------|
| `RegimeDiversifiedOpinionPools` | `regime_pools.py` | Instantiate with an SFD, then call `.offline_pipeline()` followed by `.online_pipeline()` |
| `.offline_pipeline()` | `regime_pools.py` | Pass experiment confusion metrics to filter, cluster, and select models per regime |
| `.online_pipeline()` | `regime_pools.py` | Run selected models on new data and return a DataFrame of per-regime predictions |

## Dependencies

- **Internal:** `limen.experiment` (`UniversalExperimentLoop`, `Manifest`) for re-running model experiments per regime
- **External:** `numpy`, `pandas`, `polars`, `scikit-learn` (KMeans, PCA, StandardScaler)

## Quick orientation
```text
cohort/
└── regime_pools.py   # All RDOP logic: OfflineFilter, OfflineRegime,
                      # OfflineDiversification, OnlineModelLoader,
                      # AggregationStrategy, OnlineAggregation,
                      # RegimeDiversifiedOpinionPools
```

## Gotchas / things to know

- `offline_pipeline()` must be called before `online_pipeline()` — it populates `self.regime_pools`.
- If all rows are filtered out during sanity/outlier filtering, the pipeline falls back to a single regime (cluster 0) with unfiltered data.
- `k_regimes` is a ceiling: empty clusters are silently skipped, so fewer than `k_regimes` regimes may be returned.
- The SFD passed to RDOP must have a `manifest()` method if manifest-driven; otherwise `sfd.prep` / `sfd.model` must exist.
