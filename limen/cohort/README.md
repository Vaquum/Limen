# `limen.cohort`

> Select and aggregate an ensemble of models organised into market-regime clusters.

## Responsibilities

Implements the **Regime Diversified Opinion Pools (RDOP)** pipeline: filter and cluster models by performance regime offline, then retrain and aggregate their predictions online.

Does **not** own data fetching, feature engineering, or the base experiment loop — it delegates those to `limen.experiment`.

## Key concepts

- **RegimeDiversifiedOpinionPools** – top-level orchestrator exposing `offline_pipeline()` and `online_pipeline()`
- **OfflineFilter** – removes null-containing rows and IQR outliers from confusion metrics before clustering
- **OfflineRegime** – clusters models into `k` regimes using KMeans on performance metrics
- **OfflineDiversification** – selects a diverse, representative subset of models per regime via PCA + KMeans medoid selection
- **OnlineModelLoader** – extracts parameter sets from each regime's DataFrame and retrains models using `UniversalExperimentLoop`
- **AggregationStrategy** – combines predictions from all models in a regime using mean, median, or majority-vote
- **OnlineAggregation** – coordinates per-regime experiment runs and merges the resulting prediction columns

## Entry points

| What | Where | When you'd call it |
|------|-------|--------------------|
| `RegimeDiversifiedOpinionPools` | `regime_pools.py` | Instantiate with an SFD; call `offline_pipeline()` once on historical confusion metrics, then `online_pipeline()` to generate predictions |
| `offline_pipeline()` | `regime_pools.py` | Filters, clusters, and diversifies a set of models; stores regime pools in `self.regime_pools` |
| `online_pipeline()` | `regime_pools.py` | Retrains each regime's models on fresh data and returns a Polars DataFrame with per-regime prediction columns |

## Dependencies

- **Internal:** `limen.experiment.UniversalExperimentLoop`, `limen.experiment.Manifest`
- **External:** `numpy`, `pandas`, `polars`, `scikit-learn` (`KMeans`, `PCA`, `StandardScaler`)

## Quick orientation

```text
cohort/
└── regime_pools.py   # All RDOP classes live here (OfflineFilter, OfflineRegime,
                      # OfflineDiversification, OnlineModelLoader,
                      # AggregationStrategy, OnlineAggregation,
                      # RegimeDiversifiedOpinionPools)
```

## Gotchas / things to know

- `offline_pipeline()` must be called before `online_pipeline()`; it populates `self.regime_pools`
- If all rows are filtered out by the sanity filter, a single regime containing the raw data is returned without outlier removal or clustering
- PCA component count defaults to `None` (full variance); passing `n_pca_components` speeds up very large model sets
- `aggregation_threshold` in `online_pipeline()` maps directly to `AggregationStrategy.threshold`; `None` means return raw scores rather than binary decisions
