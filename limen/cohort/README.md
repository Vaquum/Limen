# `limen.cohort`

> Inference-time cohort surfaces for aggregating decoder outputs from completed Limen experiments.

## Canonical docs

- [Cohort](../../docs/Cohort.md)
- [Trainer](../../docs/Trainer.md)

## In the research projects

The [Python project](../../docs/examples/python-research/README.md) composes selected, already evaluated fitted objects through `Cohort.from_python`. The [Compiled project](../../docs/examples/compiled-research/README.md) `walk_forward` scenario binds reconstructed qualified folds. Member identity, output units and available context must agree before aggregation.

## What this package owns

This package owns the cohort-level inference surface:

1. `Cohort.from_python(sensors)` for live mean/vote/custom aggregation, and artifact-backed `Cohort` construction for selected experiment permutations
2. single-file cohort (`sfc`) strategies for choosing those permutations from experiment artifacts

It does **not** own:

- experiment search execution (owned by `limen.experiment`)
- raw market data infrastructure
- downstream decisioning/execution logic (outside Limen)

## Key entry points

| Entry point                       | Use case | Notes                                                          |
| --------------------------------- | --------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| `Cohort`                        | Aggregate retained Python fits or compatible reconstructed native members at inference time | Supports probability-weighted and majority-vote fallback modes |
| `BUILTIN_SELECTORS`             | Map selector names to built-in selector modules | Exported at the package root |
| `limen.cohort.sfc.all.select`             | Default "use every round" selector | Preserves existing omitted-`permutation_ids` behavior          |
| `limen.cohort.sfc.top_n.select`           | Results-column ranker | Requires `results.csv`                                         |
| `limen.cohort.sfc.backtest_pareto.select` | Trading-metric Pareto selection | Uses backtest return/risk columns                              |
| `limen.cohort.sfc.diverse_metrics.select` | Metric-diverse member selection | Uses PCA/KMeans medoids over metric columns                    |

## Package map

| Module or family | Responsibility |
|---|---|
| `cohort.py` | Cohort entry points + compiled aggregation |
| `_python_cohort.py` | Live composition and callback boundary |
| `sfc/` | Single-file cohort strategies |
| `all.py` | select(context) -> ids |
| `top_n.py` | select(context, *, column, n) -> ids |
| `backtest_pareto.py` | select(context, *, target_count) -> ids |
| `diverse_metrics.py` | select(context, *, target_count) -> ids |

<!-- reference: package-cohort-layout source=limen/cohort/README.md region=layout:package role=layout -->
```text
cohort/
├── README.md
├── __init__.py
├── _fold_sources.py
├── _python_cohort.py
├── _round_sources.py
├── cohort.py
└── sfc/
```

## Cohort quick behavior

For live fitted Sensors, start with `Cohort.from_python(sensors)`. Matching explicit
Sensor output IDs permit default mean; optional preparation, output mapping and
aggregation functions provide custom control. Compiled members require an explicit
mapper/common meaning/cutoff. See the [Python contract](../../docs/Cohort.md#python-cohorts)
for alignment, blocking, availability and live provenance.

The artifact-backed constructor follows these rules:

- Construct from exactly one experiment source (`experiment_id` or `experiment_log_path`)
- Validate explicit or selector-produced `permutation_ids`
- Bind each member to its selected source and architecture; admit mixed declared binary outputs
- Infer homogeneous aggregation from architecture capability; bind mixed aggregation from native binary declarations:
  - `probability_weighted` when probability output is supported
  - `majority_vote` fallback otherwise
- `predict(raw_klines)` returns one `BarPrediction` for the last bar
- `predict_all(raw_klines)` returns `list[BarPrediction]`
- `__call__(raw_klines)` aliases `predict_all(raw_klines)`

See [Cohort](../../docs/Cohort.md) for full contract and examples.

## Selector quick behavior

- Selectors are one-file strategies with one public `select(context)` function
- Selectors receive a context dict and return permutation IDs only
- Cohort owns all validation after selector execution
- Built-ins cover all-round, single-column rank, backtest Pareto, and metric diversity selection

## Adjacent modules

- `limen.experiment` owns manifests and parameter search; `limen.inference` owns Trainer and Sensor reconstruction used by Cohort flows
- `limen.log` provides analysis surfaces commonly used before selector-driven promotion
- `limen.sfd` and reference architectures define member model output behavior

## Read next

- [Cohort](../../docs/Cohort.md)
- [Trainer](../../docs/Trainer.md)
