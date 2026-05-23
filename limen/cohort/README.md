# `limen.cohort`

> Inference-time cohort surfaces for aggregating decoder outputs from completed Limen experiments.

## Canonical docs

- [Cohort](../../docs/Cohort.md)
- [Trainer](../../docs/Trainer.md)

## What this package owns

This package owns the cohort-level inference surface:

1. `Cohort` for direct multi-member aggregation from one experiment and selected permutations
2. single-file cohort (`sfc`) strategies for choosing those permutations from experiment artefacts

It does **not** own:

- experiment search execution (owned by `limen.experiment`)
- raw market data infrastructure
- downstream decisioning/execution logic (outside Limen)

## Key entry points

| Entry point                       | Use it when                                                                             | Notes                                                          |
| --------------------------------- | --------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| `Cohort`                        | You want to aggregate selected decoders from one completed experiment at inference time | Supports probability-weighted and majority-vote fallback modes |
| `limen.cohort.sfc.all.select`             | You want the default "use every round" selector                         | Preserves existing omitted-`permutation_ids` behavior          |
| `limen.cohort.sfc.top_n.select`           | You want a simple results-column ranker                                 | Requires `results.csv`                                         |
| `limen.cohort.sfc.backtest_pareto.select` | You want trading-metric Pareto selection                                | Uses backtest return/risk columns                              |
| `limen.cohort.sfc.diverse_metrics.select` | You want metric-diverse member selection                                | Uses PCA/KMeans medoids over metric columns                    |

## Package map

```text
cohort/
├── cohort.py                # Cohort constructor + aggregation logic
└── sfc/                     # Single-file cohort strategies
    ├── all.py               # select(context) -> ids
    ├── top_n.py             # select(context, *, column, n, ...) -> ids
    ├── backtest_pareto.py   # select(context, *, target_count, ...) -> ids
    └── diverse_metrics.py   # select(context, *, target_count, ...) -> ids
```

## Cohort quick behavior

- Construct from exactly one experiment source (`experiment_id` or `experiment_log_path`)
- Validate explicit or selector-produced `permutation_ids`
- Enforce single-architecture selection
- Infer aggregation mode from architecture capability:
  - `probability_weighted` when probability output is supported
  - `majority_vote` fallback otherwise
- `predict(...)` returns ndarray/tuple contract
- `__call__(...)` returns decoder-compatible dict contract

See [Cohort](../../docs/Cohort.md) for full contract and examples.

## Selector quick behavior

- Selectors are one-file strategies with one public `select(...)` function
- Selectors receive a context dict and return permutation IDs only
- Cohort owns all validation after selector execution
- Built-ins cover all-round, single-column rank, backtest Pareto, and metric diversity selection

## Adjacent modules

- `limen.experiment` provides experiment logs, Trainer, and Sensor reconstruction used by Cohort flows
- `limen.log` provides analysis surfaces commonly used before selector-driven promotion
- `limen.sfd` and reference architectures define member model output behavior

## Read next

- [Cohort](../../docs/Cohort.md)
- [Trainer](../../docs/Trainer.md)
