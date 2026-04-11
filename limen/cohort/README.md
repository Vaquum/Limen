# `limen.cohort`

> Inference-time cohort surfaces for aggregating decoder outputs from completed Limen experiments.

## Canonical docs

- [Cohort](../../docs/Cohort.md)
- [Regime-Diversified Opinion Pools](../../docs/Regime-Diversified-Opinion-Pools.md)
- [Trainer](../../docs/Trainer.md)

## What this package owns

This package owns two cohort-level inference surfaces:

1. `Cohort` for direct multi-member aggregation from one experiment + selected permutations
2. `RegimeDiversifiedOpinionPools` (RDOP) for regime-aware pool construction and per-regime aggregation

It does **not** own:

- experiment search execution (owned by `limen.experiment`)
- raw market data infrastructure
- downstream decisioning/execution logic (outside Limen)

## Key entry points

| Entry point | Use it when | Notes |
|-------------|-------------|-------|
| `Cohort` | You want to aggregate selected decoders from one completed experiment at inference time | Supports probability-weighted and majority-vote fallback modes |
| `RegimeDiversifiedOpinionPools` | You want regime-diversified cohort construction and per-regime inference | Full offline + online RDOP workflow |

## Package map

```text
cohort/
├── cohort.py                # Cohort constructor + aggregation logic
├── regime_pools.py          # RDOP implementation
├── cohort_usecase.py        # Realistic integration/smoke tests for Cohort behavior
└── cohort_walkthrough.ipynb # Block-based notebook walkthrough with visuals
```

## Cohort quick behavior

- Construct from exactly one experiment source (`experiment_id` or `experiment_log_path`)
- Validate selected `permutation_ids` and enforce single-architecture selection
- Infer aggregation mode from architecture capability:
  - `probability_weighted` when probability output is supported
  - `majority_vote` fallback otherwise
- `predict(...)` returns ndarray/tuple contract
- `__call__(...)` returns decoder-compatible dict contract

See [Cohort](../../docs/Cohort.md) for full contract and examples.

## RDOP quick behavior

- `offline_pipeline()` selects and diversifies candidate models by regime
- `online_pipeline()` loads selected models and emits per-regime outputs
- Intended for regime-aware downstream usage, not direct execution decisions

See [Regime-Diversified Opinion Pools](../../docs/Regime-Diversified-Opinion-Pools.md).

## Adjacent modules

- `limen.experiment` provides experiment logs, Trainer, and Sensor reconstruction used by Cohort flows
- `limen.log` provides analysis surfaces commonly used before RDOP selection
- `limen.sfd` and reference architectures define member model output behavior

## Read next

- [Cohort](../../docs/Cohort.md)
- [Regime-Diversified Opinion Pools](../../docs/Regime-Diversified-Opinion-Pools.md)
- [Trainer](../../docs/Trainer.md)
