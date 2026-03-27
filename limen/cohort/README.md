# `limen.cohort`

> Build decoder cohorts from finished experiments and aggregate their opinions by regime.

## Canonical docs

- [Regime-Diversified Opinion Pools](../../docs/Regime-Diversified-Opinion-Pools.md)
- [Log](../../docs/Log.md)

## What this package owns

Owns Limen's RDOP implementation: offline filtering, clustering, diversification, model reloading, and per-regime prediction aggregation.
Does **not** own the original experiment runs, raw market data, or downstream trading decision logic.

## Key entry points

| Entry point | Use it when | Notes |
|-------------|-------------|-------|
| `RegimeDiversifiedOpinionPools` | You want the full offline-plus-online cohort workflow | Exported at the package root |
| `offline_pipeline()` | You want to select representative models by regime from experiment results | Runs before any online inference step |
| `online_pipeline()` | You want fresh predictions from the selected regime pools | Produces per-regime prediction columns |

## Adjacent modules

- `limen.log` produces the confusion-metrics tables RDOP commonly starts from.
- `limen.experiment` is reused to rerun or retrain models during the online stage.
- `Nexus`, downstream from Limen, is where trading decisions belong. This package stops at cohort outputs.

## Quick orientation

```text
cohort/
└── regime_pools.py   # Offline filtering, clustering, diversification,
                      # online model loading, aggregation, and RDOP
```

## Things to know

- `offline_pipeline()` must run before `online_pipeline()`.
- The implementation currently produces per-regime outputs. It does not add a dynamic regime selector or downstream decision policy by itself.
- Empty or unstable clustering results are handled conservatively, including fallback to fewer regimes when needed.
- Manifest-driven SFDs work here, but custom `prep` and `model` SFDs can also work if they expose the required callable surface.

## Read next

- [Regime-Diversified Opinion Pools](../../docs/Regime-Diversified-Opinion-Pools.md)
- [Log](../../docs/Log.md)
- [Trainer](../../docs/Trainer.md)
