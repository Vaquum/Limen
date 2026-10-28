# `limen.backtest`

> Evaluate aligned predictions through snapshot or event execution, with explicit product, costs and funding.

## Canonical docs

- [Backtest](../../docs/Backtest.md)
- [Benchmark](../../docs/Benchmark.md)
- [Log](../../docs/Log.md)

## In the research projects

The [Python project](../../docs/examples/python-research/README.md) constructs execution inputs and policy directly. The [Compiled project](../../docs/examples/compiled-research/README.md) binds them through its Manifest. Their `execution` scenarios retain native ledgers, declared clocks, product rules and recorded funding.

## What this package owns

Owns Limen's research execution, inventory/economics ledger and snapshot assumptions.
Signal generation and operational account management belong to their respective callers.

## Key entry points

| Entry point | Use case | Notes |
|-------------|-------------|-------|
| `backtest_snapshot` | Vectorized evaluation across rounds | `from limen.backtest import backtest_snapshot` |
| `long_flat_strategy` | Default execution model, or a template for a new strategy | Returns an `ExecutionResult`; `from limen.backtest import long_flat_strategy` |
| `short_flat_strategy` | Explicit binary short/flat snapshot approximation | Import from `limen.backtest.short_flat_strategy`; [contract](../../docs/Backtest.md#explicit-shortflat-snapshots) |
| `limen.backtest.funding` | Validated funding observations and event cash flows | Uses the declared funding clocks and product |
| `limen.backtest.observed_costs` | Recorded retrospective cost calibration and stress comparisons | Python Research; requires the pinned calibration and benchmark |
| `trade_execution` | Signed exposure with actual fill costs and persistent episodes | Accepts `TradeInputs` and `TradePolicy` with a `ProductSpec` |

The package root exports `backtest_snapshot`, `long_flat_strategy`, `ExecutionResult`, `trade_execution`, `TradeInputs`, `TradePolicy` and `ProductSpec`.

## Adjacent modules

- `limen.log` uses `backtest_snapshot()` to summarize experiment permutations.
- `limen.experiment` and `limen.sfd` sit upstream by producing the predictions that backtests consume.

## Quick orientation

| Module or family | Responsibility |
|---|---|
| `backtest_snapshot.py` | Vectorized snapshot evaluator (price validation + ledger) |
| `long_flat_strategy.py` | Default long-only execution model |
| `short_flat_strategy.py` | Explicit constant-notional short/flat snapshot model |
| `trade_contract.py` / `trade_execution.py` | Product, policy, event fills and inventory ledger |
| `funding.py` / `funding_presets.py` | Funding records and declared venue conventions |
| `observed_costs.py` | Cost calibration, dispositions and stress reports |

<!-- reference: package-backtest-layout source=limen/backtest/README.md region=layout:package role=layout -->
```text
backtest/
├── README.md
├── __init__.py
├── _cost_comparison.py
├── _cost_history.py
├── _cost_identity.py
├── _cost_statistics.py
├── _event_clock.py
├── _event_plan.py
├── _explicit_exit_strategy.py
├── _long_flat_tp_sl.py
├── _snapshot_execution.py
├── _snapshot_ledger.py
├── _snapshot_signals.py
├── backtest_snapshot.py
├── execution_events.py
├── funding_adapter.py
├── funding_presets.py
├── long_flat_strategy.py
├── observed_costs.py
├── short_flat_strategy.py
├── trade_contract.py
└── trade_execution.py
```

## Things to know

- Configure tunable fixed TP/SL on the manifest; [Backtest](../../docs/Backtest.md#tunable-take-profit-and-stop-loss) defines exits, OHLC validation, and replay requirements.
- `backtest_snapshot()` is the standard analysis path for vectorized experiment-sweep evaluation.
- Explicit `class_mapping` or `label_index` interprets multiclass/multilabel snapshot inputs; [Backtest](../../docs/Backtest.md#explicit-multiclass-and-multilabel-inputs) defines the contract.
- Snapshot return and cost outputs are basis-point scaled (`*_bps` columns), matching the basis-point fee and slippage inputs.

## Read next

- [Backtest](../../docs/Backtest.md)
- [Benchmark](../../docs/Benchmark.md)
- [Log](../../docs/Log.md)
