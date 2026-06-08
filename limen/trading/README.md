# `limen.trading`

> Provide the `Account` ledger primitive for tracking positions and balances over time.

## Canonical docs

- [Backtest](../../docs/Backtest.md)

## What this package owns

Owns the `Account` bookkeeping primitive.
Does **not** own strategy logic, signal generation, or the higher-level backtest orchestration that calls it.

## Key entry points

| Entry point | Use it when | Notes |
|-------------|-------------|-------|
| `Account` | You want a running ledger of long, short, cover, sell, and hold actions | Exported at the package root |
| `update_account()` | You want to record the next simulated action at a given price | The core mutating method on `Account` |

## Adjacent modules

- `Account` is a standalone primitive exported at the package root as `limen.Account`; no other in-package module currently consumes it.
- `limen.log` and `limen.backtest.backtest_snapshot` are downstream alternatives when you do not need a full ledger.

## Quick orientation

```text
trading/
└── account.py   # Account ledger, position tracking, and overflow guards
```

## Things to know

- The ledger supports `buy`, `sell`, `short`, `cover`, and `hold`.
- The `amount` semantics differ by action, so callers need to be explicit about whether they are expressing a USDT amount or the implied BTC quantity through price.
- Cached long and short totals exist for performance and are part of the account's correctness model.
- Precision and overflow guards are intentional. This package prefers explicit failure over silent drift.

## Read next

- [Backtest](../../docs/Backtest.md)
- [Log](../../docs/Log.md)
