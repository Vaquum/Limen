# `limen.trading`

> Simulate a trading account's position and balance through a sequence of buy, sell, short, and cover actions.

## Responsibilities

Provides the `Account` class, which maintains a full ledger of position events and computes running totals used by the backtest engine.

Does **not** own signal generation, backtesting strategy, or order routing — it is purely a bookkeeping primitive consumed by `limen.backtest.BacktestSequential`.

## Key concepts

- **Account** – stateful ledger; each `update_account()` call appends a row to every list in `self.account` and updates cached position totals
- **long_position** – BTC owned from cumulative buys minus sells (cached for O(1) access)
- **short_position** – BTC owed from cumulative borrows minus covers (cached for O(1) access)
- **net_position** – `long_position - short_position`; positive means net long
- **overflow protection** – hard limits on total USDT (1 trillion) and total BTC (1 billion) to guard against runaway simulations

## Entry points

| What | Where | When you'd call it |
|------|-------|--------------------|
| `Account` | `account.py` | Instantiate with `start_usdt`; call `update_account()` for each trade event |
| `Account.update_account()` | `account.py` | Record a `'buy'`, `'sell'`, `'short'`, `'cover'`, or `'hold'` action |
| `Account.long_position` | `account.py` | Read net long BTC held |
| `Account.short_position` | `account.py` | Read net short BTC owed |

## Dependencies

- **Internal:** none
- **External:** `datetime`, `math`

## Quick orientation

```text
trading/
└── account.py   # Account class — full position ledger
```

## Gotchas / things to know

- `update_account()` uses **USDT amounts** for `'buy'` and `'short'` but **BTC-equivalent amounts** only implicitly (derived from `amount / price`); be careful not to pass BTC directly
- All BTC quantities are stored with 15 decimal places; USDT totals are rounded to 2
- `TOLERANCE_BTC = 1e-14` prevents spurious errors when accumulated floating-point rounding makes `btc_to_sell` fractionally exceed `long_position`
- The `account` dict stores **all** history as lists; for long simulations this grows linearly — `BacktestSequential` is designed for evaluation, not production execution
