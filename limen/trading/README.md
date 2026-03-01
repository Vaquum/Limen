# `limen.trading`

> Track virtual portfolio positions (long, short, buy, sell, cover) and compute running account balances.

## Responsibilities

Owns the ledger model for a simulated trading account — recording every transaction and maintaining consistent USDT and BTC totals.
Does **not** own strategy logic, signal generation, or backtesting orchestration — `Account` is a pure bookkeeping primitive consumed by `limen.backtest`.

## Key concepts

- **Account** – stateful class; each `update_account()` call appends a new row to the internal ledger and updates cached position totals.
- **long_position** – net BTC owned from buys minus sells; cached for O(1) access.
- **short_position** – net BTC borrowed from shorts minus covers; cached for O(1) access.
- **net_position** – `long_position - short_position`; positive means net long, negative means net short.
- **Overflow protection** – hard caps at 1 trillion USDT and 1 billion BTC; raises `ValueError` if exceeded.

## Entry points

| What | Where | When you'd call it |
|------|-------|--------------------|
| `Account(start_usdt)` | `account.py` | Instantiate at the start of a backtest run |
| `account.update_account(action, amount, price_usdt)` | `account.py` | Called after each simulated trade decision |

## Dependencies

- **Internal:** consumed by `limen.backtest.BacktestSequential`
- **External:** `datetime` (stdlib), `math` (stdlib)

## Quick orientation
```text
trading/
└── account.py   # Account class — ledger, position tracking, overflow guards
```

## Gotchas / things to know

- `action` must be one of `'buy'`, `'sell'`, `'short'`, `'cover'`, or `'hold'`.
- For `'buy'`, `amount` is USDT to spend; for `'sell'`, `amount` is USDT to receive (BTC quantity is derived from `price_usdt`).
- For `'short'`, `amount` is USDT received from borrowing BTC; for `'cover'`, `amount` is USDT paid to repurchase BTC.
- `total_btc` tracks **only** the long position (actual BTC owned); it does not subtract borrowed BTC.
- BTC quantities are stored with 15-decimal precision to minimise floating-point drift over many trades.
