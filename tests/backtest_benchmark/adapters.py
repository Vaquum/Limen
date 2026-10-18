"""Public framework APIs with explicit close execution and zero costs."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from tests.backtest_benchmark.contract import (
    FRACTIONAL_UNIT, INITIAL_CASH, Result, transitions,
)


@dataclass
class Adapter:
    execute: object
    normalize: object
    scope: str
    settings: dict
    snapshot: object = None


def limen_adapter(data):
    from limen.backtest.backtest_snapshot import backtest_snapshot
    from limen.backtest.long_flat_strategy import long_flat_strategy

    count = len(data['close']) - 1
    close, opened, signal = (data[name][:count] for name in ('close', 'open', 'signal'))
    change = close - opened
    columns = {'open': opened, 'close': close, 'price_change': change,
               'predictions': signal}

    def execute():
        return long_flat_strategy(signal, opened, close, change,
                                  execution_lag_bars=1, fee_bps=0, slip_bps=0)

    def normalize(raw):
        held = raw.pos.astype(np.int8)
        entries = np.flatnonzero((held == 1) & (np.r_[0, held[:-1]] == 0)) - 1
        exits = np.flatnonzero((held == 1) & (np.r_[held[1:], 0] == 0))
        bars = np.r_[entries, exits]
        sides = np.r_[np.ones(len(entries), dtype=np.int8),
                      -np.ones(len(exits), dtype=np.int8)]
        order = np.argsort(bars, kind='stable')
        bars, sides = bars[order], sides[order]
        return Result(INITIAL_CASH * np.cumprod(1 + raw.net), held,
                      bars, sides, close[bars])

    def snapshot():
        return backtest_snapshot(columns, execution_lag_bars=1, fee_bps=0, slip_bps=0)

    return Adapter(execute, normalize, 'return kernel; public snapshot measured separately',
                   {'execution_lag_bars': 1}, snapshot)


def vectorbt_adapter(data, engine):
    import pandas as pd
    import vectorbt as vbt
    from vectorbt._engine import resolve_engine

    backend = 'rust' if engine == 'vectorbt_rust' else 'numba'
    if resolve_engine(backend) != backend:
        raise RuntimeError('Requested vectorbt backend was not selected')
    count = len(data['close']) - 1
    close = pd.Series(data['close'], index=pd.to_datetime(data['timestamp_ns']))
    desired = data['signal']
    bars, sides = transitions(desired)
    entries = np.zeros(len(close), dtype=bool)
    exits = np.zeros(len(close), dtype=bool)
    entries[bars[sides == 1]] = True
    exits[bars[sides == -1]] = True

    def execute():
        return vbt.Portfolio.from_signals(
            close, entries=entries, exits=exits, init_cash=INITIAL_CASH,
            size=np.inf, size_type='amount', direction='longonly',
            fees=0, fixed_fees=0, slippage=0, accumulate=False,
            engine=backend,
        )

    def normalize(raw):
        records = raw.orders.records
        order_bars = records['idx'].to_numpy(dtype=np.int64)
        order_sides = np.where(records['side'].to_numpy() == 0, 1, -1).astype(np.int8)
        assets = np.asarray(raw.assets(), dtype=float)
        held = np.r_[0, assets[:-1] > 0].astype(np.int8)[:count]
        return Result(np.asarray(raw.value(), dtype=float)[:count], held,
                      order_bars, order_sides, records['price'].to_numpy(dtype=float))

    return Adapter(execute, normalize, 'Portfolio.from_signals plus common output extraction',
                   {'backend': backend, 'size': 'infinite amount (all affordable units)'})


def _pandas_frame(data):
    import pandas as pd

    return pd.DataFrame({key.title(): data[key] for key in
                         ('open', 'high', 'low', 'close', 'volume')},
                        index=pd.to_datetime(data['timestamp_ns']))


def backtesting_adapter(data):
    from backtesting import Strategy
    from backtesting.lib import FractionalBacktest

    frame = _pandas_frame(data)
    desired = data['signal']
    count = len(frame) - 1

    class DesiredPosition(Strategy):
        def init(self):
            self.initial_account_equity = self.equity
            if desired[0]:
                self.buy(size=math.nextafter(1.0, 0.0))

        def next(self):
            target = desired[len(self.data) - 1]
            if target and not self.position:
                self.buy(size=math.nextafter(1.0, 0.0))
            elif not target and self.position:
                self.position.close()

    def execute():
        return FractionalBacktest(
            frame, DesiredPosition, cash=INITIAL_CASH, commission=0,
            spread=0, margin=1, trade_on_close=True, hedging=False,
            exclusive_orders=False, finalize_trades=False,
            fractional_unit=FRACTIONAL_UNIT,
        ).run()

    def normalize(raw):
        trades = raw['_trades']
        if raw['_strategy'].position:
            raise AssertionError('Backtesting.py left a terminal position open')
        entries = trades['EntryBar'].to_numpy(dtype=np.int64)
        exits = trades['ExitBar'].to_numpy(dtype=np.int64)
        bars = np.r_[entries, exits]
        sides = np.r_[np.ones(len(entries), dtype=np.int8),
                      -np.ones(len(exits), dtype=np.int8)]
        prices = np.r_[trades['EntryPrice'].to_numpy(dtype=float),
                       trades['ExitPrice'].to_numpy(dtype=float)]
        order = np.argsort(bars, kind='stable')
        held = np.zeros(count, dtype=np.int8)
        for entry, exit_bar in zip(entries, exits, strict=True):
            held[entry + 1:exit_bar + 1] = 1
        equity = raw['_equity_curve']['Equity'].to_numpy(dtype=float)[:count].copy()
        # run() backfills its unrecorded first slot from bar 1. Preserve the
        # actual broker observation at bar 0, before the zero-cost initial fill.
        equity[0] = raw['_strategy'].initial_account_equity
        return Result(equity,
                      held, bars[order], sides[order], prices[order])

    return Adapter(execute, normalize, 'FractionalBacktest.run including native statistics',
                   {'trade_on_close': True, 'finalize_trades': False,
                    'fractional_unit': FRACTIONAL_UNIT,
                    'first_equity_slot': 'observed broker equity in Strategy.init',
                    'allocation_fraction': math.nextafter(1.0, 0.0)})


def backtrader_adapter(data):
    import backtrader as bt

    frame = _pandas_frame(data)
    frame.columns = frame.columns.str.lower()
    desired = data['signal']
    count = len(frame) - 1
    timestamp_to_bar = {bt.date2num(stamp): bar for bar, stamp in enumerate(frame.index)}

    class DesiredPosition(bt.Strategy):
        def __init__(self):
            self.equity = []
            self.fills = []
            self.pending = None

        def next(self):
            self.equity.append(self.broker.getvalue())
            target = desired[len(self.data) - 1]
            if self.pending is not None:
                raise AssertionError('Market order did not settle on the next real bar')
            if target and not self.position:
                quantity = math.nextafter(self.broker.getcash() / self.data.close[0], 0.0)
                self.pending = self.buy(size=quantity)
            elif not target and self.position:
                self.pending = self.close()

        def notify_order(self, order):
            if order.status in (order.Submitted, order.Accepted):
                return
            if order.status != order.Completed:
                raise AssertionError(f'Backtrader order failed: {order.getstatusname()}')
            self.fills.append((timestamp_to_bar[order.executed.dt],
                               1 if order.isbuy() else -1, order.executed.price))
            self.pending = None

    def execute():
        cerebro = bt.Cerebro(stdstats=False)
        cerebro.adddata(bt.feeds.PandasData(dataname=frame))
        cerebro.addstrategy(DesiredPosition)
        cerebro.broker.setcash(INITIAL_CASH)
        cerebro.broker.setcommission(commission=0)
        cerebro.broker.set_coc(True)
        return cerebro.run(runonce=True, preload=True)[0]

    def normalize(raw):
        if raw.position or raw.pending is not None:
            raise AssertionError('Backtrader left a terminal position or order open')
        bars = np.asarray([fill[0] for fill in raw.fills], dtype=np.int64)
        sides = np.asarray([fill[1] for fill in raw.fills], dtype=np.int8)
        prices = np.asarray([fill[2] for fill in raw.fills], dtype=float)
        change = np.zeros(len(frame), dtype=np.int8)
        change[bars] = sides
        held = np.r_[0, np.cumsum(change)[:-1]].astype(np.int8)[:count]
        return Result(np.asarray(raw.equity, dtype=float)[:count], held, bars, sides, prices)

    return Adapter(execute, normalize, 'Cerebro.run; preload/runonce; standard observers disabled',
                   {'cheat_on_close': True, 'preload': True, 'runonce': True,
                    'standard_observers': False, 'fractional_quantity': True})


def make_adapter(engine, data):
    if engine == 'limen':
        return limen_adapter(data)
    if engine in ('vectorbt', 'vectorbt_rust'):
        return vectorbt_adapter(data, engine)
    if engine == 'backtesting_py':
        return backtesting_adapter(data)
    if engine == 'backtrader':
        return backtrader_adapter(data)
    raise ValueError(f'Unknown engine: {engine}')
