import hashlib
import re
from collections.abc import Callable
from typing import Protocol, TypedDict, cast

import numpy as np
import numpy.typing as npt
import polars as pl

from limen._optional import require_optional

Answers = npt.NDArray[np.int8]

PRICE_COLUMNS = ('open', 'high', 'low', 'close', 'volume')
# The counters Filang's dimensions are read from, under Limen's kline names. USDT traded is liquidity_sum,
# taker-buy USDT is the part of it the buyer-maker side did not trade, and trades is no_of_trades.
COUNTER_COLUMNS = ('liquidity_sum', 'maker_liquidity', 'no_of_trades')
# Filang asks for an event in `find`. This relation holds at every recorded close, so a condition can be
# asked on its own.
_EVERY_CLOSE = 'candle_relation(price(high), is: at_or_above, right: price(low))'
_KEEP = 6


class _Result(TypedDict):
    anchors: list[int]
    unknown: dict[str, list[int]]


class _Study(TypedDict):
    results: list[_Result]


class _Engine(Protocol):
    def load_csv(self, csv: str, interval_ms: int) -> None: ...
    def study(self, query: str) -> _Study: ...
    def close(self) -> None: ...


class _Recording:

    '''One frame loaded into the Filang engine, with the answers already asked of it.'''

    def __init__(self, frame: pl.DataFrame, interval_seconds: int) -> None:
        super().__init__()
        backend = require_optional('filang_engine', 'filang-engine', 'filang')
        self._error = cast(type[Exception], backend.FilangError)
        minutes = interval_seconds // 60
        self._bar = f'{minutes // 60}h' if minutes % 60 == 0 else f'{minutes}m'
        opens = frame['datetime'].dt.epoch('ms').to_numpy().astype(np.int64)
        self._row: dict[int, int] = {close + interval_seconds * 1000: row for row, close in enumerate(opens.tolist())}
        self._answers: dict[str, Answers] = {}
        self.engine = cast(Callable[[], _Engine], backend.Engine)()
        self.engine.load_csv(_csv(frame, opens), interval_seconds * 1000)

    def _rows(self, closes: list[int]) -> list[int]:
        try:
            return [self._row[close] for close in closes]
        except KeyError as error:
            raise RuntimeError('filang_pattern received an answer for a close that is not in the data') from error

    def ask(self, part: str, condition: bool) -> Answers:
        key = ('where ' if condition else 'find ') + part
        if key not in self._answers:
            head = f'on BTCUSDT {self._bar}\n'
            query = head + (f'find {_EVERY_CLOSE}\nwhere {part}' if condition else f'find {part}')
            try:
                results = self.engine.study(query)['results']
            except self._error as error:
                raise ValueError(f'definition is not a pattern Filang can answer here: {part}: {error}') from error
            if len(results) != 1:
                raise ValueError(f'definition must be one pattern; this part is {len(results)}: {part}')
            answer = np.zeros(len(self._row), dtype=np.int8)
            for closes in results[0]['unknown'].values():
                answer[self._rows(closes)] = -1
            answer[self._rows(results[0]['anchors'])] = 1
            self._answers[key] = answer
        return self._answers[key]


_RECORDINGS: dict[str, _Recording] = {}


def _csv(frame: pl.DataFrame, opens: npt.NDArray[np.int64]) -> str:
    prices = [frame[name].to_numpy().astype(np.float64) for name in PRICE_COLUMNS]
    lines: list[str]
    if not all(name in frame.columns for name in COUNTER_COLUMNS):
        lines = ['open_ms,open,high,low,close,volume']
        lines += [_bar(opens, prices, i) for i in range(frame.height)]
        return '\n'.join(lines) + '\n'
    traded = frame['liquidity_sum'].to_numpy().astype(np.float64)
    taker_buy = traded - frame['maker_liquidity'].to_numpy().astype(np.float64)
    trades = frame['no_of_trades'].to_numpy().astype(np.int64)
    if not (bool(np.isfinite(traded).all()) and bool((taker_buy >= 0).all()) and bool((trades >= 0).all())):
        raise ValueError('filang_pattern data has liquidity_sum, maker_liquidity or no_of_trades values that cannot be right')
    # Amounts are written with repr so the engine reads back the same float.
    lines = ['open_ms,open,high,low,close,base_volume,volume,taker_buy_volume,trade_count']
    lines += [','.join((_bar(opens, prices, i), repr(float(traded[i])), repr(float(taker_buy[i])), str(int(trades[i]))))
              for i in range(frame.height)]
    return '\n'.join(lines) + '\n'


def _bar(opens: npt.NDArray[np.int64], prices: list[npt.NDArray[np.float64]], row: int) -> str:
    values = [f'{column[row]:.2f}' for column in prices[:4]]
    return ','.join((str(int(opens[row])), *values, f'{prices[4][row]:.5f}'))


def _recording(frame: pl.DataFrame, interval_seconds: int) -> _Recording:
    columns = ('datetime', *PRICE_COLUMNS, *(name for name in COUNTER_COLUMNS if name in frame.columns))
    digest = hashlib.blake2b(digest_size=16)
    for name in columns:
        digest.update(frame[name].to_numpy().tobytes())
    key = f'{digest.hexdigest()}:{interval_seconds}'
    if key not in _RECORDINGS:
        while len(_RECORDINGS) >= _KEEP:
            _RECORDINGS.pop(next(iter(_RECORDINGS))).engine.close()
        _RECORDINGS[key] = _Recording(frame, interval_seconds)
    return _RECORDINGS[key]


def pattern_answers(frame: pl.DataFrame, definition: str, interval_seconds: int) -> Answers:

    '''
    Compute one Filang pattern's answer at every row: 1 true, 0 false, -1 unknown.

    Args:
        frame (pl.DataFrame): Klines dataset with 'datetime' and the price columns, and the counter columns when present
        definition (str): Filang pattern: an event, optionally followed by `where <condition>` clauses
        interval_seconds (int): Bar length of the dataset in seconds

    Returns:
        npt.NDArray[np.int8]: One answer per row of the frame
    '''

    recording = _recording(frame, interval_seconds)
    event, *conditions = re.split(r'\s+where\s+', definition.strip())
    answer = recording.ask(event, condition=False)
    for condition in conditions:
        part = recording.ask(condition, condition=True)
        # A query joins its parts: false if any is false, otherwise unknown if any is unknown, otherwise true.
        joined = np.ones(answer.shape[0], dtype=np.int8)
        joined[(answer == -1) | (part == -1)] = -1
        joined[(answer == 0) | (part == 0)] = 0
        answer = joined
    return answer
