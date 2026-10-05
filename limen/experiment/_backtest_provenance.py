from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from typing import cast

import polars as pl

SOURCE_ROW = '__backtest_source_row__'
PRICE_COLUMNS = ('datetime', 'open', 'high', 'low', 'close')


@dataclass(frozen=True)
class _SplitWitness:
    source_fingerprint: str
    source_offset: int
    source_count: int
    retained: range
    price_fingerprint: str
    error: str | None


@dataclass(frozen=True)
class _BacktestWitness:
    splits: tuple[_SplitWitness, ...]
    ml: bool


def _price_fingerprint(frame: pl.DataFrame) -> str:
    selected = frame.select(PRICE_COLUMNS)
    if isinstance(selected['datetime'].dtype, pl.Datetime):
        selected = selected.with_columns(pl.col('datetime').dt.cast_time_unit('us'))
    return sha256(selected.hash_rows(seed=0).to_numpy().tobytes()).hexdigest()


def source_splits(splits: list[pl.DataFrame]) -> list[pl.DataFrame]:
    out: list[pl.DataFrame] = []
    offset = 0
    for split in splits:
        if SOURCE_ROW in split.columns:
            raise ValueError('backtest source identity column collides with input')
        out.append(split.with_row_index(SOURCE_ROW, offset=offset))
        offset += split.height
    return out


def restore_source_rows(before: pl.DataFrame, after: pl.DataFrame) -> pl.DataFrame:
    if SOURCE_ROW not in before.columns:
        return after
    if before.height != after.height or 'datetime' not in after.columns or not before['datetime'].equals(after['datetime']):
        keys = [col for col in PRICE_COLUMNS if col in before.columns and col in after.columns]
        source = before.select([*keys, SOURCE_ROW])
        if not keys:
            return after
        source = source.join(after.select(keys), on=keys, how='semi')
        if source.select(keys).is_duplicated().any():
            return after
        return after.join(source, on=keys, how='left', maintain_order='left')
    return after.with_columns(before[SOURCE_ROW])


def capture_backtest(
    sources: list[pl.DataFrame] | None, retained_splits: list[pl.DataFrame], *, ml: bool,
) -> tuple[list[pl.DataFrame], _BacktestWitness | None]:
    if sources is None:
        return retained_splits, None
    witnesses: list[_SplitWitness] = []
    for source, retained in zip(sources, retained_splits, strict=True):
        ids = tuple(cast(list[int], retained[SOURCE_ROW].to_list())) if SOURCE_ROW in retained.columns and not retained[SOURCE_ROW].null_count() else ()
        offset = int(source[SOURCE_ROW][0]) if source.height else 0
        has_prices = all(col in source.columns and col in retained.columns for col in PRICE_COLUMNS)
        span = source.filter(pl.col(SOURCE_ROW).is_between(min(ids), max(ids))) if ids else source.clear()
        error = None
        if not has_prices:
            error = 'price_data_for_backtest requires source datetime/open/high/low/close'
        elif not ids:
            error = 'backtest source identity is missing'
        elif not span['datetime'].is_sorted() or span['datetime'].null_count():
            error = 'backtest source identity order mismatch'
        elif ml and span['datetime'].is_duplicated().any():
            error = 'backtest ambiguous alignment: duplicate source timestamps'
        elif tuple(span[SOURCE_ROW].to_list()) != ids:
            error = 'backtest censored interior or source identity order/count mismatch'
        elif not retained['datetime'].equals(span['datetime']):
            error = 'backtest source identity does not match evaluated rows'
        elif not ml and _price_fingerprint(retained) != _price_fingerprint(span):
            error = 'backtest source identity OHLC mismatch'
        witnesses.append(_SplitWitness(
            _price_fingerprint(source) if has_prices else '', offset, source.height,
            range(min(ids), max(ids) + 1) if ids else range(0), _price_fingerprint(span) if has_prices else '', error,
        ))
    clean = [split.drop(SOURCE_ROW) if SOURCE_ROW in split.columns else split for split in retained_splits]
    return clean, _BacktestWitness(tuple(witnesses), ml)


def attach_witness(data: Mapping[str, object], witness: _BacktestWitness | None) -> None:
    if witness is None:
        return
    mutable = cast(dict[str, object], data)
    mutable['_backtest_provenance'] = witness
    alignment = mutable.get('_alignment')
    if isinstance(alignment, dict):
        cast(dict[str, object], alignment)['_backtest_provenance'] = witness


def validate_witness(witness: object) -> _BacktestWitness:
    if not isinstance(witness, _BacktestWitness):
        raise ValueError('backtest provenance is required for configured TP/SL')
    selected = witness.splits[2:] if witness.ml else witness.splits
    for split in selected:
        if split.error is not None:
            raise ValueError(split.error)
    return witness


def preflight_backtest(data: Mapping[str, object]) -> None:
    if data.get('_backtest_configured'):
        witness = validate_witness(data.get('_backtest_provenance'))
        if witness.ml:
            price = data.get('price_data_for_backtest')
            if not isinstance(price, pl.DataFrame):
                raise ValueError('price_data_for_backtest is required for configured TP/SL')
            if _price_fingerprint(price) != witness.splits[2].price_fingerprint:
                raise ValueError('backtest source identity does not match price_data_for_backtest')


def replay_prices(source: pl.DataFrame, witness: object, count: int) -> pl.DataFrame:
    checked = validate_witness(witness)
    split = checked.splits[2]
    if 'datetime' not in source.columns:
        raise ValueError('backtest provenance requires datetime for replay')
    if _price_fingerprint(source) != split.source_fingerprint or source.height != split.source_count:
        raise ValueError('backtest source identity changed during replay')
    if len(split.retained) != count:
        raise ValueError('backtest source identity prediction length mismatch')
    prices = source.slice(split.retained.start - split.source_offset, count)
    if _price_fingerprint(prices) != split.price_fingerprint:
        raise ValueError('backtest source identity OHLC mismatch during replay')
    return prices


__all__ = ['PRICE_COLUMNS', 'SOURCE_ROW', 'attach_witness', 'capture_backtest', 'preflight_backtest', 'replay_prices', 'restore_source_rows', 'source_splits', 'validate_witness']
