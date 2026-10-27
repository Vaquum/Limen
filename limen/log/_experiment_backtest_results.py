from typing import Any, Literal
import polars as pl

from limen.backtest.backtest_snapshot import BACKTEST_SNAPSHOT_COLUMNS

__all__ = ['_prepare_snapshot_backtest_input', 'captured_metrics', 'experiment_backtest_results']

_EVENT_COLUMNS = ['total_return', 'ending_equity', 'max_drawdown', 'net_pnl', 'gross_pnl', 'fees', 'slippage', 'funding_pnl', 'completed_trades', 'open_trades', 'episode_win_rate', 'avg_absolute_exposure']


def captured_metrics(source: object, names: list[str], prefix: str) -> pl.DataFrame:
    frame = getattr(source, '_captured_log', getattr(source, 'experiment_log', None))
    if not isinstance(frame, pl.DataFrame):
        raise ValueError('Captured experiment rows are required for aggregate metrics')
    if 'strict_mode_error' in frame.columns:
        frame = frame.filter(pl.col('strict_mode_error').is_null() | (pl.col('strict_mode_error') == ''))
    if frame.is_empty():
        return pl.DataFrame()
    columns = [prefix + name for name in names]
    if missing := set(columns).difference(frame.columns):
        raise ValueError(f'Captured {prefix} metrics are missing or incomplete: {sorted(missing)}')
    required = [prefix + name for name in names if not name.endswith('_mean_return_pct') and name not in ('avg_win_bps', 'avg_loss_bps', 'cvar_95_pnl_bps')]
    if frame.select(pl.any_horizontal(pl.col(required).is_null() | pl.col(required).cast(pl.Float64, strict=False).is_nan()).any()).item():
        raise ValueError(f'Successful rows lack required captured {prefix} metrics')
    return frame.select(columns).rename(dict(zip(columns, names, strict=True)))


def _prepare_snapshot_backtest_input(df: pl.DataFrame, *, prediction_kind: Literal['binary', 'directional'] = 'binary') -> pl.DataFrame:
    '''Require binary predictions unless directional interpretation is explicit.'''
    if prediction_kind not in ('binary', 'directional'):
        raise ValueError("prediction_kind must be 'binary' or 'directional'")

    def _valid(series: pl.Series) -> pl.Series:
        return series.drop_nulls().filter(series.drop_nulls().is_not_nan())

    def _is_binary(series: pl.Series) -> bool:
        valid = _valid(series)
        return bool(valid.is_in([0, 1]).all())

    def _to_numeric_strict(series: pl.Series, name: str) -> pl.Series:
        numeric = series.cast(pl.Float64, strict=False)
        missing = series.is_null()
        if series.dtype.is_float():
            missing = missing | series.is_nan()
        if ((numeric.is_null() | numeric.is_nan()) & ~missing).any():
            raise ValueError(f'snapshot backtest received non-numeric {name} values')
        return numeric

    def _is_unsupported_multiclass(series: pl.Series) -> bool:
        binary_class_count = 2
        valid = _valid(series)
        return bool((valid == valid.floor()).all() and (valid >= 0).all() and valid.n_unique() > binary_class_count)

    pred = _to_numeric_strict(df['predictions'], 'prediction')
    if 'actuals' in df.columns:
        actual = _to_numeric_strict(df['actuals'], 'actual')
        if _is_unsupported_multiclass(actual):
            raise ValueError("snapshot backtest does not support multiclass actuals; use binary predictions or supported directional scores with prediction_kind='directional'")
    if _is_binary(pred):
        return df.clone()
    if _is_unsupported_multiclass(pred):
        raise ValueError("snapshot backtest does not support multiclass predictions; use binary predictions or supported directional scores with prediction_kind='directional'")
    if prediction_kind == 'binary':
        raise ValueError("snapshot backtest requires binary predictions; use prediction_kind='directional' for directional scores or threshold probabilities upstream")
    return df.with_columns((pred.fill_nan(None).fill_null(float('-inf')) > 0).cast(pl.Int64).alias('predictions'))


def experiment_backtest_results(self: Any) -> pl.DataFrame:
    '''Read captured round backtests, or captured fold backtests for walk-forward.'''
    folds = getattr(self, 'fold_results', None)
    if folds is not None and not folds.is_empty():
        metrics = {f'backtest_{key}': key for key in BACKTEST_SNAPSHOT_COLUMNS}
        if not set(metrics).issubset(folds.columns):
            metrics = {f'{key}_test': key for key in BACKTEST_SNAPSHOT_COLUMNS}
        if not set(metrics).issubset(folds.columns):
            if 'backtest_total_return' not in folds.columns:
                raise ValueError('Walk-forward fold results lack captured snapshot metrics')
            metrics = {key: key.removeprefix('backtest_') for key in folds.columns if key.startswith('backtest_')}
        return folds.select('id', 'fold', *metrics).rename(metrics)
    frame = getattr(self, '_captured_log', getattr(self, 'experiment_log', None))
    names = _EVENT_COLUMNS if isinstance(frame, pl.DataFrame) and all(f'backtest_{name}' in frame.columns for name in _EVENT_COLUMNS) else BACKTEST_SNAPSHOT_COLUMNS
    return captured_metrics(self, names, 'backtest_')
