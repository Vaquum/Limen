from typing import Any
import numpy as np
import pandas as pd

from limen.backtest.backtest_snapshot import BACKTEST_SNAPSHOT_COLUMNS

__all__ = ['experiment_backtest_results', 'captured_metrics', '_prepare_snapshot_backtest_input']

_EVENT_COLUMNS = ['total_return', 'ending_equity', 'max_drawdown', 'net_pnl', 'gross_pnl', 'fees', 'slippage', 'funding_pnl', 'completed_trades', 'open_trades', 'episode_win_rate', 'avg_absolute_exposure']


def captured_metrics(source: object, names: list[str], prefix: str) -> pd.DataFrame:
    frame = getattr(source, '_captured_log', getattr(source, 'experiment_log', None))
    if not isinstance(frame, pd.DataFrame):
        raise ValueError('Captured experiment rows are required for aggregate metrics')
    if 'strict_mode_error' in frame:
        frame = frame.loc[frame['strict_mode_error'].isna() | frame['strict_mode_error'].eq('')]
    if frame.empty:
        return pd.DataFrame()
    columns = [prefix + name for name in names]
    if missing := set(columns).difference(frame.columns):
        raise ValueError(f'Captured {prefix} metrics are missing or incomplete: {sorted(missing)}')
    required = [prefix + name for name in names if not name.endswith('_mean_return_pct') and name not in ('avg_win_bps', 'avg_loss_bps', 'cvar_95_pnl_bps')]
    if frame[required].isna().any().any():
        raise ValueError(f'Successful rows lack required captured {prefix} metrics')
    return frame[columns].rename(columns=dict(zip(columns, names, strict=True))).reset_index(drop=True).copy()


def _prepare_snapshot_backtest_input(df: pd.DataFrame) -> pd.DataFrame:

    '''Normalize binary or directional predictions for explicit snapshot replay.'''

    def _is_binary(series: pd.Series) -> bool:
        valid = series.dropna()
        return bool(valid.empty or valid.isin([0, 1]).all())

    def _to_numeric_strict(series: pd.Series, name: str) -> pd.Series:
        numeric = pd.to_numeric(series, errors='coerce')
        invalid = numeric.isna() & ~series.isna()

        if invalid.any():
            raise ValueError(f'snapshot backtest received non-numeric {name} values')

        return numeric

    def _is_unsupported_multiclass(series: pd.Series) -> bool:
        BINARY_CLASS_COUNT = 2
        valid = series.dropna()
        if valid.empty:
            return False

        integer_like = np.isclose(valid, np.round(valid)).all()
        unique_count = pd.Index(np.round(valid).astype(int)).nunique()

        return bool(integer_like and (valid >= 0).all() and unique_count > BINARY_CLASS_COUNT)

    result = df.copy()
    pred = _to_numeric_strict(result['predictions'], 'prediction')

    if 'actuals' in result:
        actual = _to_numeric_strict(result['actuals'], 'actual')
        if _is_unsupported_multiclass(actual):
            raise ValueError('snapshot backtest does not support multiclass actuals; use binary predictions or directional regression scores')

    if _is_binary(pred):
        return result

    if _is_unsupported_multiclass(pred):
        raise ValueError(
            'snapshot backtest does not support multiclass predictions; use binary predictions or directional regression scores'
        )

    result['predictions'] = (pred > 0).astype(int)

    return result


def experiment_backtest_results(self: Any) -> pd.DataFrame:

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
        return folds.select('id', 'fold', *metrics).rename(metrics).to_pandas()

    frame = getattr(self, '_captured_log', getattr(self, 'experiment_log', None))
    names = _EVENT_COLUMNS if isinstance(frame, pd.DataFrame) and all(f'backtest_{name}' in frame for name in _EVENT_COLUMNS) else BACKTEST_SNAPSHOT_COLUMNS
    return captured_metrics(self, names, 'backtest_')
