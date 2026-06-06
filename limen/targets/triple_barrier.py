import numpy as np
import polars as pl


class TripleBarrierTarget:

    '''Ternary label from Lopez de Prado's triple-barrier method.'''

    def __init__(self,
                 train_data: pl.DataFrame,
                 target_name: str,
                 upper_multiple: float = 1.0,
                 lower_multiple: float = 1.0,
                 max_horizon: int = 10,
                 span: int = 100,
                 min_periods: int = 100,
                 close_col: str = 'close') -> None:

        '''
        Store barrier settings and validate them; no fitting is performed.

        Args:
            train_data (pl.DataFrame): Training split (not used; kept for interface consistency)
            target_name (str): Name of the target column to create
            upper_multiple (float): Profit-taking barrier width in volatility units
            lower_multiple (float): Stop-loss barrier width in volatility units
            max_horizon (int): Vertical barrier as a forward window in bars
            span (int): EWMA span for the close-to-close return volatility estimate
            min_periods (int): Warmup before a volatility value is emitted
            close_col (str): Close price column used for the barrier path
        '''

        if upper_multiple <= 0:
            raise ValueError('TripleBarrierTarget upper_multiple must be positive')
        if lower_multiple <= 0:
            raise ValueError('TripleBarrierTarget lower_multiple must be positive')
        if max_horizon <= 0:
            raise ValueError('TripleBarrierTarget max_horizon must be positive')
        if span <= 0:
            raise ValueError('TripleBarrierTarget span must be positive')
        if min_periods <= 0:
            raise ValueError('TripleBarrierTarget min_periods must be positive')

        self.target_name = target_name
        self.upper_multiple = upper_multiple
        self.lower_multiple = lower_multiple
        self.max_horizon = max_horizon
        self.span = span
        self.min_periods = min_periods
        self.close_col = close_col

    def transform(self, data: pl.DataFrame) -> pl.DataFrame:

        '''
        Label each bar by the first barrier its forward close path touches.

        Args:
            data (pl.DataFrame): Split to transform; must have the close column

        Returns:
            pl.DataFrame: Data with a ternary target column added (1 upper, -1 lower,
                0 vertical; null during volatility warmup and when the vertical barrier
                extends past the available data)
        '''

        self._validate_columns(data)
        close = data[self.close_col].to_numpy().astype(float, copy=False)
        label, valid = self._first_touch(close, self._volatility(data))

        return (
            data
            .with_columns(
                [
                    pl.Series('_limen_tbl_label', label, dtype=pl.Int8),
                    pl.Series('_limen_tbl_valid', valid, dtype=pl.Boolean),
                ]
            )
            .with_columns(
                pl.when(pl.col('_limen_tbl_valid'))
                .then(pl.col('_limen_tbl_label'))
                .otherwise(None)
                .alias(self.target_name)
            )
            .drop(['_limen_tbl_label', '_limen_tbl_valid'])
        )

    def _validate_columns(self, data: pl.DataFrame) -> None:
        if self.close_col not in data.columns:
            raise ValueError(f"TripleBarrierTarget missing column: {self.close_col}")

    def _volatility(self, data: pl.DataFrame) -> np.ndarray:
        return (
            data
            .select(
                (pl.col(self.close_col) / pl.col(self.close_col).shift(1) - 1.0)
                .ewm_std(span=self.span, min_samples=self.min_periods)
                .alias('_limen_tbl_vol')
            )
            .get_column('_limen_tbl_vol')
            .to_numpy()
        )

    def _first_touch(self,
                     close: np.ndarray,
                     volatility: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = close.shape[0]
        upper = self.upper_multiple * volatility
        lower = self.lower_multiple * volatility
        label = np.zeros(n, dtype=np.int8)
        touched = np.zeros(n, dtype=bool)

        for k in range(1, self.max_horizon + 1):
            if k >= n:
                break
            forward_return = np.full(n, np.nan)
            forward_return[:n - k] = close[k:] / close[:n - k] - 1.0
            hit_upper = (~touched) & (forward_return >= upper)
            hit_lower = (~touched) & (forward_return <= -lower)
            label[hit_upper] = 1
            label[hit_lower] = -1
            touched = touched | hit_upper | hit_lower

        horizon_available = np.arange(n) <= (n - 1 - self.max_horizon)
        valid = (volatility > 0.0) & (touched | horizon_available)
        return label, valid
