import math
from typing import cast

import polars as pl


PARKINSON_SCALE = 4.0 * math.log(2.0)
SIGMA_RATIO_MIN = 0.9
SIGMA_RATIO_MAX = 1.1


class VolNormalizedReturnTarget:

    '''Continuous target as log return normalized by prior Parkinson volatility.'''

    def __init__(self,
                 train_data: pl.DataFrame,
                 target_name: str,
                 high_col: str = 'high',
                 low_col: str = 'low',
                 open_col: str = 'open',
                 close_col: str = 'close',
                 halflife: int = 50,
                 min_periods: int = 150) -> None:

        '''
        Store target settings and validate Parkinson against close-to-close volatility on train.

        Args:
            train_data (pl.DataFrame): Training split used for the volatility sanity gate
            target_name (str): Name of the target column to create
            high_col (str): High price column
            low_col (str): Low price column
            open_col (str): Open price column used for dirty-bar filtering
            close_col (str): Close price column
            halflife (int): EWMA half-life in bars
            min_periods (int): Warmup before volatility values are emitted
        '''

        if halflife <= 0:
            raise ValueError('VolNormalizedReturnTarget halflife must be positive')
        if min_periods <= 0:
            raise ValueError('VolNormalizedReturnTarget min_periods must be positive')

        self.target_name = target_name
        self.high_col = high_col
        self.low_col = low_col
        self.open_col = open_col
        self.close_col = close_col
        self.halflife = halflife
        self.min_periods = min_periods

        train = self._clean_bars(train_data)
        median_ratio = self._median_parkinson_to_close_sigma_ratio(train)
        if median_ratio is not None and not (SIGMA_RATIO_MIN <= median_ratio <= SIGMA_RATIO_MAX):
            raise ValueError(
                'median Parkinson-to-close volatility ratio '
                f'{median_ratio:.6f} is outside [{SIGMA_RATIO_MIN}, {SIGMA_RATIO_MAX}]'
            )

    def transform(self, data: pl.DataFrame) -> pl.DataFrame:

        '''
        Compute `log(close / close.shift(1)) / parkinson_sigma.shift(1)`.

        Args:
            data (pl.DataFrame): Split to transform; must have open/high/low/close columns

        Returns:
            pl.DataFrame: Cleaned data with target column added
        '''

        return self._with_target(self._clean_bars(data))

    def _clean_bars(self, data: pl.DataFrame) -> pl.DataFrame:
        self._validate_columns(data)
        return data.filter(
            (pl.col(self.high_col) >= pl.max_horizontal(self.open_col, self.close_col))
            & (pl.col(self.low_col) <= pl.min_horizontal(self.open_col, self.close_col))
        )

    def _validate_columns(self, data: pl.DataFrame) -> None:
        required = {self.high_col, self.low_col, self.open_col, self.close_col}
        missing = sorted(required - set(data.columns))
        if missing:
            raise ValueError(f"VolNormalizedReturnTarget missing columns: {missing}")

    def _with_target(self, data: pl.DataFrame) -> pl.DataFrame:
        temp_cols = [
            '_limen_vnr_log_return',
            '_limen_vnr_parkinson_variance',
            '_limen_vnr_sigma',
            '_limen_vnr_prior_sigma',
        ]
        return (
            data
            .with_columns(
                [
                    (pl.col(self.close_col).log() - pl.col(self.close_col).shift(1).log())
                        .alias(temp_cols[0]),
                    (((pl.col(self.high_col).log() - pl.col(self.low_col).log()) ** 2) / PARKINSON_SCALE)
                        .alias(temp_cols[1]),
                ]
            )
            .with_columns(
                pl.col(temp_cols[1])
                .ewm_mean(half_life=self.halflife, min_samples=self.min_periods)
                .clip(lower_bound=0.0)
                .sqrt()
                .alias(temp_cols[2])
            )
            .with_columns(
                pl.when(pl.col(temp_cols[2]) == 0.0)
                .then(None)
                .otherwise(pl.col(temp_cols[2]))
                .shift(1)
                .alias(temp_cols[3])
            )
            .with_columns((pl.col(temp_cols[0]) / pl.col(temp_cols[3])).alias(self.target_name))
            .drop(temp_cols)
        )

    def _median_parkinson_to_close_sigma_ratio(self, data: pl.DataFrame) -> float | None:
        if data.is_empty():
            return None

        stats = (
            data
            .with_columns(
                [
                    (((pl.col(self.high_col).log() - pl.col(self.low_col).log()) ** 2) / PARKINSON_SCALE)
                        .alias('_limen_vnr_p_var'),
                    ((pl.col(self.close_col).log() - pl.col(self.close_col).shift(1).log()) ** 2)
                        .alias('_limen_vnr_c_var'),
                ]
            )
            .with_columns(
                [
                    pl.col('_limen_vnr_p_var')
                    .ewm_mean(half_life=self.halflife, min_samples=self.min_periods)
                    .clip(lower_bound=0.0)
                    .sqrt()
                    .alias('_limen_vnr_p_sigma'),
                    pl.col('_limen_vnr_c_var')
                    .ewm_mean(half_life=self.halflife, min_samples=self.min_periods)
                    .clip(lower_bound=0.0)
                    .sqrt()
                    .alias('_limen_vnr_c_sigma'),
                ]
            )
            .with_columns(
                pl.when(pl.col('_limen_vnr_c_sigma') == 0.0)
                .then(None)
                .otherwise(pl.col('_limen_vnr_p_sigma') / pl.col('_limen_vnr_c_sigma'))
                .alias('_limen_vnr_sigma_ratio')
            )
        )

        ratios = (
            stats
            .select('_limen_vnr_sigma_ratio')
            .drop_nulls()
            .to_series()
            .to_list()
        )
        finite_ratios = [ratio for ratio in ratios if math.isfinite(ratio)]
        if not finite_ratios:
            return None
        return float(cast(float, pl.Series(finite_ratios).median()))
