import polars as pl
from typing_extensions import override

from limen.targets.vol_normalized_return import PARKINSON_SCALE
from limen.targets.vol_normalized_return import VolNormalizedReturnTarget


class ForwardVolNormalizedReturnTarget(VolNormalizedReturnTarget):

    '''Forward log return normalized by current Parkinson volatility.'''

    def __init__(
        self,
        train_data: pl.DataFrame,
        target_name: str,
        periods: int = 1,
        absolute: bool = False,
        halflife: int = 50,
        min_periods: int = 150,
        high_col: str = 'high',
        low_col: str = 'low',
        open_col: str = 'open',
        close_col: str = 'close',
    ) -> None:

        '''
        Store target settings and validate Parkinson volatility on train.

        Args:
            train_data (pl.DataFrame): Training split used for the volatility sanity gate
            target_name (str): Name of the target column to create
            periods (int): Forward return horizon in bars
            absolute (bool): If True, use absolute forward log return
            halflife (int): EWMA half-life in bars
            min_periods (int): Warmup before volatility values are emitted
            high_col (str): High price column
            low_col (str): Low price column
            open_col (str): Open price column used for dirty-bar filtering
            close_col (str): Close price column
        '''

        if periods <= 0:
            raise ValueError('ForwardVolNormalizedReturnTarget periods must be positive')

        self.periods = periods
        self.absolute = absolute
        super().__init__(
            train_data=train_data,
            target_name=target_name,
            high_col=high_col,
            low_col=low_col,
            open_col=open_col,
            close_col=close_col,
            halflife=halflife,
            min_periods=min_periods,
        )

    @override
    def transform(self, data: pl.DataFrame) -> pl.DataFrame:

        '''
        Compute `log(close.shift(-periods) / close) / parkinson_sigma`.

        Args:
            data (pl.DataFrame): Split to transform; must have open/high/low/close columns

        Returns:
            pl.DataFrame: Cleaned data with target column added
        '''

        return self._with_forward_target(self._clean_bars(data))

    def _with_forward_target(self, data: pl.DataFrame) -> pl.DataFrame:
        temp_cols = [
            '_limen_fvnr_forward_log_return',
            '_limen_fvnr_parkinson_variance',
            '_limen_fvnr_sigma',
            '_limen_fvnr_safe_sigma',
        ]
        forward_return = (
            pl.col(self.close_col).shift(-self.periods).log()
            - pl.col(self.close_col).log()
        )
        if self.absolute:
            forward_return = forward_return.abs()

        return (
            data
            .with_columns(
                [
                    forward_return.alias(temp_cols[0]),
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
                .alias(temp_cols[3])
            )
            .with_columns((pl.col(temp_cols[0]) / pl.col(temp_cols[3])).alias(self.target_name))
            .drop(temp_cols)
        )
