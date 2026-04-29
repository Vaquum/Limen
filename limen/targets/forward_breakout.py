import polars as pl


class ForwardBreakoutTarget:

    '''Binary target for forward price breakouts above a percentage threshold.'''

    def __init__(self, train_data: pl.DataFrame, target_name: str) -> None:

        '''
        Args:
            train_data (pl.DataFrame): Training split (not used; kept for interface consistency)
            target_name (str): Name of the target column to create
        '''

        self.target_name = target_name

    def transform(self,
                  data: pl.DataFrame,
                  forward_periods: int = 24,
                  threshold: float = 0.02,
                  shift: int = -1) -> pl.DataFrame:

        '''
        Label 1 if price rises >= threshold over next forward_periods bars.

        Args:
            data (pl.DataFrame): Split to transform (must have 'close' column)
            forward_periods (int): Look-ahead window in bars
            threshold (float): Minimum return for positive label (0.02 = 2%)
            shift (int): Additional shift applied after labeling

        Returns:
            pl.DataFrame: Data with target column added
        '''

        future_price = pl.col('close').shift(-forward_periods)
        forward_return = (future_price - pl.col('close')) / pl.col('close')
        return data.with_columns(
            (forward_return >= threshold).cast(pl.UInt8).shift(shift).alias(self.target_name)
        )
