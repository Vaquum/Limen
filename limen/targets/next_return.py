import polars as pl


class NextReturnTarget:

    '''Continuous target as next-period percentage return on close price.'''

    def __init__(self, train_data: pl.DataFrame, target_name: str) -> None:

        '''
        Store the target column name; no fitting is performed.

        Args:
            train_data (pl.DataFrame): Training split (not used; kept for interface consistency)
            target_name (str): Name of the target column to create
        '''

        self.target_name = target_name

    def transform(self,
                  data: pl.DataFrame,
                  periods: int = 1,
                  scale: float = 100.0) -> pl.DataFrame:

        '''
        Compute percentage return over the next N periods.

        Args:
            data (pl.DataFrame): Split to transform (must have 'close' column)
            periods (int): Look-ahead window in bars
            scale (float): Multiplier applied to the raw return (100 = percentage points)

        Returns:
            pl.DataFrame: Data with target column added
        '''

        return data.with_columns(
            (((pl.col('close').shift(-periods) - pl.col('close')) / pl.col('close')) * scale)
                .alias(self.target_name)
        )
