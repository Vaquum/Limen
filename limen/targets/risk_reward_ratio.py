import polars as pl


EPSILON = 0.001


class RiskRewardRatioTarget:

    '''Continuous target as the ratio of capturable breakout to absolute drawdown.'''

    def __init__(self, train_data: pl.DataFrame, target_name: str) -> None:

        '''
        Store the target column name; no fitting is performed.

        Args:
            train_data (pl.DataFrame): Training split (not used; kept for interface consistency)
            target_name (str): Name of the target column to create
        '''

        self.target_name = target_name

    def transform(self, data: pl.DataFrame) -> pl.DataFrame:

        '''
        Compute `capturable_breakout / (|max_drawdown| + EPSILON)` per row.

        Args:
            data (pl.DataFrame): Split to transform (must have `capturable_breakout`, `max_drawdown` columns)

        Returns:
            pl.DataFrame: Data with target column added
        '''

        return data.with_columns(
            (pl.col('capturable_breakout') / (pl.col('max_drawdown').abs() + EPSILON))
                .alias(self.target_name)
        )
