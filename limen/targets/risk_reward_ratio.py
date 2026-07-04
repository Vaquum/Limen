import polars as pl


EPSILON = 0.001
_REQUIRED_COLUMNS = ('capturable_breakout', 'max_drawdown')


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

        Raises:
            ValueError: If the required user-supplied columns are absent
        '''

        missing = [c for c in _REQUIRED_COLUMNS if c not in data.columns]
        if missing:
            raise ValueError(
                f"RiskRewardRatioTarget requires user-supplied columns missing "
                f"from data: {missing}. No Limen indicator or feature produces "
                f"them; they must be present in the input data."
            )

        return data.with_columns(
            (pl.col('capturable_breakout') / (pl.col('max_drawdown').abs() + EPSILON))
                .alias(self.target_name)
        ).drop(['capturable_breakout', 'max_drawdown'])
