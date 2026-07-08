import polars as pl


class NextBarUpTarget:

    '''Binary target for whether the next close is above the current close.'''

    def __init__(self, train_data: pl.DataFrame, target_name: str) -> None:

        '''
        Store the target column name; no fitting is performed.

        Args:
            train_data (pl.DataFrame): Training split (not used; kept for interface consistency)
            target_name (str): Name of the target column to create
        '''

        super().__init__()

        self.target_name = target_name

    def transform(self, data: pl.DataFrame) -> pl.DataFrame:

        '''
        Label 1 if the next close is higher than the current close.

        Args:
            data (pl.DataFrame): Split to transform (must have 'close' column)

        Returns:
            pl.DataFrame: Data with target column added
        '''

        return data.with_columns(
            (pl.col('close').shift(-1) > pl.col('close'))
                .cast(pl.UInt8)
                .alias(self.target_name)
        )
