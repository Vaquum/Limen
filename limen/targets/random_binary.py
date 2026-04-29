import numpy as np
import polars as pl


class RandomBinaryTarget:

    '''Uniformly random binary target, used as a noise benchmark.'''

    def __init__(self, train_data: pl.DataFrame, target_name: str) -> None:

        '''
        Args:
            train_data (pl.DataFrame): Training split (not used; kept for interface consistency)
            target_name (str): Name of the target column to create
        '''

        self.target_name = target_name

    def transform(self, data: pl.DataFrame) -> pl.DataFrame:

        '''
        Apply random binary labels.

        Args:
            data (pl.DataFrame): Split to transform

        Returns:
            pl.DataFrame: Data with target column added
        '''

        return data.with_columns(
            pl.Series(self.target_name, np.random.randint(0, 2, size=data.height), dtype=pl.UInt8)
        )
