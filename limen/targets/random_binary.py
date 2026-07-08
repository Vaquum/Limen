import numpy as np
import polars as pl


class RandomBinaryTarget:

    '''Uniformly random binary target, used as a noise benchmark.'''

    def __init__(self,
                 train_data: pl.DataFrame,
                 target_name: str,
                 random_state: int | None = None) -> None:

        '''
        Store the target column name; no fitting is performed.

        Args:
            train_data (pl.DataFrame): Training split (not used; kept for interface consistency)
            target_name (str): Name of the target column to create
            random_state (int | None): Seed for the random number generator. When set,
                labels are reproducible across runs given the same data sizes and call order
        '''

        super().__init__()

        self.target_name = target_name
        self._rng = np.random.default_rng(random_state)

    def transform(self, data: pl.DataFrame) -> pl.DataFrame:

        '''
        Apply random binary labels.

        Args:
            data (pl.DataFrame): Split to transform

        Returns:
            pl.DataFrame: Data with target column added
        '''

        labels = self._rng.integers(0, 2, size=data.height, dtype=np.uint8)
        return data.with_columns(pl.Series(self.target_name, labels, dtype=pl.UInt8))
