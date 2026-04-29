import polars as pl


class ThresholdBinaryTarget:

    '''Binary target from a fixed threshold comparison on a source column.'''

    def __init__(self,
                 train_data: pl.DataFrame,
                 target_name: str,
                 source_column: str,
                 threshold: float) -> None:

        '''
        Store the fixed threshold; no fitting is performed.

        Args:
            train_data (pl.DataFrame): Training split (not used; kept for interface consistency)
            target_name (str): Name of the target column to create
            source_column (str): Column to compare against threshold
            threshold (float): Fixed comparison value; rows above this are labeled 1
        '''

        self.target_name = target_name
        self.source_column = source_column
        self.threshold = threshold

    def transform(self, data: pl.DataFrame, shift: int = -1) -> pl.DataFrame:

        '''
        Apply binary flag and optional temporal shift.

        Args:
            data (pl.DataFrame): Split to transform
            shift (int): Periods to shift the label (negative = forward-looking)

        Returns:
            pl.DataFrame: Data with target column added
        '''

        return data.with_columns(
            (pl.col(self.source_column) > self.threshold)
                .cast(pl.UInt8)
                .shift(shift)
                .alias(self.target_name)
        )
