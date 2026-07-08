import polars as pl


class QuantileBinaryTarget:

    '''Binary target label from training-set quantile cutoff.'''

    def __init__(self,
                 train_data: pl.DataFrame,
                 target_name: str,
                 source_column: str,
                 quantile: float) -> None:

        '''
        Fit quantile cutoff on training data.

        Args:
            train_data (pl.DataFrame): Training split to compute cutoff from
            target_name (str): Name of the target column to create
            source_column (str): Column used to compute the quantile threshold
            quantile (float): Top-N fraction for positive label (0.3 = top 30%)
        '''

        super().__init__()

        self.target_name = target_name
        self.source_column = source_column
        self.cutoff: float = train_data.select(
            pl.col(source_column).quantile(1.0 - quantile)
        ).item()

    def transform(self, data: pl.DataFrame, shift: int = 0) -> pl.DataFrame:

        '''
        Apply binary flag and optional temporal shift.

        Args:
            data (pl.DataFrame): Split to transform
            shift (int): Periods to shift the label (negative = forward-looking)

        Returns:
            pl.DataFrame: Data with target column added
        '''

        return data.with_columns(
            (pl.col(self.source_column) > self.cutoff)
                .cast(pl.UInt8)
                .shift(shift)
                .alias(self.target_name)
        )
