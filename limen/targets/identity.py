import polars as pl


class IdentityTarget:

    '''Target class for columns already present in the data.'''

    def __init__(self, train_data: pl.DataFrame, target_name: str) -> None:

        '''
        Validate the target column exists; no fitting is performed.

        Args:
            train_data (pl.DataFrame): Training split; used to validate the target column exists
            target_name (str): Name of an existing target column
        '''

        if target_name not in train_data.columns:
            raise ValueError(
                f"Target column '{target_name}' not found in training data. "
                f"Available columns: {train_data.columns}"
            )
        self.target_name = target_name

    def transform(self, data: pl.DataFrame) -> pl.DataFrame:

        '''
        Return data unchanged after validating the target column is present.

        Args:
            data (pl.DataFrame): Split to transform

        Returns:
            pl.DataFrame: Data unchanged
        '''

        if self.target_name not in data.columns:
            raise ValueError(
                f"Target column '{self.target_name}' not found in split data. "
                f"Available columns: {data.columns}"
            )
        return data
