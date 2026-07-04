import polars as pl

_REQUIRED_COLUMNS = ('exit_reason', 'exit_net_return')


class ExitQualityTarget:

    '''Categorical target scoring trade exits by reason and net return.'''

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
                  exit_quality_high: float = 1.0,
                  exit_quality_low: float = 0.2,
                  exit_quality_medium: float = 0.5) -> pl.DataFrame:

        '''
        Score each closed trade as high, low, or medium quality based on its
        `exit_reason` and `exit_net_return`.

        Args:
            data (pl.DataFrame): Split to transform (must have `exit_reason`, `exit_net_return` columns)
            exit_quality_high (float): Score for profitable target hits or trailing-stop exits
            exit_quality_low (float): Score for stop-loss or unprofitable timeout exits
            exit_quality_medium (float): Score for neutral exits

        Returns:
            pl.DataFrame: Data with target column added

        Raises:
            ValueError: If the required user-supplied columns are absent
        '''

        missing = [c for c in _REQUIRED_COLUMNS if c not in data.columns]
        if missing:
            raise ValueError(
                f"ExitQualityTarget requires user-supplied columns missing "
                f"from data: {missing}. No Limen indicator or feature produces "
                f"them; they must be present in the input data."
            )

        return data.with_columns(
            pl.when((pl.col('exit_reason').is_in(['target_hit', 'trailing_stop'])) & (pl.col('exit_net_return') > 0))
                .then(pl.lit(exit_quality_high))
                .when((pl.col('exit_reason') == 'stop_loss') | ((pl.col('exit_reason') == 'timeout') & (pl.col('exit_net_return') < 0)))
                .then(pl.lit(exit_quality_low))
                .otherwise(pl.lit(exit_quality_medium))
                .alias(self.target_name)
        ).drop(['exit_reason', 'exit_net_return'])
