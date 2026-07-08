import polars as pl


class EmaBreakoutTarget:

    '''Binary target flagging forward price breakouts above an EMA threshold.'''

    def __init__(self, train_data: pl.DataFrame, target_name: str) -> None:

        '''
        Store the target column name; no fitting is performed.

        Args:
            train_data (pl.DataFrame): Training split (not used; kept for interface consistency)
            target_name (str): Name of the target column to create
        '''

        super().__init__()

        self.target_name = target_name

    def transform(self,
                  data: pl.DataFrame,
                  target_col: str = 'close',
                  ema_span: int = 30,
                  breakout_delta: float = 0.2,
                  breakout_horizon: int = 3) -> pl.DataFrame:

        '''
        Label 1 when the price `breakout_horizon` bars ahead exceeds the EMA
        of `target_col` by at least `breakout_delta` of the EMA value.

        Args:
            data (pl.DataFrame): Split to transform (must have `target_col` column)
            target_col (str): Column name to analyze for breakouts
            ema_span (int): Period for EMA calculation
            breakout_delta (float): Threshold for breakout detection
            breakout_horizon (int): Forward window in bars

        Returns:
            pl.DataFrame: Data with target column added
        '''

        alpha = 2.0 / (ema_span + 1)
        label_expr = (
            pl.col(target_col).shift(-breakout_horizon)
            > pl.col(target_col).ewm_mean(alpha=alpha, adjust=False) * (1 + breakout_delta)
        ).cast(pl.UInt8)

        return data.with_columns(label_expr.alias(self.target_name))
