import numpy as np
import polars as pl
from scipy.stats import norm


class RankGaussScaler:

    '''Rank transformation to Gaussian distribution via inverse normal CDF.'''

    def __init__(self,
                 x_train: pl.DataFrame,
                 n_quantiles: int = 1000) -> None:

        '''
        Fit scaler by storing training quantile boundaries per column.

        Args:
            x_train (pl.DataFrame): Training data
            n_quantiles (int): Number of quantile bins for rank interpolation

        '''

        if n_quantiles < 1:
            raise ValueError(
                f"n_quantiles must be >= 1, got {n_quantiles}"
            )

        self._n_quantiles = n_quantiles
        self._quantiles: dict[str, np.ndarray] = {}

        quantile_points = np.linspace(0, 1, n_quantiles + 1)

        for col in x_train.columns:
            if x_train[col].dtype == pl.Datetime or not x_train[col].dtype.is_numeric():
                continue

            values = x_train[col].cast(pl.Float64).drop_nulls().drop_nans().to_numpy()
            if len(values) == 0:
                continue

            self._quantiles[col] = np.quantile(values, quantile_points)


    def transform(self, df: pl.DataFrame) -> pl.DataFrame:

        '''
        Compute rank-based Gaussian transformation.

        For each column: rank values against training quantiles,
        convert to uniform [eps, 1-eps], apply inverse normal CDF.
        NaN values are preserved as NaN in the output.

        Args:
            df (pl.DataFrame): Data to transform

        Returns:
            pl.DataFrame: Transformed data with approximately normal distributions
        '''

        eps = 1.0 / (2 * self._n_quantiles)
        transformed_cols: list[pl.Series] = []

        for col, quantiles in self._quantiles.items():
            if col not in df.columns:
                continue

            values = df[col].to_numpy().astype(np.float64)
            nan_mask = np.isnan(values)

            ranks = np.searchsorted(quantiles, values, side='right')
            uniform = np.clip(ranks / len(quantiles), eps, 1 - eps)
            gaussian = norm.ppf(uniform)
            gaussian[nan_mask] = np.nan

            transformed_cols.append(pl.Series(col, gaussian))

        if not transformed_cols:
            return df

        return df.with_columns(transformed_cols)
