from typing import Any
import polars as pl


def permutation_prediction_performance(self: Any,
                                       round_id: int) -> pl.DataFrame:

    '''
    Create prediction performance table based on round id.

    Preparation failures propagate; retained round parameters are never replaced with defaults.

    Args:
        round_id (int): Round ID (i.e. nth permutation in an experiment)

    Returns:
        pl.DataFrame: Table with predictions, actuals, hit/miss, datetime, open, close, and price_change
    '''

    round_data = self.prep(self.data, round_params=self.round_params[round_id])

    if self.inverse_scaler is not None:
        perf_df = self.inverse_scaler(round_data['x_test'], self.scalers[round_id])
    else:
        perf_df = pl.DataFrame()

    perf_df = perf_df.with_columns(
        pl.Series('predictions', self.preds[round_id]),
        pl.Series('actuals', round_data['y_test']),
    ).with_columns(
        (pl.col('predictions') == pl.col('actuals')).alias('hit'),
        (pl.col('predictions') != pl.col('actuals')).alias('miss'),
    )

    price_df = self._get_test_data_with_all_cols(round_id)
    columns = ['datetime', 'open', 'close'] if 'datetime' in price_df.columns else ['open', 'close']
    return perf_df.with_columns(price_df.select(columns)).with_columns(
        (pl.col('close') - pl.col('open')).alias('price_change')
    )
