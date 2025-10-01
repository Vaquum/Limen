import pandas as pd


def _permutation_prediction_performance(self,
                                       round_id: int) -> pd.DataFrame:
    
    '''
    Create prediction performance table based on round id.
    
    Args:
        round_id (int): Round ID (i.e. nth permutation in an experiment)
        
    Returns:
        pd.DataFrame: Table with columns 'predictions', 'actuals', 'hit', 'miss', 'open', 'close', 'price_change'
    '''

    # Use the centralized _get_round_data method
    round_data = self._get_round_data(round_id)
    
    if self.inverse_scaler is not None:
        perf_df = self.inverse_scaler(round_data['x_test'], self.scalers[round_id]).to_pandas()
    else:
        perf_df = pd.DataFrame()
    
    perf_df['predictions'] = self.preds[round_id]
    perf_df['actuals'] = round_data['y_test']
    perf_df['hit'] = perf_df['predictions'] == perf_df['actuals']
    perf_df['miss'] = perf_df['predictions'] != perf_df['actuals']

    price_df = self._get_test_data_with_all_cols(round_id)
    perf_df['open'] = price_df['open']
    perf_df['close'] = price_df['close']
    perf_df['price_change'] = price_df['close'] - price_df['open']

    return perf_df