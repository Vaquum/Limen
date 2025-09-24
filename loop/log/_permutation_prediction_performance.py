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

    try:
        if hasattr(self, 'manifest') and self.manifest:
            round_data = self.prep(self.data, self.round_params[round_id], self.manifest)
        else:
            round_data = self.prep(self.data, self.round_params[round_id])
    except TypeError:
        round_data = self.prep(self.data)

    # Optionally produce perf_df from inverse scaler
    if self.inverse_scaler is not None:
        perf_df = self.inverse_scaler(round_data['x_test'], self.scalers[round_id]).to_pandas()
    else:
        perf_df = pd.DataFrame()

    # Scores (windowed length)
    perf_df['predictions'] = self.preds[round_id]
    perf_df['actuals']     = round_data['y_test']
    perf_df['hit']         = perf_df['predictions'] == perf_df['actuals']
    perf_df['miss']        = perf_df['predictions'] != perf_df['actuals']

    # Grab price columns and align to windowed length
    price_df = self._get_test_data_with_all_cols(round_id)
    # Always slice from the end to match windowed outputs
    N = len(perf_df)
    perf_df['open']        = price_df['open'][-N:].reset_index(drop=True)
    perf_df['close']       = price_df['close'][-N:].reset_index(drop=True)
    perf_df['price_change'] = perf_df['close'] - perf_df['open']

    # If you want other columns, do the same: perf_df['col'] = price_df['col'][-N:].reset_index(drop=True)

    return perf_df
