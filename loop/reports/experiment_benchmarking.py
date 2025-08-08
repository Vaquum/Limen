import loop

from loop.reports.confusion_matrix_plus import confusion_matrix_plus
from loop.reports.results_df import results_df


def experiment_benchmarking(file_path: str,
                            x: str,
                            model: object,
                            col_sort_order: list,
                            inverse_transform: callable,
                            n_top_results: int = 2) -> None:
    
    '''
    Create benchmark reports from experiment results using top-performing parameters.
    
    NOTE: This currently only works with SFMs that take data from historical.get_spot_klines.
    
    Args:
        file_path (str): Path to experiment log file
        x (str): Column name for distribution comparison analysis
        model (object): Model object with prep and model methods used in experiment
        col_sort_order (list): Column names in order for sorting experiment results
        inverse_transform (callable): Function to reverse data transformations
        n_top_results (int): Number of top results to include in benchmarking
        
    Returns:
        None: Displays benchmark reports for top-performing parameter sets
    '''

    # Read the experiment log file into a dataframe
    data = loop.reports.log_df.read_from_file(file_path)
    
    # Get the top performing ids from the log_df table
    # AND sort it based on the outcome variable of the model (e.g. here auc, precision, and accuracy for regime prediction)
    ids = data.sort_values(col_sort_order, ascending=False).head(n_top_results).index
    
    for i in ids:
    
        # Extract the round params
        params = data.iloc[i].to_dict()
    
        # Get the the data for the model
        historical = loop.HistoricalData()

        if 'n_rows' in params:
            n_rows = params['n_rows']
        else:
            n_rows = None
        
        historical.get_spot_klines(n_rows=n_rows,
                                         kline_size=params['kline_size'])
    
        # Prep the data for the model
        df_prep = model.prep(historical.data, params)
        
        # Run the model
        round_results = model.model(df_prep, params)
    
        # Visualize if required columns are present
        try:
            df_result = results_df(df_prep, round_results['_preds'], inverse_transform)
            confusion_matrix_plus(df_result, x)
            
        except KeyError:
            pass
