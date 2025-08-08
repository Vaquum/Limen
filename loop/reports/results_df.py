def results_df(df_prep: dict, preds: list, inverse_transform: callable) -> object:
    
    '''
    Create results dataframe from prepared data and predictions.
    
    Args:
        df_prep (dict): Data dictionary with test data and scaler
        preds (list): Model predictions array
        inverse_transform (callable): Function to reverse data transformations
        
    Returns:
        object: Dataframe with experiment results including predictions, actuals, and derived columns
    '''

    df_results = inverse_transform(df_prep['x_test'], df_prep['_scaler']).to_pandas()
    df_results['predictions'] = preds
    df_results['actuals'] = df_prep['y_test']
    df_results['hit'] = df_results['predictions'] == df_results['actuals']
    df_results['miss'] = df_results['predictions'] != df_results['actuals']
    df_results['price_change'] = df_results['close'] - df_results['open']

    return df_results
