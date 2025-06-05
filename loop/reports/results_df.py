from loop.transforms.logreg_transform import inverse_transform


def results_df(df_prep, preds):

    '''
    Takes in the data dictionary and the predictions and returns a dataframe with the results.

    Args:
        df_prep (dict): data dictionary
        preds (np.array): predictions
        inverse_transform (function): inverse transform function

    Returns:
        df_results (pd.DataFrame): dataframe with the experiment results
    '''

    df_results = inverse_transform(df_prep['x_test'], df_prep['_scaler']).to_pandas()
    df_results['predictions'] = preds
    df_results['actuals'] = df_prep['y_test']
    df_results['hit'] = df_results['predictions'] == df_results['actuals']
    df_results['miss'] = df_results['predictions'] != df_results['actuals']
    df_results['price_change'] = df_results['close'] - df_results['open']

    return df_results
