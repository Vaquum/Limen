def filter_lgb_params(round_params: dict) -> dict:
    
    '''
    Compute filtered parameter dictionary with only valid LightGBM parameters.
    
    Args:
        round_params (dict): Dictionary containing various parameters including LightGBM ones
        
    Returns:
        dict: Dictionary with only valid LightGBM parameters
    '''
    
    valid_lgb_params = [
        'objective', 'metric', 'boosting_type', 'num_leaves',
        'learning_rate', 'feature_fraction', 'bagging_fraction',
        'bagging_freq', 'verbose', 'num_iterations', 'force_col_wise'
    ]
    return {k: v for k, v in round_params.items() if k in valid_lgb_params}