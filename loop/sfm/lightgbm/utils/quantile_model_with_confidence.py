import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

NUM_BOOST_ROUNDS = 1500
EARLY_STOPPING = 100
LOG_EVERY_N = 0

def quantile_model_with_confidence(data, round_params, 
                                 quantiles=[0.1, 0.5, 0.9],
                                 confidence_threshold=8.0):
    '''
    Compute quantile models for prediction intervals with confidence filtering.
    
    Args:
        data (dict): UEL data dict with 'dtrain', 'dval', 'x_test', 'y_test' keys
        round_params (dict): Base parameters for LightGBM
        quantiles (list): Quantiles to train models for
        confidence_threshold (float): Threshold for interval width filtering
    
    Returns:
        dict: UEL-compatible results dict with quantile predictions
    '''
    models = {}
    
    for q in quantiles:
        params = round_params.copy()
        params.update({
            'objective': 'quantile',
            'alpha': q,
            'metric': 'quantile',
            'verbose': -1,
        })
        
        model_q = lgb.train(
            params=params,
            train_set=data['dtrain'],
            num_boost_round=NUM_BOOST_ROUNDS,
            valid_sets=[data['dtrain'], data['dval']],
            valid_names=['train', 'valid'],
            callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False),
                       lgb.log_evaluation(LOG_EVERY_N)])
        
        models[q] = model_q
    
    pred_low = models[quantiles[0]].predict(data['x_test'])
    pred_median = models[quantiles[1]].predict(data['x_test'])
    pred_high = models[quantiles[2]].predict(data['x_test'])
    
    interval_width = pred_high - pred_low
    confident_mask = interval_width < confidence_threshold
    
    final_predictions = pred_median[confident_mask] if confident_mask.sum() > 0 else pred_median
    confident_targets = data['y_test'][confident_mask] if confident_mask.sum() > 0 else data['y_test']
    
    mae = mean_absolute_error(confident_targets, final_predictions)
    rmse = root_mean_squared_error(confident_targets, final_predictions)
    r2 = r2_score(confident_targets, final_predictions)
    
    # Store additional info
    round_results = {
        'models': [models[quantiles[1]]],
        'extras': {
            'rmse': rmse, 
            'mae': mae, 
            'r2': r2,
            'confidence_used': confident_mask.sum(),
            'confidence_total': len(confident_mask),
            'confidence_rate': confident_mask.mean(),
            'avg_interval_width': interval_width.mean(),
            'quantile_models': models
        }
    }
    
    return round_results