import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

NUM_BOOST_ROUNDS = 1500
EARLY_STOPPING = 100
LOG_EVERY_N = 0

def quantile_model_with_confidence(data, round_params, 
                                 quantiles=[0.1, 0.5, 0.9],
                                 confidence_threshold=8.0,
                                 verbose=False):
    '''
    Train quantile models for prediction intervals with confidence filtering.
    
    Parameters:
    -----------
    data : dict
        UEL data dict with 'dtrain', 'dval', 'x_test', 'y_test'
    round_params : dict
        Base parameters for LightGBM
    quantiles : list
        Quantiles to train models for
    confidence_threshold : float
        Threshold for interval width filtering
    verbose : bool
        Whether to print improvement metrics
    
    Returns:
    --------
    dict : UEL-compatible results dict
    '''
    # Train quantile models for prediction intervals
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
    
    # Get prediction intervals on test set
    pred_low = models[quantiles[0]].predict(data['x_test'])    # Lower bound
    pred_median = models[quantiles[1]].predict(data['x_test'])  # Best estimate  
    pred_high = models[quantiles[2]].predict(data['x_test'])   # Upper bound
    
    # Calculate interval width (uncertainty measure)
    interval_width = pred_high - pred_low
    
    # Filter based on confidence (narrow intervals = more confident)
    confident_mask = interval_width < confidence_threshold
    
    # Use median predictions for final output
    final_predictions = pred_median[confident_mask] if confident_mask.sum() > 0 else pred_median
    confident_targets = data['y_test'][confident_mask] if confident_mask.sum() > 0 else data['y_test']
    
    # Calculate metrics on confident predictions
    mae = mean_absolute_error(confident_targets, final_predictions)
    rmse = root_mean_squared_error(confident_targets, final_predictions)
    r2 = r2_score(confident_targets, final_predictions)
    
    # Calculate improvement metrics if verbose
    if verbose:
        mae_all = mean_absolute_error(data['y_test'], pred_median)
        mae_confident = mean_absolute_error(confident_targets, final_predictions)
        improvement_factor = mae_all / mae_confident
        
        if mae_all > mae_confident:
            print(f"Confident predictions: {confident_mask.sum()}/{len(confident_mask)} ({confident_mask.mean():.1%})")
            print(f"Avg interval width: {interval_width.mean():.3f}")
            print(f"MAE improvement: {improvement_factor:.1f}x better on confident predictions")
            print(f"MAE all: {mae_all:.3f} vs MAE confident: {mae_confident:.3f}")
            print(f"Threshold {confidence_threshold} -> {confident_mask.mean():.1%} coverage")
        else:
            print('No improvement, dropping.')
    
    # Store additional info
    round_results = {
        'models': [models[quantiles[1]]],  # Return median model as primary
        'extras': {
            'rmse': rmse, 
            'mae': mae, 
            'r2': r2,
            'confidence_used': confident_mask.sum(),
            'confidence_total': len(confident_mask),
            'confidence_rate': confident_mask.mean(),
            'avg_interval_width': interval_width.mean(),
            'quantile_models': models  # Store all models
        }
    }
    
    return round_results