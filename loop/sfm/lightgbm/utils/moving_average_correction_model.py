import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

def moving_average_correction_model(data, round_params,
                                  short_window=5,
                                  medium_window=15, 
                                  long_window=30,
                                  correction_factor=0.4,
                                  trend_threshold=0.15,
                                  residual_window=35,
                                  verbose=False):
    '''
    Train LightGBM model with moving average correction based on residual tracking.
    
    Parameters:
    -----------
    data : dict
        UEL data dict with 'dtrain', 'dval', 'x_test', 'y_test'
    round_params : dict
        Base parameters for LightGBM
    short_window : int
        Window size for short-term moving average
    medium_window : int
        Window size for medium-term moving average
    long_window : int
        Window size for long-term moving average
    correction_factor : float
        Base factor for applying corrections (conservative)
    trend_threshold : float
        Threshold for detecting improving/degrading trends
    residual_window : int
        Maximum number of residuals to keep in sliding window
    verbose : bool
        Whether to print correction results
    
    Returns:
    --------
    dict : UEL-compatible results dict
    '''
    
    # Set up model parameters
    model_params = round_params.copy()
    model_params.update({
        'objective': 'regression',
        'metric': 'mae', 
        'verbose': -1,
    })
    
    # Train the model
    model = lgb.train(
        params=model_params,
        train_set=data['dtrain'],
        num_boost_round=1500,
        valid_sets=[data['dtrain'], data['dval']],
        valid_names=['train', 'valid'],
        callbacks=[lgb.early_stopping(100, verbose=False),
                   lgb.log_evaluation(0)])

    # Moving average correction functions
    def calculate_ma_corrections(recent_residuals):
        if len(recent_residuals) < 3:
            return 0.0, 0.0, 0.0
        
        # Calculate moving averages for different time horizons
        short_ma = np.mean(recent_residuals[-short_window:]) if len(recent_residuals) >= short_window else np.mean(recent_residuals)
        medium_ma = np.mean(recent_residuals[-medium_window:]) if len(recent_residuals) >= medium_window else np.mean(recent_residuals)
        long_ma = np.mean(recent_residuals[-long_window:]) if len(recent_residuals) >= long_window else np.mean(recent_residuals)
        
        return short_ma, medium_ma, long_ma

    def get_weighted_correction(short_ma, medium_ma, long_ma):
        # Weight recent patterns more heavily: 50% short, 30% medium, 20% long
        weights = [0.5, 0.3, 0.2]
        correction = (short_ma * weights[0] + medium_ma * weights[1] + long_ma * weights[2])
        return correction

    def detect_trend(recent_residuals):
        if len(recent_residuals) < 10:
            return "stable", 1.0
        
        recent_10 = recent_residuals[-10:]
        first_half_mean = np.mean(recent_10[:5])
        second_half_mean = np.mean(recent_10[5:])
        
        if first_half_mean < second_half_mean - trend_threshold:
            return "improving", 0.7  # Model getting better, lighter correction
        elif first_half_mean > second_half_mean + trend_threshold:
            return "degrading", 1.3  # Model getting worse, stronger correction
        else:
            return "stable", 1.0

    # Get all predictions
    predictions = model.predict(data['x_test'])
    corrected_predictions = predictions.copy()
    
    # Track residuals and corrections
    recent_residuals = []
    corrections_applied = []
    trend_adjustments = []
    
    for i in range(len(predictions)):
        raw_pred = predictions[i]
        
        # Calculate moving average corrections
        short_ma, medium_ma, long_ma = calculate_ma_corrections(recent_residuals)
        ma_correction = get_weighted_correction(short_ma, medium_ma, long_ma)
        
        # Detect trend and adjust correction strength
        trend, trend_factor = detect_trend(recent_residuals)
        
        # Calculate confidence based on recent stability
        if len(recent_residuals) >= 10:
            recent_std = np.std(recent_residuals[-10:])
            confidence = 1.0 / (1.0 + recent_std)
        else:
            confidence = 0.5
        
        # Apply correction with confidence and trend adjustment
        final_correction = ma_correction * confidence * trend_factor * correction_factor
        corrected_predictions[i] = raw_pred + final_correction
        
        corrections_applied.append(final_correction)
        trend_adjustments.append(trend_factor)
        
        # Update residuals tracking (simulate getting actual result)
        if i >= 1:
            actual = data['y_test'][i-1]
            residual = actual - predictions[i-1]  # Use raw prediction for residual
            recent_residuals.append(residual)
            
            # Keep sliding window of last N residuals
            if len(recent_residuals) > residual_window:
                recent_residuals = recent_residuals[-residual_window:]
    
    # Calculate metrics
    mae_original = mean_absolute_error(data['y_test'], predictions)
    mae_corrected = mean_absolute_error(data['y_test'], corrected_predictions)
    rmse = root_mean_squared_error(data['y_test'], corrected_predictions)
    r2 = r2_score(data['y_test'], corrected_predictions)
    
    # Calculate statistics
    improvement = (mae_original - mae_corrected) / mae_original * 100
    avg_correction = np.mean(np.abs(corrections_applied))
    significant_corrections = len([c for c in corrections_applied if abs(c) > 0.05])
    
    # Trend analysis
    trend_counts = {
        'improving': sum(1 for t in trend_adjustments if t < 1.0),
        'stable': sum(1 for t in trend_adjustments if t == 1.0),
        'degrading': sum(1 for t in trend_adjustments if t > 1.0)
    }
    
    if verbose:
        print(f"Moving Average Correction Results:")
        print(f"  MAE: {mae_original:.3f} → {mae_corrected:.3f} ({improvement:+.1f}%)")
        print(f"  Avg correction magnitude: {avg_correction:.3f}")
        print(f"  Significant corrections: {significant_corrections}/{len(corrections_applied)}")
        print(f"  Trend detection: {trend_counts}")
        print(f"  Final residuals count: {len(recent_residuals)}")
    
    round_results = {
        'models': [model],
        'extras': {
            'rmse': rmse, 
            'mae': mae_corrected, 
            'r2': r2,
            'mae_original': mae_original,
            'improvement_pct': improvement,
            'avg_correction': avg_correction,
            'significant_corrections': significant_corrections,
            'trend_counts': trend_counts,
            'corrections_applied': corrections_applied,
            'predictions_original': predictions,
            'predictions_corrected': corrected_predictions
        }
    }
    
    return round_results