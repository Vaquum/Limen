import numpy as np
import polars as pl
from sklearn.metrics import mean_absolute_error, r2_score


def calibrate_confidence_threshold(models, x_val, y_val, target_confidence=0.8):
    
    '''
    Compute confidence threshold using validation data.
    
    This function determines the optimal confidence threshold by analyzing model
    prediction variance on validation data. It establishes what level of prediction
    uncertainty corresponds to reliable vs unreliable predictions.
    
    Args:
        models (list): List of trained models
        x_val (array-like): Validation features for threshold calibration
        y_val (array-like): Validation targets for performance evaluation
        target_confidence (float): Target percentage of predictions to classify as "confident" (0.0 to 1.0)
        
    Returns:
        tuple: Confidence threshold and calibration statistics
    '''
    
    # Get model predictions on validation data
    val_preds = []
    for model in models:
        preds = model.predict(x_val)
        val_preds.append(preds)
    
    val_preds = np.array(val_preds)
    val_pred_mean = np.mean(val_preds, axis=0)
    val_pred_std = np.std(val_preds, axis=0)
    
    # Set threshold based on validation data distribution
    confidence_threshold = np.percentile(val_pred_std, target_confidence * 100)
    
    # Analyze calibration quality
    confident_mask = val_pred_std <= confidence_threshold
    uncertain_mask = ~confident_mask
    
    # Calculate performance on each group
    if np.sum(confident_mask) > 0:
        confident_mae = mean_absolute_error(y_val[confident_mask], val_pred_mean[confident_mask])
        confident_r2 = r2_score(y_val[confident_mask], val_pred_mean[confident_mask])
    else:
        confident_mae, confident_r2 = np.nan, np.nan
    
    if np.sum(uncertain_mask) > 0:
        uncertain_mae = mean_absolute_error(y_val[uncertain_mask], val_pred_mean[uncertain_mask])
        uncertain_r2 = r2_score(y_val[uncertain_mask], val_pred_mean[uncertain_mask])
    else:
        uncertain_mae, uncertain_r2 = np.nan, np.nan
    
    overall_mae = mean_absolute_error(y_val, val_pred_mean)
    overall_r2 = r2_score(y_val, val_pred_mean)
    
    calibration_stats = {
        'threshold': confidence_threshold,
        'confident_pct': np.mean(confident_mask) * 100,
        'val_overall_mae': overall_mae,
        'val_overall_r2': overall_r2,
        'val_confident_mae': confident_mae,
        'val_confident_r2': confident_r2,
        'val_uncertain_mae': uncertain_mae,
        'val_uncertain_r2': uncertain_r2,
        'val_std_stats': {
            'min': np.min(val_pred_std),
            'median': np.median(val_pred_std),
            'max': np.max(val_pred_std),
            'threshold': confidence_threshold
        }
    }
    
    return confidence_threshold, calibration_stats


def apply_confidence_filtering(models, x_test, y_test, confidence_threshold):
    
    '''
    Apply confidence filtering using pre-calibrated threshold.
    
    Args:
        models (list): List of trained models
        x_test (array-like): Test features for prediction and confidence assessment
        y_test (array-like): Test targets for performance evaluation
        confidence_threshold (float): Pre-calibrated confidence threshold from validation data
        
    Returns:
        dict: Results dictionary containing predictions, uncertainty, masks, and metrics
    '''
    
    # Get model predictions on test data
    test_preds = []
    for model in models:
        preds = model.predict(x_test)
        test_preds.append(preds)
    
    test_preds = np.array(test_preds)
    test_pred_mean = np.mean(test_preds, axis=0)
    test_pred_std = np.std(test_preds, axis=0)
    
    # Apply the pre-calibrated threshold
    confident_mask = test_pred_std <= confidence_threshold
    uncertain_mask = ~confident_mask
    
    # Calculate test performance
    overall_mae = mean_absolute_error(y_test, test_pred_mean)
    overall_r2 = r2_score(y_test, test_pred_mean)
    
    if np.sum(confident_mask) > 0:
        confident_mae = mean_absolute_error(y_test[confident_mask], test_pred_mean[confident_mask])
        confident_r2 = r2_score(y_test[confident_mask], test_pred_mean[confident_mask])
    else:
        confident_mae, confident_r2 = np.nan, np.nan
    
    if np.sum(uncertain_mask) > 0:
        uncertain_mae = mean_absolute_error(y_test[uncertain_mask], test_pred_mean[uncertain_mask])
        uncertain_r2 = r2_score(y_test[uncertain_mask], test_pred_mean[uncertain_mask])
    else:
        uncertain_mae, uncertain_r2 = np.nan, np.nan
        
    results = {
        'predictions': test_pred_mean,
        'uncertainty': test_pred_std,
        'confident_mask': confident_mask,
        'threshold_used': confidence_threshold,
        'coverage': np.mean(confident_mask),
        'test_metrics': {
            'overall_mae': overall_mae,
            'overall_r2': overall_r2,
            'confident_mae': confident_mae,
            'confident_r2': confident_r2,
            'uncertain_mae': uncertain_mae,
            'uncertain_r2': uncertain_r2
        },
        'individual_predictions': test_preds
    }
    
    return results


def confidence_filtering_system(models: list, data: dict, target_confidence: float = 0.8) -> tuple:
    
    '''
    Compute complete confidence filtering system with validation-based calibration.
    
    Args:
        models (list): List of trained models for confidence estimation
        data (dict): Dictionary with validation and test data splits containing 'x_val', 'y_val', 'x_test', 'y_test', 'dt_test' keys
        target_confidence (float): Target percentage of predictions to classify as confident
        
    Returns:
        tuple: Confidence threshold, filtered results, and calibration statistics
    '''
    
    # Step 1: Calibrate on validation data
    confidence_threshold, calibration_stats = calibrate_confidence_threshold(
        models, data['x_val'], data['y_val'], target_confidence
    )
    
    # Step 2: Apply to test data using calibrated threshold
    results = apply_confidence_filtering(
        models, data['x_test'], data['y_test'], confidence_threshold
    )
    
    # Step 3: Create detailed results DataFrame
    df_results = pl.DataFrame({
        'datetime': data['dt_test'],
        'prediction': results['predictions'],
        'uncertainty': results['uncertainty'],
        'is_confident': results['confident_mask'],
        'confidence_threshold': confidence_threshold,
        'actual_value': data['y_test'],
    })

    # Add confidence scores (higher = more confident)
    df_results = df_results.with_columns([
        (1 / (1 + pl.col('uncertainty'))).alias('confidence_score')
    ])
    
    # Sort by confidence score (most confident first)
    df_results = df_results.sort('confidence_score', descending=True)
    
    return results, df_results, calibration_stats
