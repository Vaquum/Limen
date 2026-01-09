import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
from sklearn.model_selection import train_test_split

from limen.utils.confidence_filtering_system import (
    calibrate_confidence_threshold,
    apply_confidence_filtering,
    confidence_filtering_system
)

def create_test_data():
    '''Create small synthetic test data'''
    np.random.seed(42)
    x = np.random.randn(300, 5)  # Small dataset
    y = x[:, 0] + x[:, 1] * 0.5 + np.random.normal(0, 0.1, 300)
    return x, y


def create_test_models():
    '''Create a few simple models for testing'''
    x, y = create_test_data()
    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.4, random_state=42)
    
    models = []
    for i in range(3):  # Create 3 models with slight variations
        train_data = lgb.Dataset(x_train, label=y_train)
        
        params = {
            'objective': 'regression',
            'metric': 'mae',
            'num_leaves': 10 + i * 5,  # Slight variation
            'learning_rate': 0.1,
            'verbose': -1,
            'random_state': 42 + i
        }
        
        model = lgb.train(params, train_data, num_boost_round=10)
        models.append(model)
    
    return models


def create_test_data_dict():
    '''Create test data dictionary with all splits'''
    x, y = create_test_data()
    
    # Split into train/val/test
    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.4, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)
    
    # Create datetime for test data
    dt_test = pd.date_range('2023-01-01', periods=len(y_test), freq='1h')
    
    return {
        'x_train': x_train,
        'y_train': y_train,
        'x_val': x_val,
        'y_val': y_val,
        'x_test': x_test,
        'y_test': y_test,
        'dt_test': dt_test
    }


def test_calibrate_confidence_threshold():
    '''Test confidence threshold calibration'''
    
    models = create_test_models()
    data = create_test_data_dict()
    
    # Test basic functionality
    threshold, stats = calibrate_confidence_threshold(
        models, data['x_val'], data['y_val'], target_confidence=0.7
    )
    
    # Basic checks
    assert isinstance(threshold, (float, np.floating))
    assert isinstance(stats, dict)
    assert threshold > 0
    assert 'threshold' in stats
    assert 'confident_pct' in stats
    
    # Test different confidence levels
    threshold2, _ = calibrate_confidence_threshold(
        models, data['x_val'], data['y_val'], target_confidence=0.9
    )
    assert isinstance(threshold2, (float, np.floating))


def test_apply_confidence_filtering():
    '''Test confidence filtering application'''
    
    models = create_test_models()
    data = create_test_data_dict()
    
    # First calibrate to get threshold
    threshold, _ = calibrate_confidence_threshold(
        models, data['x_val'], data['y_val'], target_confidence=0.7
    )
    
    # Test filtering
    results = apply_confidence_filtering(
        models, data['x_test'], data['y_test'], threshold
    )
    
    # Basic checks
    assert isinstance(results, dict)
    assert 'predictions' in results
    assert 'uncertainty' in results
    assert 'confident_mask' in results
    assert len(results['predictions']) == len(data['y_test'])
    assert len(results['uncertainty']) == len(data['y_test'])
    assert len(results['confident_mask']) == len(data['y_test'])


def test_confidence_filtering_system():
    '''Test the complete confidence filtering system'''
    
    models = create_test_models()
    data = create_test_data_dict()
    
    # Test complete system
    results, df_results, calibration_stats = confidence_filtering_system(
        models, data, target_confidence=0.8
    )
    
    # Basic checks
    assert isinstance(results, dict)
    assert isinstance(df_results, pl.DataFrame)
    assert isinstance(calibration_stats, dict)

    # Check DataFrame structure
    expected_columns = ['datetime', 'prediction', 'uncertainty', 'is_confident',
                       'confidence_threshold', 'actual_value', 'confidence_score']
    for col in expected_columns:
        assert col in df_results.columns
    
    assert len(df_results) == len(data['y_test'])
    
    # Check results structure
    assert 'predictions' in results
    assert 'confident_mask' in results
    assert 'test_metrics' in results
    
    # Test different target confidence
    results2, df_results2, _ = confidence_filtering_system(
        models, data, target_confidence=0.6
    )
    assert len(df_results2) == len(data['y_test'])
    

def test_edge_cases():
    '''Test edge cases and error handling'''
    
    models = create_test_models()
    data = create_test_data_dict()
    
    # Test with very high confidence (should still work)
    threshold, _ = calibrate_confidence_threshold(
        models, data['x_val'], data['y_val'], target_confidence=0.99
    )
    assert isinstance(threshold, (float, np.floating))
    
    # Test with very low confidence
    threshold, _ = calibrate_confidence_threshold(
        models, data['x_val'], data['y_val'], target_confidence=0.1
    )
    assert isinstance(threshold, (float, np.floating))


if __name__ == "__main__":
    
    test_calibrate_confidence_threshold()

    test_apply_confidence_filtering()

    test_confidence_filtering_system()

    test_edge_cases()