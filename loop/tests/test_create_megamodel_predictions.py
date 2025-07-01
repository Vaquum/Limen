import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split

from loop.models.lightgbm.utils.create_megamodel_predictions import create_megamodel_predictions

def create_test_data():
    '''Create small synthetic test data'''
    np.random.seed(42)
    x = np.random.randn(200, 5)  # Small dataset
    y = x[:, 0] + x[:, 1] * 0.5 + np.random.normal(0, 0.1, 200)
    return x, y


def create_mock_best_model():
    '''Create a simple mock model for testing'''
    x, y = create_test_data()
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
    
    train_data = lgb.Dataset(x_train, label=y_train)
    val_data = lgb.Dataset(x_val, label=y_val)
    
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'num_leaves': 15,
        'learning_rate': 0.1,
        'verbose': -1
    }
    
    model = lgb.train(params, train_data, num_boost_round=10, valid_sets=[val_data])
    return model


def create_test_data_dict():
    '''Create test data dictionary'''
    x, y = create_test_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    
    return {
        'x_train': x_train,
        'y_train': y_train,
        'x_val': x_val,
        'y_val': y_val,
        'x_test': x_test,
        'y_test': y_test
    }


def test_create_megamodel_predictions():
    '''Test that create_megamodel_predictions runs without errors'''
    print("Testing create_megamodel_predictions...")
    
    data = create_test_data_dict()
    best_model = create_mock_best_model()
    
    # Test basic functionality
    megamodel_preds, models = create_megamodel_predictions(best_model, data, n_models=3)
    
    # Basic checks
    assert isinstance(megamodel_preds, np.ndarray)
    assert isinstance(models, list)
    assert len(models) == 3
    assert len(megamodel_preds) == len(data['y_test'])
    assert not np.isnan(megamodel_preds).any()
    
    print("  ✅ Basic functionality works")
    
    # Test with different n_models
    megamodel_preds, models = create_megamodel_predictions(best_model, data, n_models=1)
    assert len(models) == 1
    
    print("  ✅ Different n_models works")
    
    # Test without validation data
    data_no_val = {k: v for k, v in data.items() if 'val' not in k}
    megamodel_preds, models = create_megamodel_predictions(best_model, data_no_val, n_models=2)
    assert len(models) == 2
    
    print("  ✅ Works without validation data")


if __name__ == "__main__":
    try:
        test_create_megamodel_predictions()
        print("✅ create_megamodel_predictions: ALL TESTS PASSED")
    except Exception as e:
        print(f"❌ create_megamodel_predictions: FAILED - {e}")