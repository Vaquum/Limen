import uuid
import os
import numpy as np
import polars as pl
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

import loop


def test_uel_save_load_basic():
    '''Test basic save and load functionality of UEL objects'''
    
    # Create test data
    data = pl.DataFrame({
        'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
        'feature2': [10.0, 20.0, 30.0, 40.0, 50.0],
        'target': [1, 0, 1, 0, 1]
    })
    
    # Create simple SFM
    class TestSFM:
        @staticmethod
        def params():
            return {'param1': [1, 2]}
        
        @staticmethod
        def prep(data, round_params=None):
            return {
                'x_train': data[['feature1', 'feature2']][:3],
                'y_train': data['target'][:3],
                'x_val': data[['feature1', 'feature2']][3:4],
                'y_val': data['target'][3:4],
                'x_test': data[['feature1', 'feature2']][4:],
                'y_test': data['target'][4:],
                '_alignment': {'missing_datetimes': [], 'first_test_datetime': None, 'last_test_datetime': None}
            }
        
        @staticmethod
        def model(data, round_params):
            return {
                'param1_value': round_params['param1'],
                'accuracy': 0.8,
                '_preds': [1]
            }
    
    # Create and set up UEL
    uel_original = loop.UniversalExperimentLoop(TestSFM, data=data)
    
    # Set up basic state
    uel_original.round_params = [{'param1': 1}, {'param1': 2}]
    uel_original.models = []
    uel_original.preds = [[1], [0]]
    uel_original.scalers = []
    uel_original._alignment = [
        {'missing_datetimes': [], 'first_test_datetime': None, 'last_test_datetime': None},
        {'missing_datetimes': [], 'first_test_datetime': None, 'last_test_datetime': None}
    ]
    uel_original.experiment_log = pl.DataFrame({
        'id': [0, 1],
        'param1': [1, 2],
        'accuracy': [0.8, 0.75]
    })
    uel_original.extras = []
    
    # Test save
    test_filename = f"test_{uuid.uuid4().hex[:8]}"
    uel_original.save(test_filename)
    
    expected_file = f"{test_filename}.uel"
    assert os.path.exists(expected_file), f"Save failed - file {expected_file} not created"
    assert os.path.getsize(expected_file) > 0, "Save failed - file is empty"
    
    # Test load
    uel_loaded = loop.UniversalExperimentLoop(TestSFM, filepath=expected_file)
    
    # Verify loaded state
    assert uel_original.data.equals(uel_loaded.data), "Data doesn't match"
    assert uel_original.experiment_log.equals(uel_loaded.experiment_log), "Experiment log doesn't match"
    assert uel_original.round_params == uel_loaded.round_params, "Round params don't match"
    assert uel_original.preds == uel_loaded.preds, "Predictions don't match"
    assert uel_original._alignment == uel_loaded._alignment, "Alignment doesn't match"
    
    # Test that loaded UEL can be saved again
    uel_loaded.save(f"{test_filename}_resave")
    resave_file = f"{test_filename}_resave.uel"
    assert os.path.exists(resave_file), "Resave failed"
    
    # Cleanup
    for file in [expected_file, resave_file]:
        if os.path.exists(file):
            os.remove(file)


def test_uel_save_load_complex_objects():
    '''Test save and load with complex objects like sklearn models and scalers'''
    
    # Create test data
    np.random.seed(42)  # For reproducible results
    data = pl.DataFrame({
        'feature1': np.random.rand(20),
        'feature2': np.random.rand(20),
        'target': np.random.rand(20)
    })
    
    # Create UEL with minimal setup
    class MinimalSFM:
        @staticmethod
        def params():
            return {'test': [1]}
        
        @staticmethod
        def prep(data, round_params=None):
            return {}
        
        @staticmethod
        def model(data, round_params):
            return {'test': 1}
    
    uel_original = loop.UniversalExperimentLoop(MinimalSFM, data=data)
    
    # Add complex objects
    model = LinearRegression()
    model.fit([[1, 2], [3, 4]], [1, 2])
    
    scaler = StandardScaler()
    scaler.fit([[1, 2], [3, 4]])
    
    uel_original.round_params = [{'test': 1}]
    uel_original.models = [model]
    uel_original.scalers = [scaler]
    uel_original.preds = [np.array([1.1, 1.2])]
    uel_original.extras = [{'feature_importance': np.array([0.1, 0.2])}]
    uel_original._alignment = [{'missing_datetimes': [], 'first_test_datetime': None, 'last_test_datetime': None}]
    uel_original.experiment_log = pl.DataFrame({
        'id': [0],
        'test': [1]
    })
    
    # Test save
    test_filename = f"test_complex_{uuid.uuid4().hex[:8]}"
    uel_original.save(test_filename)
    
    expected_file = f"{test_filename}.uel"
    assert os.path.exists(expected_file), "Save failed"
    
    # Test load
    uel_loaded = loop.UniversalExperimentLoop(MinimalSFM, filepath=expected_file)
    
    # Verify complex objects
    assert len(uel_loaded.models) == 1, "Model not loaded"
    assert len(uel_loaded.scalers) == 1, "Scaler not loaded"
    assert len(uel_loaded.preds) == 1, "Predictions not loaded"
    assert len(uel_loaded.extras) == 1, "Extras not loaded"
    
    # Test that models work
    pred_orig = uel_original.models[0].predict([[1, 2]])
    pred_loaded = uel_loaded.models[0].predict([[1, 2]])
    assert np.allclose(pred_orig, pred_loaded), "Model predictions don't match"
    
    # Test that scalers work
    scaled_orig = uel_original.scalers[0].transform([[1, 2]])
    scaled_loaded = uel_loaded.scalers[0].transform([[1, 2]])
    assert np.allclose(scaled_orig, scaled_loaded), "Scaler transforms don't match"
    
    # Test predictions
    assert np.allclose(uel_original.preds[0], uel_loaded.preds[0]), "Predictions don't match"
    
    # Test extras
    orig_feat_imp = uel_original.extras[0]['feature_importance']
    loaded_feat_imp = uel_loaded.extras[0]['feature_importance']
    assert np.allclose(orig_feat_imp, loaded_feat_imp), "Extras don't match"
    
    # Cleanup
    if os.path.exists(expected_file):
        os.remove(expected_file)


def test_uel_load_errors():
    '''Test error handling in load functionality'''
    
    # Create a dummy SFM for tests
    class DummySFM:
        @staticmethod
        def params():
            return {'test': [1]}
        
        @staticmethod
        def prep(data, round_params=None):
            return {}
        
        @staticmethod
        def model(data, round_params):
            return {'test': 1}
    
    # Test loading non-existent file
    try:
        loop.UniversalExperimentLoop(DummySFM(), filepath="nonexistent.uel")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        pass
    
    # Test invalid initialization (no data or filepath)
    try:
        loop.UniversalExperimentLoop(DummySFM())
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

