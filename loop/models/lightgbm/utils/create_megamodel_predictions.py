import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import lightgbm as lgb

EARLY_STOPPING = 50

def create_megamodel_predictions(best_model, data, n_models: int = 5):
    '''
    Create megamodel predictions from different train/val splits.
    
    This creates multiple lightgbm sub-models by re-splitting the train+validation data
    with different random seeds, then returns predictions and models.
    
    Parameters:
    -----------
    best_model : lgb.Booster
        The best model to use as template for parameters
    data : dict
        Dictionary containing data splits with keys:
        - 'x_train', 'y_train': Training data
        - 'x_val', 'y_val': Validation data  
        - 'x_test', 'y_test': Test data
    n_models : int, default=5
        Number of models to create in megamodel
    
    Returns:
    --------
    tuple
        - megamodel_preds (np.array): Average predictions across all models
        - models (list): List of trained models
    '''
    
    # Get base parameters from best model
    base_params = best_model.params.copy() if hasattr(best_model, 'params') else {}
    
    # Combine train and validation data
    if 'x_val' in data and 'y_val' in data:
        x_combined = np.vstack([data['x_train'], data['x_val']])
        y_combined = np.hstack([data['y_train'], data['y_val']])
    else:
        x_combined = data['x_train']
        y_combined = data['y_train']
    
    models = []
    test_predictions = []
    
    print(f"🔄 Creating megamodel with {n_models} models...")
    
    # Create megamodel models with different train/val splits
    for i in range(n_models):
        # Create different train/val split
        x_train_megamodel, x_val_megamodel, y_train_megamodel, y_val_megamodel = train_test_split(
            x_combined, y_combined,
            test_size=0.2,
            random_state=42 + i
        )
        
        # Train model with same parameters as best model
        train_dataset = lgb.Dataset(x_train_megamodel, label=y_train_megamodel)
        val_dataset = lgb.Dataset(x_val_megamodel, label=y_val_megamodel, reference=train_dataset)
        
        model = lgb.train(
            params=base_params,
            train_set=train_dataset,
            num_boost_round=best_model.num_trees(),
            valid_sets=[val_dataset],
            callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)]
        )
        
        models.append(model)
        
        # Get predictions on test set
        preds = model.predict(data['x_test'])
        test_predictions.append(preds)
        
        print(f"  Model {i+1}/{n_models} trained")
    
    # Calculate megamodel predictions (average)
    megamodel_preds = np.mean(test_predictions, axis=0)
    
    # Calculate and display metrics
    megamodel_mae = mean_absolute_error(data['y_test'], megamodel_preds)
    megamodel_r2 = r2_score(data['y_test'], megamodel_preds)
    
    # Compare with best model
    best_model_preds = best_model.predict(data['x_test'])
    best_model_mae = mean_absolute_error(data['y_test'], best_model_preds)
    best_model_r2 = r2_score(data['y_test'], best_model_preds)
    
    improvement = (best_model_mae - megamodel_mae) / best_model_mae * 100
    
    print(f"✅ Megamodel created:")
    print(f"  Best model MAE: {best_model_mae:.4f}, R²: {best_model_r2:.4f}")
    print(f"  Megamodel MAE: {megamodel_mae:.4f}, R²: {megamodel_r2:.4f}")
    print(f"  Improvement: {improvement:.2f}%")
    
    return megamodel_preds, models