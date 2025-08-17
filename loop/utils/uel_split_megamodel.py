import polars as pl
import numpy as np
import loop
from loop.utils.splits import split_sequential
from typing import List, Dict, Optional, Union, Any


DEFAULT_N_MODELS = 5
DEFAULT_SPLIT_RATIOS = (0.7, 0.15, 0.15)
DEFAULT_N_PERMUTATIONS = 2


def _run_single_uel(data: pl.DataFrame, sfm_module: Any, experiment_name: str, 
                   n_permutations: int, model_idx: int) -> Optional[Dict[str, Any]]:
    '''Compute single UEL experiment and extract best model.'''
    try:
        uel = loop.UniversalExperimentLoop(data, sfm_module)
        uel.run(
            experiment_name=experiment_name,
            n_permutations=n_permutations,
            random_search=False
        )
        
        available_metrics = [col for col in uel.experiment_log.columns 
                           if col in ['mae', 'rmse', 'mse']]
        
        if not available_metrics:
            return None
            
        metric_col = available_metrics[0]
        metric_values = uel.experiment_log[metric_col].to_list()
        
        best_idx = metric_values.index(min(metric_values))
        best_model = uel.models[best_idx]
        
        return {
            'uel_instance': uel,
            'best_idx': best_idx,
            'best_metric': metric_values[best_idx],
            'metric_name': metric_col,
            'model_idx': model_idx,
            'best_model': best_model
        }
        
    except Exception as e:
        return None


def _extract_predictions(models: List[Any], test_features: Any) -> List[Any]:
    '''Compute predictions from all models.'''
    all_predictions = []
    
    for i, model in enumerate(models):
        try:
            if isinstance(model, dict):
                if 'universal' in model:
                    predictions = model['universal'].predict(test_features)
                else:
                    model_key = list(model.keys())[0]
                    predictions = model[model_key].predict(test_features)
            elif isinstance(model, (list, tuple)) and len(model) > 0:
                actual_model = model[0]
                if isinstance(actual_model, dict):
                    if 'universal' in actual_model:
                        predictions = actual_model['universal'].predict(test_features)
                    else:
                        model_key = list(actual_model.keys())[0]
                        predictions = actual_model[model_key].predict(test_features)
                else:
                    predictions = actual_model.predict(test_features)
            else:
                predictions = model.predict(test_features)
                
            all_predictions.append(predictions)
            
        except Exception as e:
            continue
    
    return all_predictions


def _calculate_megamodel_metrics(all_predictions: List[Any], test_targets: Any) -> Dict[str, Any]:
    '''Compute megamodel performance metrics.'''
    if not all_predictions:
        return {}
    
    megamodel_predictions = np.mean(all_predictions, axis=0)
    
    from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
    
    megamodel_mae = mean_absolute_error(test_targets, megamodel_predictions)
    megamodel_r2 = r2_score(test_targets, megamodel_predictions)
    megamodel_rmse = np.sqrt(mean_squared_error(test_targets, megamodel_predictions))
    
    individual_metrics = []
    for i, preds in enumerate(all_predictions):
        mae = mean_absolute_error(test_targets, preds)
        r2 = r2_score(test_targets, preds)
        rmse = np.sqrt(mean_squared_error(test_targets, preds))
        individual_metrics.append({
            'model_idx': i,
            'mae': mae,
            'r2': r2,
            'rmse': rmse
        })
    
    best_individual_mae = min(m['mae'] for m in individual_metrics)
    mae_improvement = (best_individual_mae - megamodel_mae) / best_individual_mae * 100
    
    return {
        'megamodel_predictions': megamodel_predictions,
        'individual_predictions': all_predictions,
        'prediction_std': np.std(all_predictions, axis=0),
        'prediction_count': len(all_predictions),
        'megamodel_mae': megamodel_mae,
        'megamodel_r2': megamodel_r2,
        'megamodel_rmse': megamodel_rmse,
        'individual_metrics': individual_metrics,
        'test_targets': test_targets,
        'mae_improvement_pct': mae_improvement
    }


def uel_split_megamodel(original_data: pl.DataFrame,
                             sfm_module: Any,
                             n_models: int = DEFAULT_N_MODELS,
                             split_ratios: tuple = DEFAULT_SPLIT_RATIOS,
                             n_permutations: int = DEFAULT_N_PERMUTATIONS,
                             experiment_base_name: str = 'split_megamodel',
                             seed: Optional[int] = None) -> Dict[str, Any]:

    '''
    Compute megamodel using multiple UEL runs with different data splits.

    Args:
        original_data (pl.DataFrame): Klines dataset with 'datetime' and numeric columns
        sfm_module: Single File Model module with prep and model functions
        n_models (int): Number of models to create with different splits
        split_ratios (tuple): Train, validation, test split ratios
        n_permutations (int): Number of parameter permutations per UEL run
        experiment_base_name (str): Base name for UEL experiment names
        seed (Optional[int]): Random seed for reproducible splits
        
    Returns:
        Dict[str, Any]: Dictionary with 'megamodel_predictions', 'uel_results', 'megamodel_mae', 'mae_improvement_pct' keys
    '''

    if not isinstance(original_data, pl.DataFrame):
        raise TypeError('original_data must be a Polars DataFrame')
    
    if original_data.is_empty():
        raise ValueError('original_data cannot be empty')
    
    if 'datetime' not in original_data.columns:
        raise ValueError('original_data must contain datetime column')
    
    if n_models <= 0:
        raise ValueError('n_models must be positive')
    
    if n_permutations <= 0:
        raise ValueError('n_permutations must be positive')
    
    if not isinstance(experiment_base_name, str) or not experiment_base_name.strip():
        raise ValueError('experiment_base_name must be a non-empty string')
    
    if len(split_ratios) != 3:
        raise ValueError('split_ratios must contain exactly 3 values for train/val/test')
    
    if abs(sum(split_ratios) - 1.0) > 1e-6:
        raise ValueError('split_ratios must sum to 1.0')

    uel_results = []
    best_models = []
    
    for i in range(n_models):
        current_seed = seed + i if seed is not None else None
        
        if current_seed is not None:
            shuffled_data = original_data.sample(fraction=1.0, seed=current_seed, shuffle=True)
        else:
            shuffled_data = original_data.sample(fraction=1.0, shuffle=True)
        result = _run_single_uel(
            shuffled_data, 
            sfm_module, 
            f'{experiment_base_name}_{i}', 
            n_permutations, 
            i
        )
        
        if result is not None:
            uel_results.append({
                'uel_instance': result['uel_instance'],
                'best_idx': result['best_idx'],
                'best_metric': result['best_metric'],
                'metric_name': result['metric_name'],
                'model_idx': result['model_idx']
            })
            best_models.append(result['best_model'])
    
    if not uel_results:
        raise ValueError('All UEL runs failed - no models trained successfully')
    
    first_uel = uel_results[0]['uel_instance']
    
    all_predictions = []
    if hasattr(first_uel, 'data') and 'x_test' in first_uel.data:
        test_features = first_uel.data['x_test']
        all_predictions = _extract_predictions(best_models, test_features)
    
    results = {
        'uel_results': uel_results,
        'best_models': best_models,
        'n_successful_models': len(best_models)
    }
    
    if all_predictions and hasattr(first_uel, 'data') and 'y_test' in first_uel.data:
        test_targets = first_uel.data['y_test']
        megamodel_metrics = _calculate_megamodel_metrics(all_predictions, test_targets)
        results.update(megamodel_metrics)
    
    return results