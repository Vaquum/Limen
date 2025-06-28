# Enhanced UEL-Compatible Megamodel with Mega Model Integration
import loop
import numpy as np
import polars as pl
from typing import Dict, List, Tuple, Any
import pandas as pd
from loop.models import lightgbm
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error


class MegaModelDataSampler:
    """
    Creates different data sampling strategies for UEL experiments with integrated mega model capabilities.
    
    This class implements multiple data sampling approaches AND mega model methods to systematically 
    test which combination of data usage and modeling approach produces the best performance.
    
    Parameters:
    -----------
    df_orig : pl.DataFrame
        The original complete dataset to create samples from
    
    Attributes:
    -----------
    df_orig : pl.DataFrame
        Reference to the original dataset
    total_rows : int
        Total number of rows in the original dataset
    """
    
    def __init__(self, df_orig: pl.DataFrame):
        """
        Initialize the enhanced data sampler with the original dataset.
        
        Parameters:
        -----------
        df_orig : pl.DataFrame
            The complete dataset to create sampling variations from
        """
        self.df_orig = df_orig
        self.total_rows = len(df_orig)
        
    def temporal_windows(self, window_size: int = 15000, overlap: float = 0.2) -> List[pl.DataFrame]:
        """
        Create overlapping temporal windows from the dataset.
        
        This method assumes data is chronologically ordered and creates overlapping
        time-based windows. Useful for time series data where temporal patterns matter.
        
        Parameters:
        -----------
        window_size : int, default=15000
            Size of each temporal window in rows
        overlap : float, default=0.2
            Fraction of overlap between consecutive windows (0.0 to 1.0)
        
        Returns:
        --------
        List[pl.DataFrame]
            List of datasets, each representing a temporal window
        """
        # Skip if dataset is too small for even one window
        if self.total_rows < window_size:
            return []
        
        datasets = []
        step_size = int(window_size * (1 - overlap))
        
        start_positions = []
        i = 0
        while True:
            start_idx = i * step_size
            end_idx = start_idx + window_size
            
            if end_idx > self.total_rows:
                # Stop if we can't fit another full window
                break
            else:
                start_positions.append(start_idx)
                i += 1
                
            # Stop if we have enough windows or reached reasonable coverage
            if len(start_positions) >= 10:
                break
        
        for start_idx in start_positions:
            end_idx = start_idx + window_size
            datasets.append(self.df_orig[start_idx:end_idx])
            
        return datasets
    
    def bootstrap_samples(self, sample_size: int = 15000, n_samples: int = 5) -> List[pl.DataFrame]:
        """
        Create bootstrap samples with replacement from the original dataset.
        
        Bootstrap sampling creates new datasets by randomly sampling with replacement,
        allowing the same row to appear multiple times. This helps estimate model
        stability and performance variance.
        
        Parameters:
        -----------
        sample_size : int, default=15000
            Number of rows in each bootstrap sample
        n_samples : int, default=5
            Number of bootstrap samples to create
        
        Returns:
        --------
        List[pl.DataFrame]
            List of bootstrap sampled datasets
        """
        datasets = []
        
        for i in range(n_samples):
            rng = np.random.default_rng(seed=42 + i)  # Fixed seed for reproducibility
            indices = rng.choice(self.total_rows, size=sample_size, replace=True)
            datasets.append(self.df_orig.take(sorted(indices)))
            
        return datasets
    
    def stratified_samples(self, target_col: str = 'breakout_long', 
                          sample_size: int = 15000, n_samples: int = 5) -> List[pl.DataFrame]:
        """
        Create stratified samples based on target value distributions (without replacement).
        
        This method ensures each sample maintains the same target distribution as the
        original data by sampling proportionally from different target value ranges.
        Uses no replacement to preserve natural frequency of values.
        
        Parameters:
        -----------
        target_col : str, default='breakout_long'
            Name of the target column to stratify on
        sample_size : int, default=15000
            Total size of each stratified sample
        n_samples : int, default=5
            Number of stratified samples to create
        
        Returns:
        --------
        List[pl.DataFrame]
            List of stratified sampled datasets maintaining target distribution
        """
        return self._create_stratified_samples(target_col, sample_size, n_samples, replace=False)
    
    def stratified_samples_with_replacement(self, target_col: str = 'breakout_long', 
                                          sample_size: int = 15000, n_samples: int = 5) -> List[pl.DataFrame]:
        """
        Create stratified samples based on target value distributions (with replacement).
        
        This method ensures each sample maintains the same target distribution as the
        original data by sampling proportionally from different target value ranges.
        Uses replacement to ensure exact sample sizes even with small strata.
        
        Parameters:
        -----------
        target_col : str, default='breakout_long'
            Name of the target column to stratify on
        sample_size : int, default=15000
            Total size of each stratified sample
        n_samples : int, default=5
            Number of stratified samples to create
        
        Returns:
        --------
        List[pl.DataFrame]
            List of stratified sampled datasets maintaining target distribution
        """
        return self._create_stratified_samples(target_col, sample_size, n_samples, replace=True)
    
    def _create_stratified_samples(self, target_col: str, sample_size: int, n_samples: int, replace: bool) -> List[pl.DataFrame]:
        """
        Internal method to create stratified samples with or without replacement.
        """
        datasets = []
        
        # Create strata based on target values
        target_values = self.df_orig[target_col].to_numpy()
        non_zero_values = target_values[target_values > 0]
        
        if len(non_zero_values) > 0:
            percentiles = np.percentile(non_zero_values, [33, 66])
        else:
            percentiles = [0, 0]
        
        # Define strata
        low_mask = self.df_orig[target_col] <= percentiles[0]
        med_mask = (self.df_orig[target_col] > percentiles[0]) & (self.df_orig[target_col] <= percentiles[1])
        high_mask = self.df_orig[target_col] > percentiles[1]
        
        strata = [
            self.df_orig.filter(low_mask),
            self.df_orig.filter(med_mask), 
            self.df_orig.filter(high_mask)
        ]
        
        for i in range(n_samples):
            rng = np.random.default_rng(seed=42 + i)
            combined_sample = []
            
            # Sample from each stratum proportionally
            for j, stratum in enumerate(strata):
                if len(stratum) > 0:
                    if replace:
                        # With replacement: can oversample small strata
                        total = sum(len(s) for s in strata)
                        stratum_size = round(sample_size * len(stratum) / total)
                        indices = rng.choice(len(stratum), size=stratum_size, replace=True)
                    else:
                        total = sum(len(s) for s in strata)
                        stratum_size = min(len(stratum), round(sample_size * len(stratum) / total))
                        if stratum_size > 0:
                            indices = rng.choice(len(stratum), size=stratum_size, replace=False)
                        else:
                            continue
                    
                    combined_sample.append(stratum[sorted(indices)])
            
            if combined_sample:
                datasets.append(pl.concat(combined_sample).sort('datetime'))
            
        return datasets
    
    def random_subsets(self, sample_size: int = 15000, n_samples: int = 5) -> List[pl.DataFrame]:
        """
        Create random contiguous subsets from the dataset.
        
        Selects random contiguous chunks of data, avoiding the very beginning and end
        of the dataset to prevent edge effects. This is useful when you want to test
        model performance on different periods of your data.
        
        Parameters:
        -----------
        sample_size : int, default=15000
            Size of each contiguous subset
        n_samples : int, default=5
            Number of random subsets to create
        
        Returns:
        --------
        List[pl.DataFrame]
            List of random contiguous subsets from the original data
        """
        datasets = []
        
        for i in range(n_samples):
            rng = np.random.default_rng(seed=42 + i)
            n = len(self.df_orig)
            lo = int(n * 0.05)
            hi = int(n * 0.95) - sample_size
            
            if hi >= lo:
                start = int(rng.integers(lo, hi + 1))
                datasets.append(self.df_orig[start:start + sample_size])
            else:
                # Fallback to full dataset if sample_size too large
                datasets.append(self.df_orig)
                
        return datasets
    
    def kfold_datasets(self, k: int = 5) -> List[pl.DataFrame]:
        """
        Create K-fold cross-validation datasets by excluding one fold at a time.
        
        Each dataset contains k-1 folds of the original data, systematically excluding
        different portions. This tests model performance across different data splits
        while maximizing data usage.
        
        Parameters:
        -----------
        k : int, default=5
            Number of folds to create
        
        Returns:
        --------
        List[pl.DataFrame]
            List of datasets, each containing k-1 folds of the original data
        """
        datasets = []
        fold_size = self.total_rows // k
        
        for i in range(k):
            # Exclude fold i, use remaining k-1 folds
            start_exclude = i * fold_size
            end_exclude = (i + 1) * fold_size if i < k-1 else self.total_rows
            
            # Concatenate before and after excluded fold
            before = self.df_orig[:start_exclude] if start_exclude > 0 else None
            after = self.df_orig[end_exclude:] if end_exclude < self.total_rows else None
            
            if before is not None and after is not None:
                fold_data = pl.concat([before, after])
            elif before is not None:
                fold_data = before
            elif after is not None:
                fold_data = after
            else:
                continue
                
            datasets.append(fold_data)
            
        return datasets
    
    def full_dataset(self) -> List[pl.DataFrame]:
        """
        Return the full original dataset as a single-item list.
        
        This serves as the baseline comparison - using all available data without
        any sampling strategy. Provides the control group for comparing other methods.
        
        Returns:
        --------
        List[pl.DataFrame]
            Single-item list containing the complete original dataset
        """
        return [self.df_orig]

    def create_mega_model_from_models(self, models: List[Any], test_data: Dict[str, np.ndarray], 
                                    mega_model_size: int = 3, selection_metric: str = 'mae') -> Dict[str, Any]:
        """
        Create a mega model from top-performing models within a strategy.
        
        This method takes the best models from a sampling strategy and creates a mega model
        by averaging their predictions, integrating the original mega model approach into 
        the strategy comparison framework.
        
        Parameters:
        -----------
        models : List[Any]
            List of trained models with their performance metrics
        test_data : Dict[str, np.ndarray]
            Test data dictionary with 'x_test' and 'y_test' keys
        mega_model_size : int, default=3
            Number of top models to include in mega model
        selection_metric : str, default='mae'
            Metric to use for selecting top models ('mae', 'rmse', or 'r2')
        
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing mega model predictions and performance metrics
        """
        if len(models) < 2:
            # Not enough models for mega model, return single model results
            if models:
                single_model = models[0]['model']
                single_preds = single_model.predict(test_data['x_test'])
                return {
                    'mega_model_preds': single_preds,
                    'mega_model_mae': mean_absolute_error(test_data['y_test'], single_preds),
                    'mega_model_rmse': root_mean_squared_error(test_data['y_test'], single_preds),
                    'mega_model_r2': r2_score(test_data['y_test'], single_preds),
                    'mega_model_size': 1,
                    'models_used': [models[0]]
                }
            else:
                return None
        
        # Sort models by the selection metric
        reverse_sort = selection_metric == 'r2'  # Higher is better for R²
        sorted_models = sorted(models, 
                             key=lambda x: x[selection_metric], 
                             reverse=reverse_sort)
        
        # Select top models for mega model
        top_models = sorted_models[:min(mega_model_size, len(sorted_models))]
        
        # Generate predictions from each model
        mega_model_preds_list = []
        for model_info in top_models:
            model = model_info['model']
            preds = model.predict(test_data['x_test'])
            mega_model_preds_list.append(preds)
        
        # Average predictions
        mega_model_preds = np.mean(mega_model_preds_list, axis=0)
        
        # Calculate mega model metrics
        mega_model_mae = mean_absolute_error(test_data['y_test'], mega_model_preds)
        mega_model_rmse = np.sqrt(mean_squared_error(test_data['y_test'], mega_model_preds))
        mega_model_r2 = r2_score(test_data['y_test'], mega_model_preds)
        
        return {
            'mega_model_preds': mega_model_preds,
            'mega_model_mae': mega_model_mae,
            'mega_model_rmse': mega_model_rmse,
            'mega_model_r2': mega_model_r2,
            'mega_model_size': len(top_models),
            'models_used': top_models,
            'individual_predictions': mega_model_preds_list
        }

    def create_data_split_mega_model(self, uel_instance, prep_func, n_models: int = 3) -> Dict[str, Any]:
        """
        Create mega model from different train/val splits using UEL's data and best model.
        
        This implements the original mega model approach: take the UEL's original data,
        re-run prep with different random seeds to get different splits, train models,
        then average predictions.
        
        Parameters:
        -----------
        uel_instance : UniversalExperimentLoop
            UEL instance with original data and trained models
        prep_func : callable
            The preprocessing function used in the original UEL run
        n_models : int, default=3
            Number of models to create in mega model
        
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing mega model results and individual models
        """
            
        # Find best model by MAE
        mae_values = [d['mae'] for d in uel_instance.extras]
        best_idx = mae_values.index(min(mae_values))
        best_model = uel_instance.models[best_idx][0]
        base_params = best_model.params.copy() if hasattr(best_model, 'params') else {}
        
        # Get original dataset and run prep function to get ONE set of processed data for testing
        original_data = uel_instance.data        
        base_processed = prep_func(original_data)
                
        # Use base_processed test data for final evaluation
        x_test = base_processed['x_test']
        y_test = base_processed['y_test']
        
        models = []
        
        # Create mega model variations by modifying the prep function's random behavior
        # Since we can't easily modify random seeds in the prep function, we'll use the approach
        # of re-splitting the combined train+val data
        for i in range(n_models):            
            # Combine train and validation data from base processing
            if 'x_val' in base_processed and 'y_val' in base_processed:
                X_combined = np.vstack([base_processed['x_train'], base_processed['x_val']])
                y_combined = np.hstack([base_processed['y_train'], base_processed['y_val']])
            else:
                X_combined = base_processed['x_train']
                y_combined = base_processed['y_train']
            
            # Create different train/val split
            X_train_mega, X_val_mega, y_train_mega, y_val_mega = train_test_split(
                X_combined, y_combined,
                test_size=0.2,
                random_state=42 + i
            )
            
            # Train model with same parameters as best model
            train_data = lgb.Dataset(X_train_mega, label=y_train_mega)
            val_data = lgb.Dataset(X_val_mega, label=y_val_mega, reference=train_data)
            
            model = lgb.train(
                params=base_params,
                train_set=train_data,
                num_boost_round=best_model.num_trees(),
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )
            models.append(model)
        
        # Generate mega model predictions
        test_preds_list = []
        for model in models:
            preds = model.predict(x_test)
            test_preds_list.append(preds)
        
        # Average predictions
        mega_model_preds = np.mean(test_preds_list, axis=0)
        base_preds = best_model.predict(x_test)
        
        # Calculate metrics
        mega_model_mae = mean_absolute_error(y_test, mega_model_preds)
        mega_model_rmse = np.sqrt(mean_squared_error(y_test, mega_model_preds))
        mega_model_r2 = r2_score(y_test, mega_model_preds)
        
        base_mae = mean_absolute_error(y_test, base_preds)
        base_rmse = np.sqrt(mean_squared_error(y_test, base_preds))
        base_r2 = r2_score(y_test, base_preds)
        
        improvement = (base_mae - mega_model_mae) / base_mae * 100
        
        return {
            'mega_model_preds': mega_model_preds,
            'mega_model_mae': mega_model_mae,
            'mega_model_rmse': mega_model_rmse,
            'mega_model_r2': mega_model_r2,
            'mega_model_size': len(models),
            'models': models,
            'individual_predictions': test_preds_list,
            'base_model_mae': base_mae,
            'base_model_rmse': base_rmse,
            'base_model_r2': base_r2,
            'improvement_mae': improvement,
            'improvement_rmse': (base_rmse - mega_model_rmse) / base_rmse * 100,
            'improvement_r2': (mega_model_r2 - base_r2) / abs(base_r2) * 100,
            'test_data': {'x_test': x_test, 'y_test': y_test}
        }


def run_enhanced_megamodel_with_uel(df_orig: pl.DataFrame, prep_func, model_func, 
                                  target: str = 'breakout_long', 
                                  experiment_base_name: str = "enhanced_megamodel",
                                  enable_mega_models: bool = True,
                                  mega_model_size: int = 3) -> Dict[str, Any]:
    """
    Execute comprehensive enhanced megamodel experiment with mega model integration.
    
    This function systematically tests different data sampling approaches AND mega model
    methods to identify the optimal combination of data usage and modeling approach.
    It runs both single-model and mega model approaches for each strategy using UEL.
    
    Parameters:
    -----------
    df_orig : pl.DataFrame
        Original complete dataset to experiment with
    prep_func : callable
        Data preprocessing function to apply before model training (UEL compatible)
    model_func : callable
        Model training function (should work with UEL)
    target : str, default='breakout_long'
        Name of the target column for regression
    experiment_base_name : str, default="enhanced_megamodel"
        Base name for UEL experiments
    enable_mega_models : bool, default=True
        Whether to create and test mega models
    mega_model_size : int, default=3
        Number of models to include in mega models
    
    Returns:
    --------
    Dict[str, Any]
        Comprehensive results including single models, mega models, and comparisons
    """    
    # Initialize sampler
    sampler = MegaModelDataSampler(df_orig)
    
    # Define strategies and their datasets
    strategies = {
        'full_dataset': sampler.full_dataset(),
        'random_subsets': sampler.random_subsets(sample_size=15000, n_samples=5),
        'bootstrap_samples': sampler.bootstrap_samples(sample_size=15000, n_samples=5),
        'temporal_windows': sampler.temporal_windows(window_size=15000, overlap=0.2),
        'stratified_samples': sampler.stratified_samples(target, sample_size=15000, n_samples=5),
        'stratified_samples_with_replacement': sampler.stratified_samples_with_replacement(target, sample_size=15000, n_samples=5),
        'kfold_datasets': sampler.kfold_datasets(k=5)
    }
    
    # Store comprehensive results
    all_results = {}
    
    print("="*100)
    print("RUNNING ENHANCED MEGAMODEL EXPERIMENT WITH UEL INTEGRATION")
    print("="*100)
    
    for strategy_name, datasets in strategies.items():
        print(f"\n--- Running strategy: {strategy_name} ---")
        print(f"Number of datasets: {len(datasets)}")
        
        strategy_results = {
            'single_models': [],
            'cross_dataset_mega_models': [],
            'data_split_mega_models': []
        }
        
        # Collect all models from this strategy for cross-dataset mega model
        all_strategy_models = []
        
        for i, dataset in enumerate(datasets):
            print(f"  Dataset {i+1}/{len(datasets)} - Size: {len(dataset)} rows")
            
            try:
                # 1. Run standard UEL experiment (single models) - fully UEL compatible
                uel = loop.UniversalExperimentLoop(dataset, lightgbm)
                experiment_name = f"{experiment_base_name}_{strategy_name}_dataset_{i+1}"
                
                uel.run(
                    experiment_name=experiment_name,
                    n_permutations=24,
                    random_search=True,
                    prep=prep_func,
                    model=model_func
                )
                
                # Extract single model results
                r2_values = [d['r2'] for d in uel.extras]
                mae_values = [d['mae'] for d in uel.extras]
                rmse_values = [d['rmse'] for d in uel.extras]
                
                best_idx = mae_values.index(min(mae_values))
                
                single_model_result = {
                    'dataset_idx': i,
                    'best_mae': mae_values[best_idx],
                    'best_rmse': rmse_values[best_idx],
                    'best_r2': r2_values[best_idx],
                    'mae_mean': np.mean(mae_values),
                    'mae_std': np.std(mae_values),
                    'rmse_mean': np.mean(rmse_values),
                    'rmse_std': np.std(rmse_values),
                    'r2_mean': np.mean(r2_values),
                    'r2_std': np.std(r2_values),
                    'uel_instance': uel,
                    'best_model': uel.models[best_idx][0],
                    'dataset_size': len(dataset)
                }
                
                strategy_results['single_models'].append(single_model_result)
                
                # Collect models for mega model creation
                for j, (model, extra) in enumerate(zip(uel.models, uel.extras)):
                    all_strategy_models.append({
                        'model': model[0],
                        'mae': extra['mae'],
                        'rmse': extra['rmse'],
                        'r2': extra['r2'],
                        'dataset_idx': i,
                        'model_idx': j
                    })
                
                # 2. Create data-split mega model using UEL data directly
                if enable_mega_models:
                    print(f"    Creating data-split mega model...")
                    data_split_mega_model = sampler.create_data_split_mega_model(
                        uel_instance=uel,
                        prep_func=prep_func,
                        n_models=mega_model_size
                    )
                    
                    if data_split_mega_model:
                        data_split_mega_model['dataset_idx'] = i
                        strategy_results['data_split_mega_models'].append(data_split_mega_model)
                        print(f"    ✅ Data-split mega model created successfully")
                    else:
                        print(f"    ❌ Failed to create data-split mega model")
                
            except Exception as e:
                print(f"    Error in dataset {i+1}: {e}")
                continue
        
        # 3. Create cross-dataset mega model from best models across all datasets
        if enable_mega_models and all_strategy_models and strategy_results['single_models']:
            print(f"  Creating cross-dataset mega model from {len(all_strategy_models)} models...")
            
            # Use test data from the first successful dataset
            test_data = None
            for single_result in strategy_results['single_models']:
                if 'uel_instance' in single_result:
                    uel_instance = single_result['uel_instance']
                    if hasattr(uel_instance, 'data') and 'x_test' in uel_instance.data:
                        test_data = {
                            'x_test': uel_instance.data['x_test'],
                            'y_test': uel_instance.data['y_test']
                        }
                        break
            
            if test_data is not None:
                cross_dataset_mega_model = sampler.create_mega_model_from_models(
                    models=all_strategy_models,
                    test_data=test_data,
                    mega_model_size=mega_model_size,
                    selection_metric='mae'
                )
                
                if cross_dataset_mega_model:
                    strategy_results['cross_dataset_mega_models'].append(cross_dataset_mega_model)
                    print(f"  ✅ Cross-dataset mega model created successfully")
                else:
                    print(f"  ❌ Failed to create cross-dataset mega model")
            else:
                print(f"  ❌ No test data available for cross-dataset mega model")
        elif enable_mega_models:
            print(f"  ⚠️ Skipping cross-dataset mega model - insufficient models ({len(all_strategy_models)} models, {len(strategy_results['single_models'])} single results)")
        
        all_results[strategy_name] = strategy_results
    
    # Create comprehensive comparison
    comparison_data = []
    
    for strategy_name, results in all_results.items():
        if results['single_models']:  # Only process strategies with successful results
            
            # Single model stats
            single_mae_values = [r['best_mae'] for r in results['single_models']]
            single_rmse_values = [r['best_rmse'] for r in results['single_models']]
            single_r2_values = [r['best_r2'] for r in results['single_models']]
            
            comparison_row = {
                'strategy': strategy_name,
                'datasets_count': len(results['single_models']),
                'avg_dataset_size': np.mean([r['dataset_size'] for r in results['single_models']]),
                
                # Single model performance
                'single_mae_mean': np.mean(single_mae_values),
                'single_mae_std': np.std(single_mae_values),
                'single_rmse_mean': np.mean(single_rmse_values),
                'single_rmse_std': np.std(single_rmse_values),
                'single_r2_mean': np.mean(single_r2_values),
                'single_r2_std': np.std(single_r2_values),
            }
            
            # Data-split mega model stats
            if results['data_split_mega_models']:
                ds_mega_model_mae = [r['mega_model_mae'] for r in results['data_split_mega_models']]
                ds_mega_model_rmse = [r['mega_model_rmse'] for r in results['data_split_mega_models']]
                ds_mega_model_r2 = [r['mega_model_r2'] for r in results['data_split_mega_models']]
                ds_improvements = [r['improvement_mae'] for r in results['data_split_mega_models']]
                
                comparison_row.update({
                    'ds_mega_model_mae_mean': np.mean(ds_mega_model_mae),
                    'ds_mega_model_mae_std': np.std(ds_mega_model_mae),
                    'ds_mega_model_rmse_mean': np.mean(ds_mega_model_rmse),
                    'ds_mega_model_rmse_std': np.std(ds_mega_model_rmse),
                    'ds_mega_model_r2_mean': np.mean(ds_mega_model_r2),
                    'ds_mega_model_r2_std': np.std(ds_mega_model_r2),
                    'ds_mega_model_improvement_mae': np.mean(ds_improvements)
                })
            
            # Cross-dataset mega model stats
            if results['cross_dataset_mega_models']:
                cd_mega_model = results['cross_dataset_mega_models'][0]  # Should only be one
                comparison_row.update({
                    'cd_mega_model_mae': cd_mega_model['mega_model_mae'],
                    'cd_mega_model_rmse': cd_mega_model['mega_model_rmse'],
                    'cd_mega_model_r2': cd_mega_model['mega_model_r2'],
                    'cd_mega_model_size': cd_mega_model['mega_model_size'],
                    'cd_mega_model_improvement_mae': (np.mean(single_mae_values) - cd_mega_model['mega_model_mae']) / np.mean(single_mae_values) * 100
                })
            
            comparison_data.append(comparison_row)
    
    # Create comparison DataFrame
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        
        # Determine best performing approach for each strategy - only use columns that exist
        available_mae_cols = []
        if 'single_mae_mean' in comparison_df.columns:
            available_mae_cols.append('single_mae_mean')
        if 'ds_mega_model_mae_mean' in comparison_df.columns:
            available_mae_cols.append('ds_mega_model_mae_mean')
        if 'cd_mega_model_mae' in comparison_df.columns:
            available_mae_cols.append('cd_mega_model_mae')
        
        if available_mae_cols:
            comparison_df['best_mae'] = comparison_df[available_mae_cols].min(axis=1, skipna=True)
            comparison_df['best_approach'] = comparison_df[available_mae_cols].idxmin(axis=1, skipna=True)
        else:
            comparison_df['best_mae'] = comparison_df['single_mae_mean']
            comparison_df['best_approach'] = 'single_mae_mean'
        
        # Sort by best performance
        comparison_df = comparison_df.sort_values('best_mae')
        
        print("\n" + "="*120)
        print("ENHANCED MEGAMODEL COMPARISON SUMMARY")
        print("="*120)
        print(comparison_df.round(4))
        
        # Find overall best
        best_strategy_row = comparison_df.iloc[0]
        best_strategy = best_strategy_row['strategy']
        best_approach = best_strategy_row['best_approach']
        best_mae = best_strategy_row['best_mae']
        
        print(f"\n🏆 BEST OVERALL COMBINATION:")
        print(f"   Strategy: {best_strategy}")
        print(f"   Approach: {best_approach}")
        print(f"   Best MAE: {best_mae:.4f}")
        
        # Show improvement from mega models
        if enable_mega_models:
            print(f"\n📈 MEGA MODEL IMPROVEMENTS:")
            for _, row in comparison_df.iterrows():
                strategy = row['strategy']
                if not pd.isna(row.get('ds_mega_model_improvement_mae', np.nan)):
                    print(f"   {strategy} - Data Split Mega Model: {row['ds_mega_model_improvement_mae']:+.2f}% MAE improvement")
                if not pd.isna(row.get('cd_mega_model_improvement_mae', np.nan)):
                    print(f"   {strategy} - Cross Dataset Mega Model: {row['cd_mega_model_improvement_mae']:+.2f}% MAE improvement")
        
        # Save results
        comparison_df.to_csv('enhanced_megamodel_comparison.csv', index=False)
        print(f"\n📊 Results saved to 'enhanced_megamodel_comparison.csv'")
        
    else:
        print("❌ No successful results to compare")
        comparison_df = pd.DataFrame()
        best_strategy = None
        best_approach = None
    
    return {
        'detailed_results': all_results,
        'comparison_summary': comparison_df,
        'best_strategy': best_strategy,
        'best_approach': best_approach,
        'mega_models_enabled': enable_mega_models
    }


def integrate_enhanced_megamodel_into_workflow(df_orig: pl.DataFrame, prep, model, 
                                             target: str = 'breakout_long',
                                             enable_mega_models: bool = True,
                                             mega_model_size: int = 3) -> Dict[str, Any]:
    """
    Integration wrapper for the enhanced megamodel experiment with mega model capabilities.
    
    This function provides a simple interface to run the comprehensive enhanced megamodel
    experiment that tests both data sampling strategies and mega model approaches.
    Fully compatible with UEL workflows.
    
    Parameters:
    -----------
    df_orig : pl.DataFrame
        Original dataset to experiment with
    prep : callable
        Data preprocessing function (UEL compatible)
    model : callable
        Model training function (UEL compatible)
    target : str, default='breakout_long'
        Target column name for regression
    enable_mega_models : bool, default=True
        Whether to create and test mega models alongside single models
    mega_model_size : int, default=3
        Number of models to include in mega model approaches
    
    Returns:
    --------
    Dict[str, Any]
        Complete results including:
        - detailed_results: Full results for each strategy (single + mega models)
        - comparison_summary: DataFrame comparing all approaches
        - best_strategy: Name of best-performing data sampling strategy
        - best_approach: Best modeling approach (single/mega model)
        - mega_models_enabled: Whether mega model methods were used
    
    Actions:
    --------
    1. Display experiment setup and configuration
    2. Execute comprehensive strategy + mega model testing using UEL
    3. Compare single models vs data-split mega models vs cross-dataset mega models
    4. Identify optimal combination of data strategy and modeling approach
    5. Return detailed results for further analysis or model deployment
    """
    print(f"Starting Enhanced Megamodel Experiment with UEL Integration...")
    print(f"Dataset size: {len(df_orig)} rows")
    print(f"Target: {target}")
    print(f"Mega model methods enabled: {enable_mega_models}")
    if enable_mega_models:
        print(f"Mega model size: {mega_model_size} models")
    print("-" * 80)
    
    # Run the enhanced megamodel experiment
    results = run_enhanced_megamodel_with_uel(
        df_orig=df_orig,
        prep_func=prep,
        model_func=model,
        target=target,
        experiment_base_name=f"enhanced_megamodel_{target}",
        enable_mega_models=enable_mega_models,
        mega_model_size=mega_model_size
    )
    
    # Print summary of what was tested
    print(f"\n📋 EXPERIMENT SUMMARY:")
    print(f"   Strategies tested: {len(results['detailed_results'])}")
    
    total_single_models = sum(len(strategy_results['single_models']) 
                             for strategy_results in results['detailed_results'].values())
    print(f"   Total single models trained: {total_single_models}")
    
    if enable_mega_models:
        total_ds_mega_models = sum(len(strategy_results['data_split_mega_models']) 
                                  for strategy_results in results['detailed_results'].values())
        total_cd_mega_models = sum(len(strategy_results['cross_dataset_mega_models']) 
                                  for strategy_results in results['detailed_results'].values())
        print(f"   Data-split mega models created: {total_ds_mega_models}")
        print(f"   Cross-dataset mega models created: {total_cd_mega_models}")
    
    return results


# Additional utility functions for UEL compatibility

def get_best_model_from_results(results: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    """
    Extract the single best-performing model and its details from experiment results.
    
    Parameters:
    -----------
    results : Dict[str, Any]
        Results from run_enhanced_megamodel_with_uel()
    
    Returns:
    --------
    Tuple[Any, Dict[str, Any]]
        Best model object and dictionary with its performance details
    """
    if not results['comparison_summary'].empty:
        best_strategy = results['best_strategy']
        best_approach = results['best_approach']
        
        strategy_results = results['detailed_results'][best_strategy]
        
        if 'single' in best_approach:
            # Best single model
            single_models = strategy_results['single_models']
            best_single = min(single_models, key=lambda x: x['best_mae'])
            return best_single['best_model'], {
                'type': 'single_model',
                'strategy': best_strategy,
                'mae': best_single['best_mae'],
                'rmse': best_single['best_rmse'],
                'r2': best_single['best_r2']
            }
        
        elif 'ds_mega_model' in best_approach:
            # Best data-split mega model
            ds_mega_models = strategy_results['data_split_mega_models']
            best_ds_mega_model = min(ds_mega_models, key=lambda x: x['mega_model_mae'])
            return best_ds_mega_model['models'], {
                'type': 'data_split_mega_model',
                'strategy': best_strategy,
                'mae': best_ds_mega_model['mega_model_mae'],
                'rmse': best_ds_mega_model['mega_model_rmse'],
                'r2': best_ds_mega_model['mega_model_r2'],
                'mega_model_size': best_ds_mega_model['mega_model_size']
            }
        
        elif 'cd_mega_model' in best_approach:
            # Best cross-dataset mega model
            cd_mega_model = strategy_results['cross_dataset_mega_models'][0]
            return cd_mega_model['models_used'], {
                'type': 'cross_dataset_mega_model',
                'strategy': best_strategy,
                'mae': cd_mega_model['mega_model_mae'],
                'rmse': cd_mega_model['mega_model_rmse'],
                'r2': cd_mega_model['mega_model_r2'],
                'mega_model_size': cd_mega_model['mega_model_size']
            }
    
    return None, {}


def predict_with_best_model(results: Dict[str, Any], new_data: np.ndarray) -> np.ndarray:
    """
    Make predictions using the best model/mega model identified from the experiment.
    
    Parameters:
    -----------
    results : Dict[str, Any]
        Results from run_enhanced_megamodel_with_uel()
    new_data : np.ndarray
        New data to make predictions on (preprocessed)
    
    Returns:
    --------
    np.ndarray
        Predictions from the best-performing model/mega model
    """
    best_model, model_info = get_best_model_from_results(results)
    
    if best_model is None:
        raise ValueError("No best model found in results")
    
    model_type = model_info['type']
    
    if model_type == 'single_model':
        # Single model prediction
        return best_model.predict(new_data)
    
    elif model_type in ['data_split_mega_model', 'cross_dataset_mega_model']:
        # Mega model prediction - average predictions from all models
        if model_type == 'data_split_mega_model':
            models = best_model  # List of models
        else:
            models = [m['model'] for m in best_model]  # Extract model objects
        
        predictions = []
        for model in models:
            preds = model.predict(new_data)
            predictions.append(preds)
        
        return np.mean(predictions, axis=0)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def save_experiment_results(results: Dict[str, Any], base_filename: str = "enhanced_megamodel_results"):
    """
    Save comprehensive experiment results to multiple files for analysis.
    
    Parameters:
    -----------
    results : Dict[str, Any]
        Results from run_enhanced_megamodel_with_uel()
    base_filename : str
        Base filename for saving results (will be extended with suffixes)
    """
    import pickle
    import json
    
    # Save comparison summary
    if not results['comparison_summary'].empty:
        results['comparison_summary'].to_csv(f"{base_filename}_comparison.csv", index=False)
        print(f"✅ Comparison summary saved to {base_filename}_comparison.csv")
    
    # Save detailed results (pickle for full objects)
    with open(f"{base_filename}_detailed.pkl", 'wb') as f:
        pickle.dump(results['detailed_results'], f)
    print(f"✅ Detailed results saved to {base_filename}_detailed.pkl")
    
    # Save summary info as JSON
    summary_info = {
        'best_strategy': results['best_strategy'],
        'best_approach': results['best_approach'],
        'mega_models_enabled': results['mega_models_enabled'],
        'strategies_tested': list(results['detailed_results'].keys()),
        'total_experiments': sum(len(strategy_results['single_models']) 
                               for strategy_results in results['detailed_results'].values())
    }
    
    with open(f"{base_filename}_summary.json", 'w') as f:
        json.dump(summary_info, f, indent=2)
    print(f"✅ Summary info saved to {base_filename}_summary.json")


# Note: The original create_partitioned_dataset_mega_model_predictions functionality
# is now built into run_mega_model_experiment() via create_data_split_mega_model()
# No separate function needed - everything is integrated into the main workflow


# =============================================================================
# MAIN ENTRY POINT - Use this function for everything
# =============================================================================

def run_mega_model_experiment(df_orig: pl.DataFrame, prep_func, model_func, 
                             target: str = 'breakout_long',
                             enable_mega_models: bool = True,
                             mega_model_size: int = 3) -> Dict[str, Any]:
    """
    THE MAIN FUNCTION - Run complete mega model experiment to find optimal approach.
    
    This is the single entry point that:
    1. Tests all data sampling strategies (full, bootstrap, temporal, etc.)
    2. For each strategy: tests single models vs mega models
    3. Includes the original create_partitioned_* mega model approach
    4. Finds the absolute best combination
    5. Returns everything you need for production
    
    Parameters:
    -----------
    df_orig : pl.DataFrame
        Your complete dataset
    prep_func : callable
        Your data preprocessing function (same as you use with UEL)
    model_func : callable  
        Your model function (same as you use with UEL)
    target : str, default='breakout_long'
        Target column name
    enable_mega_models : bool, default=True
        Whether to test mega models (recommended: True)
    mega_model_size : int, default=3
        Number of models in each mega model
    
    Returns:
    --------
    Dict with everything you need:
        - 'best_model': The actual best model/mega model ready for production
        - 'best_info': Performance details of the best approach
        - 'comparison_summary': DataFrame comparing all approaches
        - 'how_to_use': Instructions for making predictions
    """
    print("🚀 Starting Complete Mega Model Experiment")
    print(f"   Dataset: {len(df_orig)} rows, Target: {target}")
    print(f"   Mega models: {'Enabled' if enable_mega_models else 'Disabled'}")
    print("=" * 80)
    
    # Run the comprehensive experiment
    full_results = integrate_enhanced_megamodel_into_workflow(
        df_orig=df_orig,
        prep=prep_func,
        model=model_func,
        target=target,
        enable_mega_models=enable_mega_models,
        mega_model_size=mega_model_size
    )
    
    # Extract the best model for easy use
    best_model, best_info = get_best_model_from_results(full_results)
    
    # Create simple prediction function
    def make_predictions(new_data: np.ndarray) -> np.ndarray:
        """Make predictions with the best model found."""
        return predict_with_best_model(full_results, new_data)
    
    # Simplified results for easy use
    simple_results = {
        'best_model': best_model,
        'best_info': best_info,
        'comparison_summary': full_results['comparison_summary'],
        'make_predictions': make_predictions,
        'how_to_use': {
            'to_predict': "Use results['make_predictions'](your_preprocessed_data)",
            'best_approach': f"{best_info.get('type', 'unknown')} from {best_info.get('strategy', 'unknown')} strategy",
            'performance': f"MAE: {best_info.get('mae', 'N/A'):.4f}" if 'mae' in best_info else "N/A"
        },
        '_full_details': full_results  # Hidden full results if needed
    }
    
    print(f"\n🎯 BEST APPROACH FOUND:")
    print(f"   {simple_results['how_to_use']['best_approach']}")
    print(f"   {simple_results['how_to_use']['performance']}")
    print(f"\n💡 TO MAKE PREDICTIONS:")
    print(f"   {simple_results['how_to_use']['to_predict']}")
    
    return simple_results


# Simple example - this is all you need:
"""
import polars as pl

# Your data and functions (same as you use with UEL)
df = pl.read_csv("data.csv")

def my_prep(dataset):
    # Your prep logic
    return preprocessed_data

def my_model(params):
    # Your model logic  
    return model

# Run experiment - this does everything
results = run_mega_model_experiment(
    df_orig=df,
    prep_func=my_prep,
    model_func=my_model,
    target='your_target'
)

# Use best model
predictions = results['make_predictions'](new_data)
print(f"Best approach: {results['how_to_use']['best_approach']}")
"""