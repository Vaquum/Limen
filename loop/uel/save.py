import pickle
import os
from loop.explorer.loop_explorer import loop_explorer


def save(uel_instance, experiment_name):
    '''
    Save the UEL object to a file named f'{experiment_name}.uel'.
    
    This method saves the complete state of the UEL object after a run,
    including all data, results, parameters, models, and other artifacts.
    
    Args:
        uel_instance: The UniversalExperimentLoop instance to save
        experiment_name (str): The name for the saved file (without .uel extension)
    '''
    filename = f"{experiment_name}.uel"
    
    # Create a dictionary with all the important state
    # Note: We exclude functions that might not be pickleable
    state_dict = {
        'data': uel_instance.data,
        'experiment_log': getattr(uel_instance, 'experiment_log', None),
        'round_params': getattr(uel_instance, 'round_params', []),
        'preds': getattr(uel_instance, 'preds', []),
        'scalers': getattr(uel_instance, 'scalers', []),
        'extras': getattr(uel_instance, 'extras', []),
        'models': getattr(uel_instance, 'models', []),
        '_alignment': getattr(uel_instance, '_alignment', []),
        'params': uel_instance.params,
        # Include computed results if they exist
        'experiment_confusion_metrics': getattr(uel_instance, 'experiment_confusion_metrics', None),
        'experiment_backtest_results': getattr(uel_instance, 'experiment_backtest_results', None),
        'experiment_parameter_correlation': getattr(uel_instance, 'experiment_parameter_correlation', None),
        '_log': getattr(uel_instance, '_log', None)
    }
    
    # Note: We intentionally exclude 'model' and 'prep' functions as they may not be serializable
    # Users should reconstruct UEL with original SFM if they need to run again
    
    try:
        with open(filename, 'wb') as f:
            pickle.dump(state_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        # If pickle fails, try to save with more limited state
        try:
            limited_state = {
                'data': uel_instance.data,
                'experiment_log': getattr(uel_instance, 'experiment_log', None),
                'round_params': getattr(uel_instance, 'round_params', []),
                'preds': getattr(uel_instance, 'preds', []),
                '_alignment': getattr(uel_instance, '_alignment', []),
                'params': uel_instance.params
            }
            with open(filename, 'wb') as f:
                pickle.dump(limited_state, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e2:
            raise RuntimeError(f'Failed to save UEL object to {filename}: {str(e)} (Limited save also failed: {str(e2)})')


def load(uel_instance, filepath):
    '''
    Load UEL object state from a .uel file.
    
    This method restores the complete state of a UEL object that was previously
    saved using the save() method, recreating the exact same state as after 
    the original run.
    
    Note: The original 'model' and 'prep' functions are not restored and will be None.
    If you need to run the UEL again, initialize with the original SFM.
    
    Args:
        uel_instance: The UniversalExperimentLoop instance to load into
        filepath (str): Path to the .uel file to load
    '''
    if not os.path.exists(filepath):
        raise FileNotFoundError(f'UEL file not found: {filepath}')
    
    try:
        with open(filepath, 'rb') as f:
            state_dict = pickle.load(f)
        
        # Restore all attributes
        for key, value in state_dict.items():
            setattr(uel_instance, key, value)
        
        # Set function attributes to None since they weren't saved
        if not hasattr(uel_instance, 'model'):
            uel_instance.model = None
        if not hasattr(uel_instance, 'prep'):
            uel_instance.prep = None
            
        # Initialize missing attributes to empty if they don't exist
        for attr in ['extras', 'models', 'round_params', 'preds', 'scalers', '_alignment']:
            if not hasattr(uel_instance, attr):
                setattr(uel_instance, attr, [])
                
        # Recreate the explorer function if we have the log
        if hasattr(uel_instance, '_log') and uel_instance._log is not None:
            def _explorer():
                loop_explorer(uel_instance)
            uel_instance.explorer = _explorer
        
    except Exception as e:
        raise RuntimeError(f'Failed to load UEL object from {filepath}: {str(e)}')