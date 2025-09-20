import time
import pickle
import os
from tqdm import tqdm
import polars as pl
import sqlite3

from loop.utils.param_space import ParamSpace
from loop.log.log import Log
from loop.explorer.loop_explorer import loop_explorer


class UniversalExperimentLoop:

    '''
    UniversalExperimentLoop class for running experiments.
    '''

    def __init__(self,
                 data=None,
                 single_file_model=None,
                 filepath=None):
        
        '''Initializes the UniversalExperimentLoop.
        
        Args:
            data (pl.DataFrame, optional): The data to use for the experiment.
            single_file_model (SingleFileModel, optional): The single file model to use for the experiment.
            filepath (str, optional): Path to a saved .uel file to load from.
        '''

        if filepath is not None:
            # Load from saved file
            self.load(filepath)
        else:
            # Initialize normally
            if data is None or single_file_model is None:
                raise ValueError("Either filepath must be provided, or both data and single_file_model must be provided")
            
            self.data = data
            self.model = single_file_model.model
            self.params = single_file_model.params()
            self.prep = single_file_model.prep
            self.extras = []
            self.models = []

    def run(self,
            experiment_name,
            n_permutations=10000,
            prep_each_round=False,
            random_search=True,
            maintain_details_in_params=False,
            context_params=None,
            save_to_sqlite=False,
            params=None,
            prep=None,
            model=None):
        
        '''
        Run the experiment `n_permutations` times. 

        NOTE: If you want to use a custom `params` or `prep` or `model`
        function, you can pass them as arguments and permanently change
        `single_file_model` for that part. Make sure that the inputs and
        returns are same as the ones outlined in `docs/Single-File-Model.md`.

        Args:
            experiment_name (str): The name of the experiment.
            n_permutations (int): The number of permutations to run.
            prep_each_round (bool): Whether to use `prep` for each round or just first.
            random_search (bool): Whether to use random search or not.
            maintain_details_in_params (bool): Whether to maintain experiment details in params.
            context_params (dict): The context parameters to use for the experiment.
            save_to_sqlite (bool): Whether to save the results to a SQLite database.
            params (dict): The parameters to use for the experiment.
            prep (function): The function to use to prepare the data.
            model (function): The function to use to run the model.

        Returns:
            pl.DataFrame: The results of the experiment
        '''

        self.round_params = []
        self.models = []
        self.preds = []
        self.scalers = []
        self._alignment = []
        
        if save_to_sqlite is True:
            self.conn = sqlite3.connect("/opt/experiments/experiments.sqlite")

        if params is not None:
            self.params = params()
        
        if prep is not None:
            self.prep = prep
        
        if model is not None:
            self.model = model

        self.param_space = ParamSpace(params=self.params,
                                      n_permutations=n_permutations)
        
        for i in tqdm(range(n_permutations)):

            # Start counting execution_time            
            start_time = time.time()

            # Generate the parameter values for the current round
            round_params = self.param_space.generate(random_search=random_search)

            # Add context parameters to round_params
            if context_params is not None:
                round_params.update(context_params)

            # Add experiment details to round_params
            if maintain_details_in_params is True:
                round_params['_experiment_details'] = {
                    'current_index': i,
                }

            # Always prep data with round_params passed in
            if prep_each_round is True:
                data_dict = self.prep(self.data, round_params=round_params)

            # Otherwise, only for the first round, prep data without round_params passed in
            else:
                if i == 0:
                    data_dict = self.prep(self.data)

            # Perform the model training and evaluation
            round_results = self.model(data=data_dict, round_params=round_params)

            # Remove the experiment details from the results
            if maintain_details_in_params is True:
                round_params.pop('_experiment_details')

            # Add alignment details
            self._alignment.append(data_dict['_alignment'])

            # Handle any extra results that are returned from the model
            if 'extras' in round_results.keys():
                self.extras.append(round_results['extras'])
                round_results.pop('extras')

            # Handle any models that are returned from the model
            if 'models' in round_results.keys():
                self.models.append(round_results['models'])
                round_results.pop('models')

            if '_preds' in round_results.keys():
                self.preds.append(round_results['_preds'])
                round_results.pop('_preds')

            if '_scaler' in data_dict.keys():
                self.scalers.append(data_dict['_scaler'])
                data_dict.pop('_scaler')

            # Add the round number and execution time to the results
            round_results['id'] = i
            round_results['execution_time'] = round(time.time() - start_time, 2)

            self.round_params.append(round_params)

            for key in round_params.keys():
                round_results[key] = round_params[key]

            # Handle writing to the DataFrame
            if i == 0:
                self.experiment_log = pl.DataFrame(round_results)
            else:
                self.experiment_log = self.experiment_log.vstack(pl.DataFrame([round_results]))

            if save_to_sqlite is True:
                # Handle writing to the database
                self.experiment_log.to_pandas().tail(1).to_sql(experiment_name,
                                                    self.conn,
                                                    if_exists="append",
                                                    index=False)
            # Handle writing to the file
            if i == 0:
                header_colnames = ','.join(list(round_results.keys()))
                with open(experiment_name + '.csv', 'a') as f:
                    f.write(f"{header_colnames}\n")

            log_string = f"{', '.join(map(str, self.experiment_log.row(i)))}\n"
            with open(experiment_name + '.csv', 'a') as f:
                f.write(log_string)

        if save_to_sqlite is True:
            self.conn.close()

        # Add Log, Benchmark, and Backtest properties
        cols_to_multilabel = self.experiment_log.select(pl.col(pl.Utf8)).columns
        
        self._log = Log(uel_object=self, cols_to_multilabel=cols_to_multilabel)

        self.experiment_confusion_metrics = self._log.experiment_confusion_metrics('price_change')
        self.experiment_backtest_results = self._log.experiment_backtest_results()
        self.experiment_parameter_correlation = self._log.experiment_parameter_correlation

        def _explorer():
            loop_explorer(self)

        self.explorer = _explorer

    def save(self, experiment_name):
        '''
        Save the UEL object to a file named f"{experiment_name}.uel".
        
        This method saves the complete state of the UEL object after a run,
        including all data, results, parameters, models, and other artifacts.
        
        Args:
            experiment_name (str): The name for the saved file (without .uel extension)
        '''
        filename = f"{experiment_name}.uel"
        
        # Create a dictionary with all the important state
        # Note: We exclude functions that might not be pickleable
        state_dict = {
            'data': self.data,
            'experiment_log': getattr(self, 'experiment_log', None),
            'round_params': getattr(self, 'round_params', []),
            'preds': getattr(self, 'preds', []),
            'scalers': getattr(self, 'scalers', []),
            'extras': getattr(self, 'extras', []),
            'models': getattr(self, 'models', []),
            '_alignment': getattr(self, '_alignment', []),
            'params': self.params,
            # Include computed results if they exist
            'experiment_confusion_metrics': getattr(self, 'experiment_confusion_metrics', None),
            'experiment_backtest_results': getattr(self, 'experiment_backtest_results', None),
            'experiment_parameter_correlation': getattr(self, 'experiment_parameter_correlation', None),
            '_log': getattr(self, '_log', None)
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
                    'data': self.data,
                    'experiment_log': getattr(self, 'experiment_log', None),
                    'round_params': getattr(self, 'round_params', []),
                    'preds': getattr(self, 'preds', []),
                    '_alignment': getattr(self, '_alignment', []),
                    'params': self.params
                }
                with open(filename, 'wb') as f:
                    pickle.dump(limited_state, f, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as e2:
                raise RuntimeError(f"Failed to save UEL object to {filename}: {str(e)} (Limited save also failed: {str(e2)})")

    def load(self, filepath):
        '''
        Load UEL object state from a .uel file.
        
        This method restores the complete state of a UEL object that was previously
        saved using the save() method, recreating the exact same state as after 
        the original run.
        
        Note: The original 'model' and 'prep' functions are not restored and will be None.
        If you need to run the UEL again, initialize with the original SFM.
        
        Args:
            filepath (str): Path to the .uel file to load
        '''
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"UEL file not found: {filepath}")
        
        try:
            with open(filepath, 'rb') as f:
                state_dict = pickle.load(f)
            
            # Restore all attributes
            for key, value in state_dict.items():
                setattr(self, key, value)
            
            # Set function attributes to None since they weren't saved
            if not hasattr(self, 'model'):
                self.model = None
            if not hasattr(self, 'prep'):
                self.prep = None
                
            # Initialize missing attributes to empty if they don't exist
            for attr in ['extras', 'models', 'round_params', 'preds', 'scalers', '_alignment']:
                if not hasattr(self, attr):
                    setattr(self, attr, [])
                    
            # Recreate the explorer function if we have the log
            if hasattr(self, '_log') and self._log is not None:
                def _explorer():
                    loop_explorer(self)
                self.explorer = _explorer
            
        except Exception as e:
            raise RuntimeError(f"Failed to load UEL object from {filepath}: {str(e)}")
