import time
from tqdm import tqdm
import polars as pl
import sqlite3

from loop.utils.param_space import ParamSpace
from loop.log.log import Log
from loop.explorer.loop_explorer import loop_explorer


class UniversalExperimentLoop:
    '''UniversalExperimentLoop class for running experiments.'''

    def __init__(self,
                 data,
                 single_file_model=None,
                 manifest=None):
        '''
        Initializes the UniversalExperimentLoop.
        
        Args:
            data (pl.DataFrame): The data to use for the experiment.
            single_file_model (SingleFileModel or module, optional): Legacy single file model or manifest-based module
            manifest (Manifest, optional): Manifest-based configuration (overrides single_file_model.manifest if present)
        '''
        self.data = data
        self.manifest = manifest
        
        # Support both legacy and manifest approaches
        if manifest is not None:
            # Explicit manifest provided
            self.model = None
            self.prep = manifest.prepare_data
            self.params = None
        elif single_file_model is not None:
            # Check if single_file_model has manifest attribute (new style)
            if hasattr(single_file_model, 'manifest') and callable(single_file_model.manifest):
                # New manifest-based module
                self.manifest = single_file_model.manifest()
                self.model = None
                self.prep = self.manifest.prepare_data
                self.params = single_file_model.params() if hasattr(single_file_model, 'params') else None
            elif hasattr(single_file_model, 'model'):
                # Legacy single file model
                self.model = single_file_model.model
                self.params = single_file_model.params() if callable(getattr(single_file_model, 'params', None)) else single_file_model.params
                self.prep = single_file_model.prep
            else:
                raise ValueError("single_file_model must have either 'manifest' or 'model' attribute")
        else:
            raise ValueError("Either single_file_model or manifest must be provided")
        
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

        Args:
            experiment_name (str): The name of the experiment.
            n_permutations (int): The number of permutations to run.
            prep_each_round (bool): Whether to use `prep` for each round or just first.
            random_search (bool): Whether to use random search or not.
            maintain_details_in_params (bool): Whether to maintain experiment details in params.
            context_params (dict): The context parameters to use for the experiment.
            save_to_sqlite (bool): Whether to save the results to a SQLite database.
            params (dict or function): The parameters to use for the experiment.
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

        # Handle params override
        if params is not None:
            if callable(params):
                self.params = params()
            else:
                self.params = params
        
        # Validate params are available
        if self.params is None:
            raise ValueError("params must be provided either in initialization or run() call")
        
        # Handle prep override
        if prep is not None:
            self.prep = prep
        
        # Handle model override (for legacy mode)
        if model is not None:
            self.model = model

        self.param_space = ParamSpace(params=self.params,
                                      n_permutations=n_permutations)
        
        for i in tqdm(range(n_permutations)):
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

            # Data preparation
            if prep_each_round is True:
                # Prep with round_params every round
                if self.manifest:
                    data_dict = self.prep(self.data, round_params)
                else:
                    data_dict = self.prep(self.data, round_params=round_params)
            else:
                # Only prep on first round
                if i == 0:
                    if self.manifest:
                        data_dict = self.prep(self.data, round_params)
                    else:
                        data_dict = self.prep(self.data)

            # Model execution
            if self.manifest and self.manifest.model_function:
                # Use manifest's model configuration
                round_results = self.manifest.run_model(data_dict, round_params)
            elif self.model:
                # Use legacy model function
                round_results = self.model(data=data_dict, round_params=round_params)
            else:
                raise ValueError("No model function configured")

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