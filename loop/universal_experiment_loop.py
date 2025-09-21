import time
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
                 data,
                 single_file_model):
        
        '''Initializes the UniversalExperimentLoop.
        
        Args:
            data (pl.DataFrame): The data to use for the experiment.
            single_file_model (SingleFileModel): The single file model to use for the experiment.
        '''

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
            model=None,
            manifest=None):
        
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

        self.manifest = manifest

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
                if self.manifest:
                    data_dict = self.prep(
                        self.data,
                        round_params=round_params,
                        manifest=self.manifest)
                else:
                    data_dict = self.prep(self.data, round_params=round_params)

            # Otherwise, only for the first round, prep data without round_params passed in
            else:
                if i == 0:
                    if manifest:
                        data_dict = self.prep(self.data, manifest=manifest)
                    else:
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
