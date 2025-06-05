import numpy as np
import time
from tqdm import tqdm
import polars as pl
import sqlite3

from loop.utils.param_space import ParamSpace


class UniversalExperimentLoop:

    def __init__(self,
                 data,
                 single_file_model):

        self.data = data
        self.model = single_file_model.model
        self.params = single_file_model.params()
        self.prep = single_file_model.prep
        self.extras = []
        self.models = []

    def run(self,
            experiment_name,
            n_permutations=None,
            prep_each_round=False,
            random_search=True,
            maintain_details_in_params=False,
            params=None,
            prep=None,
            model=None):
        
        '''
        Runs the experiment `n_permutations` times. 

        NOTE: If you want to use a custom `params` or `prep` or `model`
        function, you can pass them as arguments and permanently change
        `single_file_model` for that part. Make sure that the inputs and
        returns are same as the ones outlined in `docs/Single-File-Model.md`.

        Args:
            experiment_name (str): The name of the experiment
            n_permutations (int): The number of permutations to run
            prep_each_round (bool): Whether to use `prep` for each round or just first
            random_search (bool): Whether to use random search or not
            maintain_details_in_params (bool): Whether to maintain experiment details in params
            params (dict): The parameters to use for the experiment
            prep (function): The function to use to prepare the data
            model (function): The function to use to run the model
        '''

        self.conn = sqlite3.connect("/opt/experiments/experiments.sqlite")

        if params is not None:
            self.params = params()
        
        if prep is not None:
            self.prep = prep
        
        if model is not None:
            self.model = model

        self.param_space = ParamSpace(params=self.params)
            
        if n_permutations is None:
            n_permutations = self.param_space.n_permutations

        for i in tqdm(range(n_permutations)):

            # Start counting execution_time            
            start_time = time.time()

            # Generate the parameter values for the current round
            round_params = self.param_space.generate(random_search=random_search)

            # Add experiment details to round_params
            if maintain_details_in_params is True:
                round_params['_experiment_details'] = {
                    'current_index': i,
                }

            # Always prep data with round_params passed in
            if prep_each_round is True:
                data = self.prep(self.data, round_params=round_params)

            # Otherwise, only for the first round, prep data without round_params passed in
            else:
                if i == 0:
                    data = self.prep(self.data)

            # Perform the model training and evaluation
            round_results = self.model(data=data, round_params=round_params)

            # Remove the experiment details from the results
            if maintain_details_in_params is True:
                round_results.pop('_experiment_details')

            # Handle any extra results that are returned from the model
            if 'extras' in round_results.keys():
                self.extras.append(round_results['extras'])
                round_results.pop('extras')

            # Handle any models that are returned from the model
            if 'models' in round_results.keys():
                self.models.append(round_results['models'])
                round_results.pop('models')

            if '_scaler' in round_results.keys():
                self.scaler = round_results['_scaler']
                round_results.pop('_scaler')

            # Add the round number and execution time to the results
            round_results['id'] = i
            round_results['execution_time'] = round(time.time() - start_time, 2)

            for key in round_params.keys():
                round_results[key] = round_params[key]

            # Handle writing to the DataFrame
            if i == 0:
                self.log_df = pl.DataFrame(round_results)
            else:
                self.log_df = self.log_df.vstack(pl.DataFrame([round_results]))

            # Handle writing to the database
            self.log_df.to_pandas().tail(1).to_sql(experiment_name,
                                                   self.conn,
                                                   if_exists="append",
                                                   index=False)
            # Handle writing to the file
            if i == 0:
                header_colnames = ','.join(list(round_results.keys()))
                with open(experiment_name + '.csv', 'a') as f:
                    f.write(f"{header_colnames}\n")

            log_string = f"{', '.join(map(str, self.log_df.row(i)))}\n"
            with open(experiment_name + '.csv', 'a') as f:
                f.write(log_string)

        self.conn.close()
