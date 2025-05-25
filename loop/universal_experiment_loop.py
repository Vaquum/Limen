import numpy as np
import time
from tqdm import tqdm
import polars as pl
import sqlite3

from loop import data


class UniversalExperimentLoop:

    def __init__(self,
                 data,
                 single_file_model):

        self.data = data
        self.model = single_file_model.model
        self.params = single_file_model.params()
        self.prep = single_file_model.prep

        self.conn = sqlite3.connect("/opt/experiments/experiments.sqlite")

    def run(self,
            experiment_name,
            n_permutations=10,
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
            params (dict): The parameters to use for the experiment
            prep (function): The function to use to prepare the data
            model (function): The function to use to run the model
        '''

        if params is not None:
            self.params = params
        
        if prep is not None:
            self.prep = prep
        
        if model is not None:
            self.model = model
            
        for i in tqdm(range(n_permutations)):
            
            if i == 0:
                data = self.prep(self.data)
            
            start_time = time.time()

            round_params = self._generate_permutation()
    
            round_results = self.model(data=data, round_params=round_params)

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
            
    def _generate_permutation(self):
        
        out_dict = {}

        for key in self.params.keys():
            out_dict[key] = np.random.choice(list(self.params[key]))

        return out_dict
