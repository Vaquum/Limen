import numpy as np
import time
from tqdm import tqdm
import polars as pl
import sqlite3


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
            n_permutations=10):
        
        for i in tqdm(range(n_permutations)):
            
            start_time = time.time()

            round_params = self._generate_permutation()

            data, round_params = self.prep(self.data, round_params)
         
            model, round_results = self.model(data=data, round_params=round_params)

            # Then, add id and execution time to demark end of of measurement result cols
            round_results['id'] = i
            round_results['execution_time'] = time.time() - start_time

            # Then, add the permutation parameters
            for key in round_params.keys():
                round_results[key] = round_params[key]

            if i == 0:
                self.log_df = pl.DataFrame(round_results)
            else:
                self.log_df = self.log_df.vstack(pl.DataFrame([round_results]))

            self.log_df.to_pandas().tail(1).to_sql(experiment_name,
                                                   self.conn,
                                                   if_exists="append",
                                                   index=False)

            log_string = f"{', '.join(map(str, self.log_df.row(i)))}\n"
            with open(experiment_name + '.csv', 'a') as f:
                f.write(log_string)

        self.conn.close()
            
    def _generate_permutation(self):
        
        out_dict = {}

        for key in self.params.keys():
            out_dict[key] = np.random.choice(list(self.params[key]))

        return out_dict
