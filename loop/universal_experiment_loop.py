import numpy as np
import time
from tqdm import tqdm
import polars as pl
import sqlite3


class UniversalExperimentLoop:

    def __init__(self,
                 data,
                 model,
                 params):

        self.data = data
        self.model = model
        self.params = params()

        self.conn = sqlite3.connect("/opt/experiments/experiments.sqlite")

    def params(self):
        return self.params()    

    def prep(self):
        return self.prep(self.data)
    
    def model(self):
        return self.model(self.data, params=self.params)

    def run(self,
            experiment_name,
            n_permutations=10):
        
        for i in tqdm(range(n_permutations)):
            
            start_time = time.time()

            permutation = self._generate_permutation()

            # First, add the measurement results            
            results_dict, metric_col_names = self.model(data=self.data, params=permutation)

            # Then, add id and execution time to demark end of of measurement result cols
            results_dict['id'] = i
            results_dict['execution_time'] = time.time() - start_time

            # Then, add the permutation parameters
            for key in permutation.keys():
                results_dict[key] = permutation[key]

            if i == 0:
                self.log_df = pl.DataFrame(results_dict)
            else:
                self.log_df = self.log_df.vstack(pl.DataFrame([results_dict]))

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
