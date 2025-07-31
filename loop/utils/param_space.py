import random
import polars as pl
from itertools import product

class ParamSpace:
    
    def __init__(self, params, n_permutations):

        keys = list(params)
        combos = [dict(zip(keys, c)) for c in product(*(params[k] for k in keys))]
        combos = random.sample(combos, k=n_permutations)
        self.df_params = pl.DataFrame(combos)
        self.n_permutations = self.df_params.height

    def generate(self, random_search=True):
        
        if self.df_params.is_empty():
            return None

        if random_search:
            row_no = random.randrange(self.df_params.height)
        else:
            row_no = 0

        round_params = dict(zip(self.df_params.columns, self.df_params.row(row_no)))

        self.df_params = (
            self.df_params
              .with_row_index("__idx")
              .filter(pl.col("__idx") != row_no)
              .drop("__idx")
        )

        return round_params
