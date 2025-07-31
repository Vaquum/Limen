import random
import polars as pl
from itertools import product

class ParamSpace:
    
    def __init__(self, params):

        print("params", params)
        
        print("Starting to initialize ParamSpace")
        keys = list(params)
        print("keys generated")
        combos = [dict(zip(keys, c)) for c in product(*(params[k] for k in keys))]
        print("combos generated")
        self.df_params = pl.DataFrame(combos)
        print("df_params generated")
        self.n_permutations = self.df_params.height

    def generate(self, random_search=True):
        
        if self.df_params.is_empty():
            return None

        if random_search:
            row_no = random.randrange(self.df_params.height)
        else:
            row_no = 0

        print("Starting to generate round_params")
        round_params = dict(zip(self.df_params.columns, self.df_params.row(row_no)))
        print("round_params generated")

        print("Starting to filter df_params")
        self.df_params = (
            self.df_params
              .with_row_index("__idx")
              .filter(pl.col("__idx") != row_no)
              .drop("__idx")
        )
        print("df_params filtered")
        return round_params
