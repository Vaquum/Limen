import random
import polars as pl
from itertools import product

class ParamSpace:
    
    '''
    Create parameter space manager for hyperparameter sampling.
    
    Args:
        params (dict): Dictionary of parameter names and their possible values.
        n_permutations (int): Number of parameter combinations to sample.
    '''
    
    def __init__(self, params: dict, n_permutations: int):

        keys = list(params)
        combos = [dict(zip(keys, c)) for c in product(*(params[k] for k in keys))]
        combos = random.sample(combos, k=n_permutations)
        self.df_params = pl.DataFrame(combos)
        self.n_permutations = self.df_params.height

    def generate(self, random_search: bool = True) -> dict:
        
        '''
        Compute next parameter combination from the parameter space.
        
        Args:
            random_search (bool): Whether to select parameters randomly or sequentially
            
        Returns:
            dict: Dictionary of parameter names and selected values, or None if space is exhausted
        '''
        
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
