import random
import polars as pl

class ParamSpace:
    
    '''
    Create parameter space manager for hyperparameter sampling.
    
    Args:
        params (dict): Dictionary of parameter names and their possible values.
        n_permutations (int): Number of parameter combinations to sample.
    '''
    
    def __init__(self, params: dict, n_permutations: int):

        self.params = params
        self.keys = list(params.keys())
        self.param_sizes = [len(params[k]) for k in self.keys]
        self.total_space = 1
        for size in self.param_sizes:
            self.total_space *= size

        # Generate n_permutations unique random indices
        if n_permutations >= self.total_space:
            indices = list(range(self.total_space))
        else:
            indices = random.sample(range(self.total_space), n_permutations)

        # Convert indices to parameter combinations
        combos = [self._index_to_combo(idx) for idx in indices]
        self.df_params = pl.DataFrame(combos)
        self.n_permutations = self.df_params.height

    def _index_to_combo(self, index):
        combo = {}
        remaining_index = index
        for i, key in enumerate(self.keys):
            size = self.param_sizes[i]
            combo[key] = self.params[key][remaining_index % size]
            remaining_index //= size
        return combo

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
              .with_row_index('__idx')
              .filter(pl.col('__idx') != row_no)
              .drop('__idx')
        )

        return round_params
