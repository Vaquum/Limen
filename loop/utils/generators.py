import numpy as np


def generate_permutation(params: callable) -> dict:
    
    '''
    Compute random parameter permutation from parameter space.
    
    Args:
        params (callable): Function that returns dictionary of parameter options
        
    Returns:
        dict: Dictionary with randomly selected values for each parameter
    '''
    
    out_dict = {}

    for key in params().keys():
        out_dict[key] = np.random.choice(params()[key])

    return out_dict


def generate_parameter_range(start: float, stop: float, step: float, round_by: int = 3) -> list:
    
    '''
    Compute list of rounded parameter values within specified range.
    
    Args:
        start (float): Starting value of the range
        stop (float): Ending value of the range
        step (float): Step size between values
        round_by (int): Number of decimal places to round to
        
    Returns:
        list: List of rounded values from start to stop with specified step
    '''

    values = np.arange(start, stop, step).tolist()
    values = [round(value, round_by) for value in values]

    return values
