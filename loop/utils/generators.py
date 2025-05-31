import numpy as np


def generate_permutation(params):
    
    out_dict = {}

    for key in params().keys():
        out_dict[key] = np.random.choice(params()[key])

    return out_dict


def generate_parameter_range(start, stop, step, round_by=3):

    '''Generates a list of rounded values'''

    values = np.arange(start, stop, step).tolist()
    values = [round(value, round_by) for value in values]

    return values
