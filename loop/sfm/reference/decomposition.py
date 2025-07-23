import numpy as np
from scipy.stats import entropy, wasserstein_distance
from tslearn.metrics import dtw
import random
import zlib

def params():

    return {
        'n_events': [1000, 2000, 4000],
        'shift': list(range(1, 11)),
        'noise': [random.uniform(-1000, 1000) for _ in range(5)]
    }

def prep(data):

    return data

def model(data, params):

    def prep_for_measure_transformation_simplicity(x):
        
        x = np.array(x)
        x = x - np.mean(x)
        x = x / (np.std(x) + 1e-12)
        
        return x
    
    def compressibility_ratio(x):
        
        raw = str(x).encode("utf-8")
        compressed = zlib.compress(raw)
        return len(compressed) / len(raw)
    
    def measure_transformation_simplicity(original, transformed):
        
        value_shift = entropy(original, transformed) + wasserstein_distance(original, transformed)
    
        original = prep_for_measure_transformation_simplicity(original)
        transformed = prep_for_measure_transformation_simplicity(transformed)
        
        order_shift = dtw(original, transformed)
        structure_gain = compressibility_ratio(transformed) - compressibility_ratio(original)
        
        return {
            "value": round(value_shift.item(), 3),
            "position": round(order_shift, 3),
            "structure": round(structure_gain, 3)}
    
    shift = params['shift']
    original = data['price'].tail(params['n_events'])[shift:]
    transformed = data['price'].tail(params['n_events']).shift(shift)[shift:] + params['noise']
    
    measurement_result = measure_transformation_simplicity(original, transformed)

    return measurement_result, list(measurement_result.keys())
