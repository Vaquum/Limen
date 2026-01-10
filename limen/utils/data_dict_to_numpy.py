def data_dict_to_numpy(data: dict, keys: list = None) -> dict:

    '''
    Convert data dictionary entries from polars/pandas to numpy arrays.

    Args:
        data (dict): Dictionary with data entries (e.g., x_train, y_train, etc.)
        keys (list): Keys to convert. Defaults to x_train, y_train, x_val, y_val, x_test, y_test

    Returns:
        dict: Dictionary with values converted to numpy arrays where applicable
    '''

    if keys is None:
        keys = ['x_train', 'y_train', 'x_val', 'y_val', 'x_test', 'y_test']

    result = {}
    for key in keys:
        if key in data:
            val = data[key]
            result[key] = val.to_numpy() if hasattr(val, 'to_numpy') else val

    return result
