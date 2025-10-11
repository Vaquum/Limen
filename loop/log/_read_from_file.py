import pandas as pd


def _read_from_file(self, file_path: str) -> pd.DataFrame:
    
    '''
    Create cleaned experiment log DataFrame from file.
    
    Args:
        file_path (str): Path to experiment log CSV file
        
    Returns:
        pd.DataFrame: Cleaned log data with whitespace-trimmed object columns
    '''
    
    with open(file_path, 'r') as f:

        lines = f.readlines()

        for i, line in enumerate(lines):

            if i != 0:
                if line.startswith('recall'):
                    lines.pop(i)

    with open('__temp__.csv', 'w') as f:
        
        f.writelines(lines)

    data = pd.read_csv('__temp__.csv')

    for col in data.columns:
        if data[col].dtype == object:
            
            mask = data[col].notnull()
            data[col] = data[col].mask(mask, data[col].astype(str).str.strip())

    return data
