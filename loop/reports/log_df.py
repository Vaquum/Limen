from collections import OrderedDict

import pandas as pd
import wrangle


def read_from_file(file_path: str) -> object:
    
    '''
    Read experiment log file and return cleaned dataframe.
    
    Args:
        file_path (str): Path to experiment log CSV file
        
    Returns:
        object: Cleaned pandas DataFrame with experiment log data
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

    # Trim leading/trailing spaces from all string values in the dataframe
    for col in data.columns:
        if data[col].dtype == object:
            mask = data[col].notnull()
            data[col] = data[col].mask(mask, data[col].astype(str).str.strip())

    return data


def outcome_df(log_df: object,
               cols_to_drop: list,
               cols_to_multilabel: list = None,
               type: str = 'categorical') -> object:
    
    '''
    Create outcome-based dataframe from experiment log data.
    
    Args:
        log_df (object): Output dataframe from read_from_file function
        cols_to_drop (list): Column names to remove from dataframe
        cols_to_multilabel (list): Column names to convert to multilabel format
        type (str): Outcome type, must be 'categorical' or 'regression'
        
    Returns:
        object: Outcome-based dataframe sorted by performance
    '''

    if type == 'categorical':
        log_df['outcome'] = ((log_df['recall'] + log_df['precision'] + log_df['auc'] + log_df['accuracy']) / 4).round(2)

    elif type == 'regression':
        assert False, f"Regression not implemented"

    else:
        assert False, f"Invalid type, has to be regression or categorical."

    if cols_to_multilabel is not None:
        for col in cols_to_multilabel:
            log_df = wrangle.col_to_multilabel(log_df, col)

    log_df.drop(cols_to_drop, axis=1, inplace=True)

    return log_df.sort_values('outcome', ascending=False)


def corr_df(outcome_df: object) -> object:
    
    '''
    Compute feature correlation dataframe from outcome-based data.
    
    Args:
        outcome_df (object): Output dataframe from outcome_df function
        
    Returns:
        object: Feature correlation dataframe with outcome correlations
    '''

    corr_dict = OrderedDict()

    for head in [len(outcome_df), len(outcome_df) // 5, 100, 50, 10, 5, 1]:
        corr_dict[f"top_{head}_corr"] = outcome_df.head(head).corr()["outcome"]

    df_corr = pd.DataFrame(corr_dict)

    return df_corr
