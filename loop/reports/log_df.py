from collections import OrderedDict

import pandas as pd
import wrangle


def read_from_file(file_path):

    with open(file_path, 'r') as f:
        
        lines = f.readlines()
        
        for i, line in enumerate(lines):
    
            if i != 0:
                if line.startswith('recall'):
                    lines.pop(i)
    
    with open('__temp__.csv', 'w') as f:
        f.writelines(lines)
    
    data = pd.read_csv('__temp__.csv')

    return data

def outcome_df(log_df):

    log_df['outcome'] = ((log_df['recall'] + log_df['precision'] + log_df['auc'] + log_df['accuracy']) / 4).round(2)
    
    log_df = wrangle.col_to_multilabel(log_df, 'solver')
    log_df = wrangle.col_to_multilabel(log_df, 'feature_to_drop')
    
    log_df.drop(['recall', 'precision', 'f1score', 'auc', 'accuracy', 'id', 'execution_time', 'penalty', 'shift', 'q'], axis=1, inplace=True)
    
    return log_df.sort_values('outcome', ascending=False)

def corr_df(outcome_log_df):

    corr_dict = OrderedDict()
    
    for head in [len(outcome_log_df), len(outcome_log_df) // 5, 100, 50, 10, 5, 1]:
        corr_dict[f"top_{head}_corr"] = outcome_log_df.head(head).corr()["outcome"]
    
    df_corr = pd.DataFrame(corr_dict)

    return df_corr.transpose()
