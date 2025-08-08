import numpy as np
import pandas as pd

def probabilities_to_classes(probs_df,
                             sideways_fraction=0.4):

    '''
    Compute classes from probabilities for three classes (-1, 0, 1).
    
    Args:
        probs_df (pd.DataFrame): Columns [-1, 0, 1] with probabilities
        sideways_fraction (float): Fraction of points to label as sideways (0)

    Returns:
        np.ndarray: Array of class labels (-1, 0, 1)
    '''
    
    # Assign class with highest probability initially
    initial_classes = probs_df.idxmax(axis=1).astype(int)

    # Determine the number of sideways labels
    num_sideways = int(len(probs_df) * sideways_fraction)

    # Calculate the "sideways confidence" (probability of class 0)
    sideways_confidence = probs_df[0]

    # Get indices for top sideways probabilities
    sideways_indices = sideways_confidence.sort_values(ascending=False).iloc[:num_sideways].index

    # Set sideways labels
    final_classes = initial_classes.copy()
    final_classes.loc[sideways_indices] = 0

    # For remaining indices, set label to either bullish (1) or bearish (-1)
    remaining_indices = final_classes[final_classes != 0].index
    final_classes.loc[remaining_indices] = probs_df.loc[remaining_indices, [-1, 1]].idxmax(axis=1).astype(int)

    return final_classes.values
