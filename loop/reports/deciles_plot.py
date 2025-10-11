import numpy as np
import matplotlib.pyplot as plt


def deciles_plot(data: object, column: str) -> None:
    
    '''
    Create deciles visualization showing average outcome across data distribution.
    
    Args:
        data (object): Dataset containing the specified column for deciles analysis
        column (str): Column name to compute deciles for
        
    Returns:
        None: Displays matplotlib line plot with deciles analysis
    '''

    out = data[column]
    
    out_sorted = out.sort_values().reset_index(drop=True)
    
    n = len(out_sorted)
    idx = np.linspace(0, n, 11, dtype=int)
    smoothed = [
        out_sorted.iloc[idx[i] : idx[i+1]].mean()
        for i in range(10)
    ]
    
    x_steps = np.arange(1, 11)
    
    plt.figure(figsize=(6, 4))
    plt.plot(x_steps, smoothed, marker='o', linestyle='-')
    plt.xticks(x_steps)                   
    plt.xlabel("Step (1 → 10)")
    plt.ylabel("Average Outcome (per decile)")
    plt.title(f"Outcome deciles for {len(data)} rounds of experiment.")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()