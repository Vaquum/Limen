import matplotlib.pyplot as plt
import seaborn as sns


def confusion_matrix_plus(df_results, x, y_lim_correction=6.5, outlier_quantiles=[0.01, 0.99]):
    
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), facecolor='white')
    axes = axes.flatten()
    
    lower, upper = df_results[x].quantile(outlier_quantiles)
    df_filtered = df_results[(df_results[x] >= lower) & (df_results[x] <= upper)]
    
    conditions = [
        ((df_filtered['predictions'] == 1) & (df_filtered['actuals'] == 1), 'True Positive', '#9D8CD8'),
        ((df_filtered['predictions'] == 1) & (df_filtered['actuals'] == 0), 'False Positive', '#FFB3D9'),
        ((df_filtered['predictions'] == 0) & (df_filtered['actuals'] == 0), 'True Negative', '#9D8CD8'),
        ((df_filtered['predictions'] == 0) & (df_filtered['actuals'] == 1), 'False Negative', '#FFB3D9')
    ]
    
    xlim = (df_filtered[x].min(), df_filtered[x].max())
    max_freq = max([len(df_filtered[mask][x]) for mask, _, _ in conditions]) // 30 + 5
    
    # Calculate model score
    tp_mean = df_filtered[conditions[0][0]][x].mean()
    tp_median = df_filtered[conditions[0][0]][x].median()
    fp_mean = df_filtered[conditions[1][0]][x].mean()
    fp_median = df_filtered[conditions[1][0]][x].median()
    model_score = (tp_mean - fp_mean) + (tp_median - fp_median)
    
    fig.suptitle(f'Model Score: ${model_score:.0f}', fontsize=18, fontweight='bold', y=0.98)
    
    for ax, (mask, title, color) in zip(axes, conditions):
        ax.hist(df_filtered[mask][x], bins=30, color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
        mean_val = df_filtered[mask][x].mean()
        median_val = df_filtered[mask][x].median()
        win_rate = (df_filtered[mask][x] > 0).mean() * 100
        ax.text(0.7, 0.9, f'μ = ${mean_val:.0f}', transform=ax.transAxes, fontsize=12, color='#666666')
        ax.text(0.7, 0.84, f'M = ${median_val:.0f}', transform=ax.transAxes, fontsize=12, color='#666666')
        ax.text(0.7, 0.78, f'W = {win_rate:.0f}%', transform=ax.transAxes, fontsize=12, color='#666666')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel(x, fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_xlim(xlim)
        ax.set_ylim(0, max_freq*y_lim_correction)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout(pad=3.0)
    plt.show()