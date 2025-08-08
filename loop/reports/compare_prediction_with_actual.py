import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats


def compare_prediction_with_actual(prep, model, validate):

    '''
    Compare predictions with actual values.

    Args:
        prep (function): The prep function used in the experiment
        model (function): The `model` function used in the experiment
        validate (dict): The validation data

    Returns:
        None
    '''

    x_validate = prep(validate, mode='predict')
    predictions = np.array([i for i in model.predict(x_validate)])
    actuals = validate['close_roc'].values

    df = pd.DataFrame({'actuals': actuals, 'predictions': predictions})
    df['residuals'] = df['actuals'] - df['predictions']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    df[['actuals', 'predictions']].plot(ax=axes[0, 0])
    axes[0, 0].set_title('Do predictions closely match actual values?', fontsize=12)

    df[['actuals', 'predictions']].plot(kind='hist', bins=50, alpha=0.6, ax=axes[0, 1])
    axes[0, 1].set_title('Do distributions of predictions closely mirror actuals?', fontsize=12)

    directional_accuracy = np.sign(df['actuals']) == np.sign(df['predictions'])
    directional_accuracy.value_counts().plot.bar(ax=axes[0, 2], alpha=0.6, color=['green', 'red'])
    axes[0, 2].set_xticklabels(['Correct Direction', 'Incorrect Direction'], rotation=0)
    axes[0, 2].set_title('Does the model predict the correct direction?', fontsize=12)
    axes[0, 2].set_ylabel('Count')

    df.plot.scatter(x='predictions', y='residuals', alpha=0.6, ax=axes[1, 0])
    axes[1, 0].axhline(0, linestyle='--', color='pink')
    axes[1, 0].set_title('Are residuals randomly scattered around zero?', fontsize=12)

    osm, osr = stats.probplot(df['residuals'], dist="norm")
    axes[1, 1].scatter(osm[0], osm[1], alpha=0.6)
    axes[1, 1].plot(osm[0], osm[0]*osr[0] + osr[1], color='pink', linestyle='--')
    axes[1, 1].set_title('Do residuals follow a normal distribution?', fontsize=12)

    df.plot.scatter(x='actuals', y='predictions', alpha=0.6, ax=axes[1, 2])
    lims = [min(df.min()), max(df.max())]
    axes[1, 2].plot(lims, lims, '--', color='pink')
    axes[1, 2].set_xlim(lims)
    axes[1, 2].set_ylim(lims)
    axes[1, 2].set_title('Are predictions closely clustered around the identity line?', fontsize=12)

    plt.subplots_adjust(wspace=0.25, hspace=0.25)
    plt.show()
