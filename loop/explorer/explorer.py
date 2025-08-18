from loop.explorer.loop_explorer import loop_explorer

import random


class Explorer:

    '''Visual data exploration for the Universal Experiment Loop'''

    def __init__(self, uel):
        
        '''
        Initialize the Explorer object.

        uel (UniversalExperimentLoop): Universal Experiment Loop object

        Returns:
            None
        '''

        self._port = random.randint(5001, 5500)
        self._uel = uel

    def input_data(self):
        
        '''
        Visualize the input data.

        Returns:
            None
        '''

        loop_explorer(data=self._uel.data, port=self._port)

    def experiment_log(self):
        
        '''
        Visualize the experiment log.

        Returns:
            None
        '''

        loop_explorer(data=self._uel.experiment_log)

    def experiment_parameter_correlation(self, x):

        '''
        Visualize the experiment parameter correlation.

        x (str): The column to visualize the correlation of.

        Returns:
            None
        '''

        loop_explorer(data=self._uel.experiment_parameter_correlation(x))

    def experiment_confusion_metrics(self):

        '''
        Visualize the confusion metrics.

        Returns:
            None
        '''
    
        loop_explorer(data=self._uel.experiment_confusion_metrics)

    def experimentbacktest_results(self):
        
        '''
        Visualize the backtest results.

        Returns:
            None
        '''

        loop_explorer(data=self._uel.experiment_backtest_results)
