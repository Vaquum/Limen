from loop.explorer.loop_explorer import loop_explorer

import random


class Explore:

    def __init__(self, uel):

        self._port = random.randint(5001, 5500)
        self._uel = uel

    def input_data(self):
    
        loop_explorer(data=self._uel.data, port=self._port)
    
    def experiment_log(self):
    
        loop_explorer(data=self._uel.log_df, port=self._port)

    def feature_correlation(self):
    
        loop_explorer(data=self._uel.feature_correlation('auc'), port=self._port)

    def confusion_metrics(self):
    
        loop_explorer(data=self._uel.confusion_metrics('price_change'), port=self._port)

    def backtest_results(self):
    
        loop_explorer(data=self._uel.backtest_results(), port=self._port)