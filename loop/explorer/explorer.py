from loop.explorer.loop_explorer import loop_explorer

from typing import TYPE_CHECKING
import random

if TYPE_CHECKING:
    from loop.universal_experiment_loop import UniversalExperimentLoop


class Explorer:

    '''Render visual data exploration interface for the Universal Experiment Loop.'''

    def __init__(self, uel: 'UniversalExperimentLoop') -> None:
        
        '''
        Initialize the Explorer object.

        Args:
            uel (UniversalExperimentLoop): Universal Experiment Loop object

        Returns:
            None: None
        '''

        self._uel = uel

    def input_data(self) -> None:
        
        '''
        Render the input dataset in the Explorer.

        Args:
            
        Returns:
            None: None
        '''

        loop_explorer(data=self._uel.data)

    def experiment_log(self) -> None:
        
        '''
        Render the experiment log in the Explorer.

        Args:
            
        Returns:
            None: None
        '''

        loop_explorer(data=self._uel.experiment_log)

    def experiment_parameter_correlation(self, x: str) -> None:

        '''
        Render the experiment parameter correlation view.

        Args:
            x (str): Column name used for correlation analysis

        Returns:
            None: None
        '''

        loop_explorer(data=self._uel.experiment_parameter_correlation(x))

    def experiment_confusion_metrics(self) -> None:

        '''
        Render the confusion metrics in the Explorer.

        Args:
            
        Returns:
            None: None
        '''
    
        loop_explorer(data=self._uel.experiment_confusion_metrics)

    def experiment_backtest_results(self) -> None:
        
        '''
        Render the backtest results in the Explorer.

        Args:
            
        Returns:
            None: None
        '''

        loop_explorer(data=self._uel.experiment_backtest_results)
