from typing import Any


class Sensor:

    '''Callable wrapper around a trained model for live inference.'''

    def __init__(self,
                 model: Any,
                 round_params: dict[str, Any],
                 metadata: dict[str, Any],
                 results: dict[str, Any] | None = None) -> None:

        '''
        Create a Sensor from a trained model and experiment context.

        Args:
            model (Any): Trained ReferenceModel instance, or None for Pass 1
            round_params (dict[str, Any]): Parameter values used for this permutation
            metadata (dict[str, Any]): Experiment metadata from metadata.json
            results (dict[str, Any] | None): Model evaluation results from Pass 1

        '''

        self._model = model
        self._round_params = dict(round_params)
        self._metadata = dict(metadata)
        self._results = dict(results) if results is not None else None


    @property
    def model(self) -> Any:

        return self._model


    @property
    def round_params(self) -> dict[str, Any]:

        return dict(self._round_params)


    @property
    def metadata(self) -> dict[str, Any]:

        return dict(self._metadata)


    @property
    def results(self) -> dict[str, Any] | None:

        return dict(self._results) if self._results is not None else None


    def __call__(self, data: dict) -> dict:

        '''
        Run inference using the trained model.

        Args:
            data (dict): Data dictionary with feature arrays

        Returns:
            dict: Prediction results

        Raises:
            ValueError: If no trained model is available

        '''

        if self._model is None:
            raise ValueError(
                'Sensor has no trained model. '
                'Pass 1 sensors store results only. '
                'Use Pass 2 training to create callable sensors.'
            )

        return self._model.predict(data)
