from typing import Any


class Sensor:

    '''Callable wrapper around a trained model for live inference.'''

    def __init__(self,
                 model: Any,
                 permutation_id: int,
                 round_params: dict[str, Any],
                 metadata: dict[str, Any],
                 results: dict[str, Any] | None = None) -> None:

        '''
        Create a Sensor from a trained model and experiment context.

        Args:
            model (Any): Trained ReferenceModel instance
            permutation_id (int): Round ID from experiment log
            round_params (dict[str, Any]): Parameter values used for this permutation
            metadata (dict[str, Any]): Experiment metadata from metadata.json
            results (dict[str, Any] | None): Model evaluation results from Pass 1

        '''

        self._model = model
        self._permutation_id = permutation_id
        self._round_params = dict(round_params)
        self._metadata = dict(metadata)
        self._results = dict(results) if results is not None else None


    @property
    def model(self) -> Any:

        return self._model


    @property
    def permutation_id(self) -> int:

        return self._permutation_id


    @property
    def round_params(self) -> dict[str, Any]:

        return dict(self._round_params)


    @property
    def metadata(self) -> dict[str, Any]:

        return dict(self._metadata)


    @property
    def results(self) -> dict[str, Any] | None:

        return dict(self._results) if self._results is not None else None


    def predict(self, data: dict) -> dict:

        '''
        Compute predictions from feature data.

        Args:
            data (dict): Data dictionary with x_test

        Returns:
            dict: Prediction results with '_preds' key

        Raises:
            ValueError: If no trained model is available

        '''

        if self._model is None:
            raise ValueError('Sensor has no trained model.')

        return self._model.predict(data)


    def __call__(self, data: dict) -> dict:

        return self.predict(data)
