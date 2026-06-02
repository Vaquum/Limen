from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from limen.sfd.reference_architecture.base import ReferenceModel
from limen.yaml.compiler import CompiledSFD


@dataclass
class BarPrediction:

    '''Prediction result for a single bar.'''

    datetime: Any
    prediction: int | float | None
    probability: float | None
    reason: str | None  # None = valid prediction; 'warm-up', 'inside-training-window' = invalid


class Sensor:

    '''Inference wrapper around a trained YAML model for live bar-by-bar prediction.'''

    def __init__(self,
                 yaml_reference: dict[str, Any],
                 model: ReferenceModel,
                 fitted_params: dict[str, Any],
                 round_params: dict[str, Any],
                 permutation_id: int | None = None) -> None:

        '''
        Create a Sensor from a validated trained model.

        Args:
            yaml_reference (dict): Parsed YAML experiment dict from metadata.json
            model (ReferenceModel): Validated trained model
            fitted_params (dict): Fitted scaler/PCA state from the winning round
            round_params (dict): Full parameter dict from the winning round
            permutation_id (int | None): Round ID from the experiment log — required
                for cohort binding via Cohort.set_members

        '''

        self._yaml_reference = copy.deepcopy(yaml_reference)
        self._model = model
        self._fitted_params = dict(fitted_params)
        self._round_params = dict(round_params)
        self._manifest: Any = None
        self.permutation_id = permutation_id


    @property
    def round_params(self) -> dict[str, Any]:

        return self._round_params


    def __call__(self, data: Any) -> Any:

        return self.predict(data)


    def _get_manifest(self) -> Any:

        if self._manifest is None:
            self._manifest = CompiledSFD(self._yaml_reference).manifest()
        return self._manifest


    def predict(self, raw_klines: pl.DataFrame | dict) -> BarPrediction | dict:

        '''
        Prepare raw klines and return a prediction for the last bar.

        When called with a dict, delegates directly to the underlying model —
        compatible with the Cohort decoder interface.

        Args:
            raw_klines (pl.DataFrame | dict): Raw klines DataFrame for bar-by-bar
                prediction, or a decoder-style dict for direct model inference

        Returns:
            BarPrediction | dict: BarPrediction for DataFrame input, raw model
                prediction dict for dict input

        Raises:
            ValueError: If the window is too small, the last bar is a warm-up bar,
                or any bar falls inside the training/test window

        '''

        if isinstance(raw_klines, dict):
            return self._model.predict(raw_klines)

        manifest = self._get_manifest()
        data, indicator_lookback = manifest.sensor_input_prep(
            raw_klines, self._fitted_params, self._round_params
        )
        decoder_lookback = getattr(manifest, 'decoder_lookback', 1)
        if decoder_lookback > 1:
            raise NotImplementedError(
                'predict does not yet support decoder_lookback > 1'
            )

        self._raise_if_inside_training_window(data, manifest)

        valid_rows = len(data) - indicator_lookback
        if valid_rows < decoder_lookback:
            raise ValueError(
                f"Insufficient data: need {indicator_lookback + decoder_lookback} bars "
                f"({indicator_lookback} indicator warm-up + {decoder_lookback} decoder "
                f"window), got {len(data)}"
            )

        feature_cols = [c for c in data.columns if c != 'datetime']
        last_row = data[-1]
        if any(last_row[c][0] is None for c in feature_cols):
            raise ValueError('Last bar is a warm-up bar — cannot predict')

        x = np.array(last_row.select(feature_cols).row(0), dtype=float).reshape(1, -1)
        pred_result = self._model.predict({'x_test': x})

        dt = last_row['datetime'][0] if 'datetime' in data.columns else None
        return BarPrediction(
            datetime=dt,
            prediction=_extract_scalar(pred_result.get('_preds')),
            probability=_extract_scalar(pred_result.get('_probs')),
            reason=None,
        )


    def predict_all(self, raw_klines: pl.DataFrame) -> list[BarPrediction]:

        '''
        Prepare raw klines and return one prediction per input bar.

        Warm-up bars and bars inside the training window have prediction=None
        with reason set. Valid bars have predictions populated. Output length
        always equals len(raw_klines) for alignment with the input.

        Args:
            raw_klines (pl.DataFrame): Raw klines from live feed, same schema as
                the manifest data source

        Returns:
            list[BarPrediction]: One entry per bar in the post-bar-formation data.
                Length equals len(raw_klines) when bar_type is 'base' (no bar
                aggregation). When bar formation is active, length equals the
                aggregated bar count, which is smaller than len(raw_klines).

        '''

        manifest = self._get_manifest()
        data, indicator_lookback = manifest.sensor_input_prep(
            raw_klines, self._fitted_params, self._round_params
        )
        decoder_lookback = getattr(manifest, 'decoder_lookback', 1)

        if decoder_lookback > 1:
            raise NotImplementedError(
                'predict_all does not yet support decoder_lookback > 1'
            )

        inside_window = self._inside_training_window_mask(data, manifest)
        feature_cols = [c for c in data.columns if c != 'datetime']
        datetimes = data['datetime'].to_list() if 'datetime' in data.columns else [None] * len(data)

        results: list[BarPrediction | None] = [None] * len(data)
        valid_indices: list[int] = []

        for i in range(len(data)):
            if inside_window[i]:
                results[i] = BarPrediction(
                    datetime=datetimes[i],
                    prediction=None,
                    probability=None,
                    reason='inside-training-window',
                )
            elif i < indicator_lookback:
                results[i] = BarPrediction(
                    datetime=datetimes[i],
                    prediction=None,
                    probability=None,
                    reason='warm-up',
                )
            else:
                valid_indices.append(i)

        if valid_indices:
            x = data[valid_indices].select(feature_cols).to_numpy().astype(float)
            pred_result = self._model.predict({'x_test': x})
            preds = pred_result.get('_preds', [])
            probs = pred_result.get('_probs')

            for j, idx in enumerate(valid_indices):
                results[idx] = BarPrediction(
                    datetime=datetimes[idx],
                    prediction=_extract_scalar(preds[j]),
                    probability=_extract_scalar(probs[j]) if probs is not None else None,
                    reason=None,
                )

        return results  # type: ignore[return-value]


    def _raise_if_inside_training_window(self,
                                         data: pl.DataFrame,
                                         manifest: Any) -> None:

        if manifest.split_dates is None or 'datetime' not in data.columns:
            return
        train_start, _, _, _, _, test_end = manifest.split_dates
        inside = self._inside_training_window_mask(data, manifest)
        if any(inside):
            raise ValueError(
                f"Input data contains bars inside the training/test window "
                f"[{train_start}, {test_end}). Sensor expects data strictly "
                f"after {test_end}."
            )


    def _inside_training_window_mask(self,
                                     data: pl.DataFrame,
                                     manifest: Any) -> list[bool]:

        if manifest.split_dates is None or 'datetime' not in data.columns:
            return [False] * len(data)
        train_start, _, _, _, _, test_end = manifest.split_dates
        result = []
        for dt in data['datetime'].to_list():
            if dt is None:
                result.append(False)
                continue
            # normalise to date for comparison — split_dates stores date objects
            dt_date = dt.date() if hasattr(dt, 'date') else dt
            result.append(train_start <= dt_date < test_end)
        return result


def _extract_scalar(arr: Any) -> int | float | None:

    '''Extract a Python scalar from an array-like or from a scalar directly.'''

    if arr is None:
        return None
    try:
        if hasattr(arr, 'ndim') and arr.ndim == 0:
            return arr.item()
        val = arr[0]
        return val.item() if hasattr(val, 'item') else val
    except (IndexError, TypeError):
        return None
