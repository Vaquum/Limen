from __future__ import annotations

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
                 yaml_reference: dict,
                 model: ReferenceModel,
                 fitted_params: dict,
                 round_params: dict) -> None:

        '''
        Create a Sensor from a validated Pass 1 model.

        Args:
            yaml_reference (dict): Parsed YAML experiment dict from metadata.json
            model (ReferenceModel): Trained model from Pass 1
            fitted_params (dict): Fitted scaler/PCA state from the winning round
            round_params (dict): Full parameter dict from the winning round

        '''

        self._yaml_reference = dict(yaml_reference)
        self._model = model
        self._fitted_params = dict(fitted_params)
        self._round_params = dict(round_params)
        self._manifest: Any = None


    def _get_manifest(self) -> Any:

        if self._manifest is None:
            self._manifest = CompiledSFD(self._yaml_reference).manifest()
        return self._manifest


    def predict(self, raw_klines: pl.DataFrame) -> BarPrediction:

        '''
        Prepare raw klines and return a prediction for the last bar.

        Args:
            raw_klines (pl.DataFrame): Raw klines from live feed, same schema as
                the manifest data source

        Returns:
            BarPrediction: Prediction for the last bar with reason=None

        Raises:
            ValueError: If the window is too small, the last bar is a warm-up bar,
                or any bar falls inside the training/test window

        '''

        manifest = self._get_manifest()
        data, indicator_lookback = manifest.sensor_input_prep(
            raw_klines, self._fitted_params, self._round_params
        )
        decoder_lookback = getattr(manifest, 'decoder_lookback', 1)

        self._raise_if_inside_training_window(data, manifest)

        valid_rows = len(data) - indicator_lookback
        if valid_rows < decoder_lookback:
            raise ValueError(
                f"Insufficient data: need {indicator_lookback + decoder_lookback} bars "
                f"({indicator_lookback} indicator warm-up + {decoder_lookback} decoder "
                f"window), got {len(data)}"
            )

        last_row = data[-1]
        if last_row.null_count() > 0:
            raise ValueError('Last bar is a warm-up bar — cannot predict')

        feature_cols = [c for c in data.columns if c != 'datetime']
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
            list[BarPrediction]: One entry per input bar

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

        results: list[BarPrediction | None] = [None] * len(data)
        valid_indices: list[int] = []

        for i in range(len(data)):
            dt = data[i]['datetime'][0] if 'datetime' in data.columns else None
            if inside_window[i]:
                results[i] = BarPrediction(
                    datetime=dt,
                    prediction=None,
                    probability=None,
                    reason='inside-training-window',
                )
            elif i < indicator_lookback:
                results[i] = BarPrediction(
                    datetime=dt,
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
                dt = data[idx]['datetime'][0] if 'datetime' in data.columns else None
                results[idx] = BarPrediction(
                    datetime=dt,
                    prediction=_extract_scalar([preds[j]]),
                    probability=_extract_scalar([probs[j]]) if probs is not None else None,
                    reason=None,
                )

        return results  # type: ignore[return-value]


    def _raise_if_inside_training_window(self,
                                         data: pl.DataFrame,
                                         manifest: Any) -> None:

        if manifest.split_dates is None or 'datetime' not in data.columns:
            return
        train_start, _, _, _, _, test_end = manifest.split_dates
        inside = [
            dt is not None and train_start <= dt < test_end
            for dt in data['datetime'].to_list()
        ]
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
        return [
            dt is not None and train_start <= dt < test_end
            for dt in data['datetime'].to_list()
        ]


def _extract_scalar(arr: Any) -> int | float | None:

    '''Extract a Python scalar from the first element of an array-like.'''

    if arr is None:
        return None
    try:
        val = arr[0]
        return val.item() if hasattr(val, 'item') else val
    except (IndexError, TypeError):
        return None
