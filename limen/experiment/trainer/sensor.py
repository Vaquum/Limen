from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any
from typing import Literal

import numpy as np
import polars as pl

from limen.sfd.reference_architecture.base import ReferenceModel
from limen.yaml.compiler import CompiledSFD


PredictionReason = Literal['warm-up', 'inside-training-window', 'null-features', 'sensor-error']


@dataclass
class BarPrediction:

    '''Prediction result for a single bar.'''

    datetime: Any
    prediction: int | float | None
    probability: float | None
    reason: PredictionReason | None


logger = logging.getLogger(__name__)


class Sensor:

    '''Inference wrapper around a trained YAML model for live bar-by-bar prediction.'''

    def __init__(self,
                 yaml_reference: dict[str, Any],
                 model: ReferenceModel,
                 fitted_params: dict[str, Any],
                 round_params: dict[str, Any],
                 permutation_id: str | None = None,
                 manifest_id: str | None = None) -> None:

        '''
        Create a Sensor from a validated trained model.

        Args:
            yaml_reference (dict): Parsed YAML experiment dict from metadata.json
            model (ReferenceModel): Validated trained model
            fitted_params (dict): Fitted scaler/PCA state from the winning round
            round_params (dict): Full parameter dict from the winning round
            permutation_id (str | None): Round ID from the experiment log — required
                for cohort binding via Cohort.set_members
            manifest_id (str | None): SHA-256 content hash of the YAML manifest,
                carried from metadata.json for traceability

        '''

        self._yaml_reference = copy.deepcopy(yaml_reference)
        self._model = model
        self._fitted_params = dict(fitted_params)
        self._round_params = dict(round_params)
        self._manifest: Any = None
        self.permutation_id = permutation_id
        self.manifest_id = manifest_id


    @property
    def round_params(self) -> dict[str, Any]:

        return self._round_params


    def __call__(self, raw_klines: pl.DataFrame) -> list[BarPrediction]:

        return self.predict_all(raw_klines)


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
            BarPrediction: Prediction for the last bar. Expected non-prediction
                conditions (warm-up, inside-training-window, null-features) are
                returned as BarPrediction with the corresponding reason rather
                than raised. Unexpected exceptions return reason='sensor-error'.

        '''

        manifest = self._get_manifest()
        decoder_lookback = getattr(manifest, 'decoder_lookback', 1)
        if decoder_lookback > 1:
            raise NotImplementedError('Sensor predict does not yet support decoder_lookback > 1')

        try:
            data, indicator_lookback = manifest.sensor_input_prep(
                raw_klines, self._fitted_params, self._round_params
            )

            if len(data) == 0:
                return BarPrediction(datetime=None, prediction=None, probability=None, reason='warm-up')

            dt = data[-1]['datetime'][0] if 'datetime' in data.columns else None

            if self._last_bar_inside_training_window(dt, manifest):
                return BarPrediction(datetime=dt, prediction=None, probability=None, reason='inside-training-window')

            valid_rows = len(data) - indicator_lookback
            if valid_rows < decoder_lookback:
                return BarPrediction(datetime=dt, prediction=None, probability=None, reason='warm-up')

            feature_cols = [c for c in data.columns if c != 'datetime']
            last_row = data[-1]
            if any(last_row[c][0] is None for c in feature_cols):
                return BarPrediction(datetime=dt, prediction=None, probability=None, reason='null-features')

            x = np.array(last_row.select(feature_cols).row(0), dtype=float).reshape(1, -1)
            pred_result = self._model.predict({'x_test': x})
            return BarPrediction(
                datetime=dt,
                prediction=_extract_scalar(pred_result.get('_preds')),
                probability=_extract_scalar(pred_result.get('_probs')),
                reason=None,
            )
        except Exception as e:
            logger.warning('Sensor predict failed (permutation_id=%s): %s', self.permutation_id, e, exc_info=True)
            return BarPrediction(datetime=None, prediction=None, probability=None, reason='sensor-error')


    def predict_all(self, raw_klines: pl.DataFrame) -> list[BarPrediction]:

        '''
        Prepare raw klines and return one prediction per input bar.

        Warm-up bars and bars inside the training window have prediction=None
        with reason set. Valid bars have predictions populated. Output length
        equals the post-bar-formation row count, not necessarily len(raw_klines).

        Args:
            raw_klines (pl.DataFrame): Raw klines from live feed, same schema as
                the manifest data source

        Returns:
            list[BarPrediction]: One entry per bar in the post-bar-formation data.
                Length equals len(raw_klines) when bar_type is 'base' (no bar
                aggregation). When bar formation is active, length equals the
                aggregated bar count, which is smaller than len(raw_klines).
                Returns reason='sensor-error' on unexpected exceptions rather
                than raising.

        '''

        manifest = self._get_manifest()
        decoder_lookback = getattr(manifest, 'decoder_lookback', 1)
        if decoder_lookback > 1:
            raise NotImplementedError('Sensor predict_all does not yet support decoder_lookback > 1')

        n_fallback = len(raw_klines)
        try:
            data, indicator_lookback = manifest.sensor_input_prep(
                raw_klines, self._fitted_params, self._round_params
            )
            n_fallback = len(data)

            inside_window = self._inside_training_window_mask(data, manifest)
            feature_cols = [c for c in data.columns if c != 'datetime']
            datetimes = data['datetime'].to_list() if 'datetime' in data.columns else [None] * len(data)

            if feature_cols:
                row_has_null = (
                    data.select(feature_cols)
                    .select(pl.any_horizontal([pl.col(c).is_null() for c in feature_cols]))
                    .to_series()
                    .to_list()
                )
            else:
                row_has_null = [False] * len(data)

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
                elif row_has_null[i]:
                    results[i] = BarPrediction(
                        datetime=datetimes[i],
                        prediction=None,
                        probability=None,
                        reason='null-features',
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

        except Exception as e:
            logger.warning('Sensor predict_all failed (permutation_id=%s): %s', self.permutation_id, e, exc_info=True)
            return [
                BarPrediction(datetime=None, prediction=None, probability=None, reason='sensor-error')
                for _ in range(n_fallback)
            ]


    def _last_bar_inside_training_window(self,
                                         dt: Any,
                                         manifest: Any) -> bool:

        if manifest.split_dates is None or dt is None:
            return False
        train_start, _, _, _, _, test_end = manifest.split_dates
        dt_date = dt.date() if hasattr(dt, 'date') else dt
        return train_start <= dt_date < test_end


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
    if isinstance(arr, (int, float)) and not isinstance(arr, bool):
        return arr
    try:
        if hasattr(arr, 'ndim') and arr.ndim == 0:
            return arr.item()
        val = arr[0]
        return val.item() if hasattr(val, 'item') else val
    except (IndexError, TypeError):
        return None
