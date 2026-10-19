from __future__ import annotations

import copy
import limen.inference._fold_sensor as _fold_guard
from collections.abc import Mapping, Sequence
from limen.backtest.trade_contract import JsonValue, contract_digest
from limen.experiment._prepare_trade_context import validate_inference_contract
import logging
from datetime import datetime
from typing import Any, Literal

import numpy as np
import polars as pl

from limen.sfd.reference_architecture.base import ReferenceModel
from limen.yaml.compiler import CompiledSFD
from limen.inference._python_sensor import BarPrediction, PredictionReason, HistoryT, StateT, OutputT, Infer, MapOutput, PythonSensorBase
from limen.inference._sensor_from_model import Missing, ModelInput, ModelPairs, Prepare, Predict, identity_pairs, model_infer


__all__ = ['BarPrediction', 'PredictionReason', 'Sensor']

logger = logging.getLogger(__name__)


class Sensor:

    '''Inference wrapper around a trained YAML model for live bar-by-bar prediction.'''
    _trade_contract: Mapping[str, JsonValue] | None = None
    _trade_digest: str | None = None

    def __init__(self,
                 yaml_reference: dict[str, Any],
                 model: ReferenceModel,
                 fitted_params: dict[str, Any],
                 round_params: dict[str, Any],
                 permutation_id: str | None = None,
                 manifest_id: str | None = None, *,
                 trade_contract: Mapping[str, JsonValue] | None = None) -> None:


        super().__init__()

        _fold_guard.reject_manual(yaml_reference, self)
        self._yaml_reference = copy.deepcopy(yaml_reference)
        self._model = model
        self._fitted_params = dict(fitted_params)
        self._round_params = dict(round_params)
        self._manifest: Any = None
        self.permutation_id = permutation_id
        self.manifest_id = manifest_id
        self._trade_contract = copy.deepcopy(dict(trade_contract)) if trade_contract is not None else None
        self._trade_digest = contract_digest(self._trade_contract) if self._trade_contract is not None else None
        if self._model.prediction_mode == 'target_exposure' and self._trade_contract is None:
            raise ValueError('Signed Sensor construction requires its frozen trade contract')
        if self._trade_contract is not None:
            validate_inference_contract(self._get_manifest().resolve_trade_policy(self._round_params), self._model.prediction_mode, self._trade_contract, getattr(self._model, 'learning_binding', None))


    @property
    def trade_contract(self) -> Mapping[str, JsonValue] | None:
        return copy.deepcopy(self._trade_contract)

    @staticmethod
    def from_python(*, fitted_state: StateT, params: Mapping[str, object], infer: Infer[HistoryT, StateT, OutputT], map_output: MapOutput[OutputT], state_id: str,
                    fitting_cutoff: datetime, selection_cutoff: datetime, output_id: str | None = None) -> _PythonSensor[HistoryT, StateT, OutputT]:
        return _PythonSensor(fitted_state, params, infer, map_output, state_id, fitting_cutoff, selection_cutoff, output_id)

    @staticmethod
    def from_model(model: StateT, *, selection_cutoff: datetime, prepare: Prepare[HistoryT] | None = None,
                   feature_columns: Sequence[str] | None = None, params: Mapping[str, object] | None = None,
                   predict: Predict[StateT, OutputT] | None = None, map_output: MapOutput[OutputT] | None = None,
                   state_id: str | None = None, fitting_cutoff: datetime | None = None, output_id: str | None = None,
                   missing: Missing = 'block', model_input: ModelInput = 'auto') -> _PythonSensor[HistoryT, StateT, ModelPairs]:
        infer, identity = model_infer(model, selection_cutoff, prepare, feature_columns, predict, map_output, state_id, missing, model_input)
        return Sensor.from_python(fitted_state=model, params={} if params is None else params, infer=infer, map_output=identity_pairs, state_id=identity, fitting_cutoff=selection_cutoff if fitting_cutoff is None else fitting_cutoff, selection_cutoff=selection_cutoff, output_id=output_id)

    @property
    def prediction_mode(self) -> Literal['binary', 'target_exposure', 'python']:
        return self._model.prediction_mode

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

        '''Predict the last bar; distinguish unavailable data from flat and report per-bar failures.'''

        if hasattr(self, '_fold_context'):
            return _fold_guard.last_prediction(self, raw_klines)

        manifest = self._get_manifest()
        decoder_lookback = getattr(manifest, 'decoder_lookback', 1)
        if decoder_lookback > 1:
            raise NotImplementedError('Sensor predict does not yet support decoder_lookback > 1')

        try:
            data, indicator_lookback = manifest.sensor_input_prep(
                raw_klines, self._fitted_params, self._round_params
            )

            dt = data[-1]['datetime'][0] if len(data) and 'datetime' in data.columns else None

            if self._last_bar_inside_training_window(dt, manifest):
                return BarPrediction(datetime=dt, prediction=None, probability=None, reason='inside-training-window')

            valid_rows = len(data) - indicator_lookback
            if len(data) == 0 or valid_rows < decoder_lookback:
                return BarPrediction(datetime=dt, prediction=None, probability=None, reason='warm-up')

            feature_cols = [c for c in data.columns if c not in ('datetime', '__trade_available_at_ns__')]
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
                available_at_ns=int(last_row['__trade_available_at_ns__'][0]) if '__trade_available_at_ns__' in last_row.columns else None,
                trade_contract_digest=self._trade_digest,
            )
        # Live inference must never crash on one bar — any model/scaler/data
        # failure degrades to a sensor-error result.
        except Exception as e:  # noqa: BLE001
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

        _fold_guard.begin(self, manifest, raw_klines)
        n_fallback = len(raw_klines)
        try:
            data, indicator_lookback = manifest.sensor_input_prep(
                raw_klines, self._fitted_params, self._round_params
            )
            n_fallback = len(data)

            inside_window = self._inside_training_window_mask(data, manifest)
            feature_cols = [c for c in data.columns if c not in ('datetime', '__trade_available_at_ns__')]
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
                        available_at_ns=int(data['__trade_available_at_ns__'][idx]) if '__trade_available_at_ns__' in data.columns else None,
                        trade_contract_digest=self._trade_digest,
                    )

            return results  # type: ignore[return-value]

        # Live inference must never crash on one batch — any model/scaler/data
        # failure degrades to sensor-error results.
        except Exception as e:  # noqa: BLE001
            logger.warning('Sensor predict_all failed (permutation_id=%s): %s', self.permutation_id, e, exc_info=True)
            return _fold_guard.recover(self, manifest, e, n_fallback)


    def _last_bar_inside_training_window(self, dt: object, manifest: object) -> bool:
        return _fold_guard.last_inside(self, dt, manifest)

    def _inside_training_window_mask(self, data: pl.DataFrame, manifest: object) -> list[bool]:
        return _fold_guard.inside_mask(self, data, manifest)


class _PythonSensor(PythonSensorBase[HistoryT, StateT, OutputT], Sensor):
    '''Native Sensor using caller-owned Python state and inference.'''


def _extract_scalar(arr: Any) -> int | float | None:

    '''Extract a Python scalar from an array-like or from a scalar directly.'''

    if arr is None:
        return None
    if isinstance(arr, bool):
        return None
    if isinstance(arr, (int, float)):
        return arr
    try:
        if hasattr(arr, 'ndim') and arr.ndim == 0:
            return arr.item()
        val = arr[0]
        return val.item() if hasattr(val, 'item') else val
    except (IndexError, TypeError):
        return None
