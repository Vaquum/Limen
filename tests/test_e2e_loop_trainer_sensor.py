'''
End-to-end integration tests: experiment loop → Trainer → Sensor inference.

These tests use real HistoricalData API calls (network required) and cover the
full pipeline from YAML experiment to trained sensors with live bar prediction.
'''
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from textwrap import dedent
from unittest.mock import patch

import polars as pl

from limen.cli.commands.run import run_experiment
from limen.data import historical_data
from limen.experiment.trainer import Trainer
from limen.experiment.trainer.sensor import BarPrediction


# Adapted from logreg_label_control.yaml:
#   - split_config replaced with explicit split_dates
#   - start_date_limit/end_date_limit removed from data_source.params (blocked by validator)
#   - mode: development for testing
#   - n_permutations reduced to 10
_LABEL_CONTROL_YAML = dedent('''\
    schema_version: "1.0"
    metadata:
      name: btc_logreg_15m_random_ctrl
      limen_version: "3.8.0"
      mode: development
      description: Random-label control (noise floor) for the 15m logreg pipeline
    sfd:
      manifest:
        type: ml
        data_source:
          method: limen.data.HistoricalData.get_spot_klines
          params:
            kline_size: 900
        split_dates:
          train_start: "2025-01-01"
          train_end: "2025-04-30"
          val_start: "2025-04-30"
          val_end: "2025-05-14"
          test_start: "2025-05-14"
          test_end: "2025-05-28"
        features:
          - func: limen.features.fractional_diff
            params:
              d: "{frac_diff_d}"
              cols: [close]
              threshold: 0.001
          - func: limen.features.log_returns.log_returns
            params:
              group: momentum
          - func: limen.indicators.roc
            params:
              period: "{roc_period}"
              group: momentum
          - func: limen.indicators.ppo
            params:
              group: momentum
          - func: limen.indicators.wilder_rsi
            params:
              period: "{rsi_period}"
              group: momentum
          - func: limen.indicators.cci
            params:
              group: momentum
          - func: limen.indicators.willr
            params:
              group: momentum
          - func: limen.indicators.mom
            params:
              group: momentum
          - func: limen.indicators.atr
            params:
              period: "{atr_period}"
              group: volatility
          - func: limen.indicators.natr
            params:
              group: volatility
          - func: limen.indicators.bbands
            params:
              period: "{bbands_period}"
              group: volatility
          - func: limen.indicators.stddev
            params:
              group: volatility
          - func: limen.indicators.rolling_volatility
            params:
              group: volatility
          - func: limen.indicators.obv
            params:
              group: volume
          - func: limen.indicators.mfi
            params:
              group: volume
          - func: limen.features.vwap
            params:
              group: volume
          - func: limen.features.kline_imbalance
            params:
              group: volume
          - func: limen.features.dollar_volume
            params:
              group: volume
          - func: limen.features.volume_regime
            params:
              group: volume
          - func: limen.indicators.ema
            params:
              period: "{ema_period}"
              group: trend
          - func: limen.indicators.sma
            params:
              group: trend
          - func: limen.features.trend_strength
            params:
              group: trend
          - func: limen.features.range_pct
            params:
              group: microstructure
          - func: limen.features.body_to_range
            params:
              group: microstructure
          - func: limen.features.close_position
            params:
              group: microstructure
        target:
          name: random_binary
          class: limen.targets.RandomBinaryTarget
          fit_params:
            random_state: "{random_state}"
        scaler:
          from_params: scaler_type
        feature_ablation:
          drop_count_key: feature_drop_count
          seed_key: feature_drop_seed
        pca_compression: {}
        reference_architecture: limen.sfd.reference_architecture.logreg_binary
        calibration:
          probability_calibration:
            func: limen.calibration.sklearn_probability_calibrator
            params:
              method: "{cal_method}"
          threshold_function:
            func: limen.calibration.grid_threshold_optimizer
            params:
              metric: limen.metrics.balanced_metric.balanced_metric
              threshold_min: "{threshold_min}"
              threshold_max: "{threshold_max}"
              threshold_step: "{threshold_step}"
      params:
        frac_diff_d: [0.0, 0.1, 0.2, 0.3, 0.4]
        roc_period: [1, 4, 8, 16, 48, 96]
        atr_period: [7, 14, 28]
        rsi_period: [7, 14, 28]
        ema_period: [12, 26, 96]
        bbands_period: [20, 48, 96]
        scaler_type: [robust]
        feature_groups: [all, momentum, volatility, volume, "momentum|volatility", "momentum|volume"]
        feature_drop_count: [1, 2, 3, 4, 5]
        feature_drop_seed: [42]
        auto_pca: [true, false]
        pca_k: [15, 16, 17, 18]
        use_calibration: [true, false]
        use_threshold: [true, false]
        cal_method: [isotonic, sigmoid]
        threshold_min: [0.20, 0.30, 0.40]
        threshold_max: [0.55, 0.65, 0.75]
        threshold_step: [0.02, 0.05]
        solver: [lbfgs, liblinear, saga]
        penalty: [l2]
        dual: [false]
        tol: [0.0001, 0.001, 0.01]
        C: [0.01, 0.05, 0.1, 0.5, 1.0, 3.0, 10.0]
        fit_intercept: [true, false]
        intercept_scaling: [1.0]
        class_weight: [0.40, 0.50, 0.60, 0.75, 0.90]
        random_state: [42]
        max_iter: [60, 120, 240, 500]
        multi_class: [deprecated]
        verbose: [0]
        warm_start: [false]
        n_jobs: [-1]
        l1_ratio: [null]
    uel:
      n_permutations: 10
      search_strategy:
        type: random
      prep_each_round: true
      output_format: csv
''')

_N_PERMUTATIONS = 10
_TOP_N = 3
_INFERENCE_START = '2025-05-29'
_INFERENCE_END = '2025-05-31'


def test_e2e_label_control_trainer_sensor() -> None:

    with TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / 'label_control'
        exp_dir.mkdir()

        # ── Experiment ────────────────────────────────────────────────────────

        with TemporaryDirectory() as yaml_dir, patch('click.echo'), patch('click.secho'):
            yaml_path = Path(yaml_dir) / 'label_control.yaml'
            yaml_path.write_text(_LABEL_CONTROL_YAML.replace(
                'output_format: csv',
                f'output_format: csv\n  output_path: "{exp_dir}"',
            ))
            ok = run_experiment(yaml_path)

        assert ok
        assert (exp_dir / 'results.csv').exists()
        assert (exp_dir / 'round_data.jsonl').exists()
        assert (exp_dir / 'metadata.json').exists()

        results = pl.read_csv(exp_dir / 'results.csv')
        assert len(results) == _N_PERMUTATIONS
        assert results['id'].n_unique() == _N_PERMUTATIONS
        assert 'accuracy' in results.columns
        assert 'auc' in results.columns

        metadata = json.loads((exp_dir / 'metadata.json').read_text())
        assert 'yaml_reference' in metadata
        assert 'limen_version' in metadata
        assert metadata['yaml_reference']['metadata']['name'] == 'btc_logreg_15m_random_ctrl'

        round_ids: list[int] = []
        with (exp_dir / 'round_data.jsonl').open() as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    round_ids.append(json.loads(stripped)['round_id'])
        assert len(round_ids) == _N_PERMUTATIONS

        # ── Trainer ───────────────────────────────────────────────────────────

        top_ids = results.sort('accuracy', descending=True).head(_TOP_N)['id'].to_list()

        trainer = Trainer(exp_dir)
        sensors = trainer.train(top_ids)
        assert len(sensors) == _TOP_N

        for sensor in sensors:
            assert sensor.permutation_id in top_ids
            rp = sensor.round_params
            assert '_dropped_features' in rp
            assert isinstance(rp['_dropped_features'], list)
            assert 'feature_drop_count' in rp
            assert 'feature_drop_seed' in rp
            assert 'auto_pca' in rp

        # ── Inference data ────────────────────────────────────────────────────

        inference_klines = historical_data.HistoricalData().get_spot_klines(
            kline_size=900,
            start_date_limit=_INFERENCE_START,
            end_date_limit=_INFERENCE_END,
        )
        assert len(inference_klines) > 50
        kline_dts = inference_klines['datetime'].to_list()

        # ── Sensor inference — three windows per sensor ───────────────────────

        for sensor in sensors:

            # ── Full window ───────────────────────────────────────────────────

            all_preds = sensor.predict_all(inference_klines)
            assert len(all_preds) == len(inference_klines)

            n_warmup = sum(1 for p in all_preds if p.reason == 'warm-up')
            n_inside = sum(1 for p in all_preds if p.reason == 'inside-training-window')
            n_valid  = sum(1 for p in all_preds if p.reason is None)

            assert n_inside == 0
            assert n_valid > 0
            assert n_warmup + n_valid == len(inference_klines)
            assert all(p.datetime == dt for p, dt in zip(all_preds, kline_dts))

            bar_pred = sensor.predict(inference_klines)
            assert isinstance(bar_pred, BarPrediction)
            assert bar_pred.reason is None
            assert bar_pred.prediction in (0, 1)
            assert bar_pred.probability is not None
            assert 0.0 <= bar_pred.probability <= 1.0
            assert bar_pred.datetime == kline_dts[-1]
            last_valid = next(p for p in reversed(all_preds) if p.reason is None)
            assert last_valid.prediction == bar_pred.prediction

            assert n_warmup + 5 <= len(inference_klines), (
                f'inference window too small: {len(inference_klines)} bars, n_warmup={n_warmup}'
            )

            # ── Minimum window (n_warmup + 1 bars) ───────────────────────────

            min_klines = inference_klines[:n_warmup + 1]
            min_dts = min_klines['datetime'].to_list()

            min_preds = sensor.predict_all(min_klines)
            assert len(min_preds) == n_warmup + 1
            assert sum(1 for p in min_preds if p.reason == 'warm-up') == n_warmup
            assert sum(1 for p in min_preds if p.reason is None) == 1
            assert all(p.datetime == dt for p, dt in zip(min_preds, min_dts))

            min_bar_pred = sensor.predict(min_klines)
            assert min_bar_pred.reason is None
            assert min_bar_pred.prediction in (0, 1)
            assert min_bar_pred.datetime == min_dts[-1]
            assert min_bar_pred.prediction == min_preds[-1].prediction

            # ── Few bars window (n_warmup + 5 bars) ──────────────────────────

            few_klines = inference_klines[:n_warmup + 5]
            few_dts = few_klines['datetime'].to_list()

            few_preds = sensor.predict_all(few_klines)
            assert len(few_preds) == n_warmup + 5
            assert sum(1 for p in few_preds if p.reason == 'warm-up') == n_warmup
            assert sum(1 for p in few_preds if p.reason is None) == 5
            assert all(p.datetime == dt for p, dt in zip(few_preds, few_dts))

            few_bar_pred = sensor.predict(few_klines)
            assert few_bar_pred.reason is None
            assert few_bar_pred.prediction in (0, 1)
            assert few_bar_pred.datetime == few_dts[-1]
            assert few_bar_pred.prediction == few_preds[-1].prediction
