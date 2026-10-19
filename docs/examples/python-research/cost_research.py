# region observed_cost_model
import json
from pathlib import Path
from types import SimpleNamespace

import polars as pl

from limen.backtest.observed_costs import CostCalibration, CostCandidate, CostConditions
from limen.backtest.trade_contract import (
    ProductSpec, TradeInputs, TradePolicy, source_binding,
)
from limen.experiment import UniversalExperimentLoop
from limen.experiment.param_domain import ParamDomain
from limen.experiment.param_search.grid_strategy import GridStrategy


class CallerModel:
    def fit(self, training):
        self.mean = training["price"].mean()
        return self

    def predict(self, evaluation, mode, scale):
        exposure = [1.0 if price >= self.mean else
                    (0.0 if mode == "binary" else -scale)
                    for price in evaluation["price"]]
        exposure[-1] = 0.0
        return pl.Series("target", exposure, dtype=pl.Float64)


def run_cost_experiment(calibration_path, specification_path, benchmark_path,
                        output, *, mode="binary", hour=1):
    expected = json.loads(Path(specification_path).read_text())["calibration_digest"]
    calibration = CostCalibration.read(calibration_path, expected_digest=expected)
    benchmark = pl.DataFrame(json.loads(Path(benchmark_path).read_text()), schema={
        "row_id": pl.String, "phase": pl.String, "time_ns": pl.Int64,
        "replay_available_at_ns": pl.Int64, "price": pl.Float64,
        "reference_id": pl.String, "observed_receipt_ns": pl.Int64,
    })
    first = benchmark["time_ns"][0] + (hour - 1) * 3_600_000_000_000
    recorded = benchmark.filter(pl.col("time_ns").is_between(
        first, first + 420_000_000_000))
    output = Path(output)
    output.mkdir()
    comparisons = []
    space = {"scale": [0.5]}

    def prep(frame, round_params):
        return {"training": frame.head(3), "evaluation": frame.tail(5)}

    def model(data, round_params):
        evaluation = data["evaluation"]
        fitted = CallerModel().fit(data["training"])
        predictions = fitted.predict(evaluation, mode, round_params["scale"])
        signals = evaluation.select(
            "row_id", pl.col("time_ns").alias("available_at_ns")
        ).with_columns(predictions)
        observations = evaluation.select(
            "row_id", pl.col("time_ns").alias("start_ns"),
            pl.col("time_ns").alias("end_ns"),
            pl.col("replay_available_at_ns").alias("open_available_at_ns"),
            pl.col("replay_available_at_ns").alias("available_at_ns"),
            *(pl.col("price").alias(name) for name in ("open", "high", "low", "close")),
        )
        start, end = evaluation["time_ns"][0], evaluation["time_ns"][-1]
        binding = source_binding(observations, "recorded_midpoints", start, end,
                                 1, "retrospective_assumed_exchange_label")
        inputs = TradeInputs(10_000.0, start, end, signals, observations, None, (binding,))
        policy = TradePolicy(
            ProductSpec("linear_perpetual", "BTC", "BTC", "USDC", 0.00001, 1.0),
            prediction_mode=mode, fee_bps=5.0, slip_bps=5.0,
        )
        conditions: CostConditions = {
            "venue": "hyperliquid", "environment": "mainnet",
            "product_kind": "linear_perpetual", "instrument": "BTC",
            "base_currency": "BTC", "quote_currency": "USDC",
            "reference_id": benchmark["reference_id"][0],
            "reference_convention": "minute_midpoint_label",
            "clock": "assumed_exchange_label",
        }
        candidate = CostCandidate(inputs, policy, f"caller:experiment-{hour}",
                                  conditions, benchmark)
        comparison = calibration.compare({"caller": candidate})
        comparison.write(output / "comparison.json")
        comparisons.append(comparison)
        return {
            "original_return": comparison.ledger("caller", "original").metrics["total_return"],
            "calibrated_return": comparison.ledger("caller", "calibrated").metrics["total_return"],
        }

    sfd = SimpleNamespace(params=lambda: space, prep=prep, model=model)
    experiment = UniversalExperimentLoop(
        data=recorded, sfd=sfd, search_strategy=GridStrategy(ParamDomain(space)),
    )
    experiment.run(str(output / "experiment"), n_permutations=1, workers=1,
                   post_processing=False, progress_bar=False)
    return experiment, comparisons[0]
# endregion observed_cost_model
