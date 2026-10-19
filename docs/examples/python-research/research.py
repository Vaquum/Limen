import argparse
import hashlib
import json
from pathlib import Path

from types import SimpleNamespace

import numpy as np
import polars as pl
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from limen import Cohort, Sensor, UniversalExperimentLoop
from limen.experiment.param_domain import ParamDomain
from limen.experiment.param_search import GridStrategy


# region preparation
def features(bars, window):
    return bars.with_columns(
        (pl.col("close") / pl.col("close").shift(window) - 1).alias("momentum")
    )


def prepare_split(bars, round_params):
    return features(bars, round_params["window"]).with_columns(
        (pl.col("close").shift(-1) / pl.col("open").shift(-1) - 1)
        .alias("next_return")
    ).slice(12).drop_nulls(["momentum", "next_return"])


def prep(bars, round_params):
    # Split before features and labels; no target crosses a boundary.
    return {
        "train": prepare_split(bars[:960], round_params),
        "validation": prepare_split(bars[960:1440], round_params),
    }


def fit(train, round_params):
    return make_pipeline(StandardScaler(), Ridge(alpha=round_params["alpha"])).fit(
        train.select("momentum"), train["next_return"].to_numpy()
    )


def backtest(validation, forecasts):
    positions = np.sign(forecasts)
    net = positions * validation["next_return"].to_numpy() - abs(positions) * 0.0002
    return float(np.mean(net))


def prepare_live(history, params):
    return features(history, params["window"]).with_columns(
        (pl.col("datetime").dt.epoch("ns") + 900_000_000_000).alias("available_at_ns"),
    )


# endregion preparation


# region experiment
def run_research(bars, experiment_name):
    research = bars[:1440]
    space = {"window": [4, 12], "alpha": [0.1, 1.0]}
    retained = {}

    def model(data, round_params):
        fitted = fit(data["train"], round_params)
        forecasts = fitted.predict(data["validation"].select("momentum"))
        score = backtest(data["validation"], forecasts)
        retained[(round_params["window"], round_params["alpha"])] = fitted
        return {"validation_net_return": score}

    sfd = SimpleNamespace(params=lambda: space, prep=prep, model=model)
    uel = UniversalExperimentLoop(
        data=research, sfd=sfd, search_strategy=GridStrategy(ParamDomain(space))
    )
    # An absolute filename stem writes only this CSV; no persisted round artifacts.
    uel.run(experiment_name, n_permutations=4, workers=1,
            post_processing=False, progress_bar=False)
    selected = (
        uel.experiment_log.sort(
            ["window", "validation_net_return", "alpha"],
            descending=[False, True, False],
        ).unique(subset="window", keep="first", maintain_order=True)
        .select("window", "alpha").to_dicts()
    )
    sensors = [
        Sensor.from_model(retained[(p["window"], p["alpha"])], prepare=prepare_live,
                          params=p, fitting_cutoff=bars["datetime"][960],
                          selection_cutoff=bars["datetime"][1440], output_id="next_bar_return")
        for p in selected
    ]
    cohort = Cohort.from_python(sensors)
    return cohort, uel, selected, retained


# endregion experiment


def main():
    parser = argparse.ArgumentParser(description="Run caller-owned Python research on recorded Bitcoin bars.")
    import scenarios
    parser.add_argument("--scenario", choices=["selection", "all", *scenarios.SCENARIOS], default="selection")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.scenario != "selection":
        chosen = list(scenarios.SCENARIOS) if args.scenario == "all" else [args.scenario]
        reports = {name: scenarios.run(name, args.data_dir.resolve(), args.output_dir.resolve() / name) for name in chosen}
        report = {"interface": "python", "scenario": args.scenario, "scenarios": reports}
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False, default=str) + "\n")
        print(json.dumps(report, indent=2, allow_nan=False, default=str))
        return
    source = args.data_dir / "spot_15m_20250101_20250531.parquet"
    expected = "eb3392d7b4ccbd17a45ad9996e1c51d0f1b004c908a4a0264ff7efa959f1999a"
    if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
        raise ValueError("Recorded quarter-hour input digest differs")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bars = pl.read_parquet(source, n_rows=1920)
    cohort, uel, selected, retained = run_research(bars, str((args.output_dir / "research").resolve()))
    predictions = cohort.predict_all(bars[1428:1920])[12:]
    report = {
        "interface": "python", "scenario": args.scenario,
        "rounds": uel.experiment_log.height, "selected": selected,
        "retained_models": len(retained), "inference_rows": len(predictions),
        "last_prediction": predictions[-1].prediction,
        "fitting_cutoff": bars["datetime"][960].isoformat(),
        "selection_cutoff": bars["datetime"][1440].isoformat(),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
