import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl
from ruamel.yaml import YAML

from limen import Trainer, UniversalExperimentLoop
from limen.yaml import CompiledSFD
from limen.yaml.compiler import build_search_strategy


def completed_log(loop: UniversalExperimentLoop) -> pl.DataFrame:
    log = loop.experiment_log
    if log is None:
        raise ValueError("The experiment has no completed result log")
    return log


# region experiment
def run_research(data_dir, output_dir):
    config = YAML(typ="safe").load(Path(__file__).with_name("selection.yaml"))
    source = data_dir / "spot_15m_20250101_20250531.parquet"
    expected = "eb3392d7b4ccbd17a45ad9996e1c51d0f1b004c908a4a0264ff7efa959f1999a"
    if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
        raise ValueError("Recorded quarter-hour input digest differs")
    # Resolve the driver's public --data-dir argument into the recorded run definition.
    config["sfd"]["manifest"]["data_source"]["params"]["file_path_or_url"] = str(source.resolve())
    compiled = CompiledSFD(config)
    loop = UniversalExperimentLoop(
        sfd=compiled, search_strategy=build_search_strategy(config),
        experiment_dir=output_dir, yaml_reference=config,
    )
    loop.run("compiled-selection", n_permutations=2, workers=1,
             post_processing=True, progress_bar=False)
    selected = completed_log(loop).sort("val_backtest_total_return", descending=True)["id"].to_list()[:1]
    sensors = Trainer(output_dir).train(selected)
    return loop, sensors, selected
# endregion experiment


def main():
    parser = argparse.ArgumentParser(description="Run the declared Compiled Research pipeline on recorded Bitcoin bars.")
    import scenarios
    parser.add_argument("--scenario", choices=["selection", "all", *scenarios.SCENARIOS], default="selection")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.scenario != "selection":
        chosen = list(scenarios.SCENARIOS) if args.scenario == "all" else [args.scenario]
        reports = {name: scenarios.run(name, args.data_dir.resolve(), args.output_dir.resolve() / name) for name in chosen}
        report = {"interface": "compiled", "scenario": args.scenario, "scenarios": reports}
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False, default=str) + "\n")
        print(json.dumps({"interface": report["interface"], "scenario": args.scenario, "executed": list(reports), "summary": str((args.output_dir / "summary.json").resolve())}, indent=2))
        return
    loop, sensors, selected = run_research(args.data_dir, args.output_dir)
    report = {"interface": "compiled", "scenario": args.scenario,
              "rounds": completed_log(loop).height, "selected_rounds": len(selected),
              "reconstructed_sensors": len(sensors)}
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
