from collections.abc import Mapping
from pathlib import Path

import click
import polars as pl


def export_results(uel_cfg: Mapping[str, object], experiment_log: pl.DataFrame | None, results_dir: Path) -> None:
    if uel_cfg.get('output_format', 'csv') == 'parquet' and experiment_log is not None:
        parquet_path = results_dir / 'results.parquet'
        experiment_log.write_parquet(str(parquet_path))
        click.echo(f"  Parquet → {parquet_path}")
