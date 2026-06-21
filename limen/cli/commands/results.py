import json
from pathlib import Path

import click

from limen.yaml.store import short_id

_PREFERRED_METRICS = ['val_score', 'auc', 'accuracy', 'balanced_metric', 'backtest_pnl_per_bar_bps']
_NON_METRIC_COLUMNS = {'id', '_id', '_warnings', '_round_index', '_injected',
                       '_generation_index', '_search_strategy', 'execution_time',
                       'strict_mode_error'}


def run_results(results_dir: Path,
                metric: str | None,
                top: int,
                ascending: bool,
                as_json: bool) -> bool:

    '''
    Summarize a finished run and rank its permutations by a metric.

    Args:
        results_dir (Path): A run directory containing results.csv
        metric (str | None): Column to rank by; a sensible default is chosen if None
        top (int): Number of best permutations to show
        ascending (bool): Rank ascending (for metrics where lower is better)
        as_json (bool): Emit the ranked rows as JSON instead of a table

    Returns:
        bool: True on success, False on failure

    '''

    csv_path = results_dir / 'results.csv'
    if not csv_path.exists():
        click.secho(f"  ✗ No results.csv in '{results_dir}'.", fg='red')
        return False

    import polars as pl

    df = pl.read_csv(csv_path)
    if df.height == 0:
        click.secho('  ✗ results.csv is empty.', fg='red')
        return False

    metadata = _load_metadata(results_dir)
    swept = [p for p in _swept_params(metadata) if p in df.columns]
    manifest_id = metadata.get('manifest_id')

    if metric is None:
        metric = _default_metric(df)
        if metric is None:
            click.secho(
                '  ✗ Specify --metric. Available: ' + ', '.join(_metric_columns(df, swept)),
                fg='red',
            )
            return False
    if metric not in df.columns:
        click.secho(
            f"  ✗ Metric '{metric}' not found. Available: " + ', '.join(_metric_columns(df, swept)),
            fg='red',
        )
        return False

    ranked = df.sort(metric, descending=not ascending, nulls_last=True).head(top)
    display_cols = ['id', metric, *swept]
    rows = ranked.select([c for c in display_cols if c in ranked.columns]).to_dicts()

    if as_json:
        payload = {
            'manifest_id': manifest_id,
            'metric': metric,
            'permutations': df.height,
            'results': rows,
        }
        click.echo(json.dumps(payload, indent=2, default=str))
        return True

    direction = 'lowest' if ascending else 'highest'
    manifest_note = f"   (manifest sha256:{short_id(manifest_id)})" if isinstance(manifest_id, str) else ''
    click.echo(f"{df.height} permutations — top {len(rows)} by {direction} {metric}{manifest_note}\n")
    for rank, row in enumerate(rows, start=1):
        short = str(row.get('id', '?'))[:8]
        value = row.get(metric)
        params = '  '.join(f"{p}={row[p]}" for p in swept if p in row)
        click.echo(f"  {rank}. {short}  {metric}={value}   {params}")

    return True


def _load_metadata(results_dir: Path) -> dict:

    metadata_path = results_dir / 'metadata.json'
    if not metadata_path.exists():
        return {}
    try:
        return json.loads(metadata_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return {}


def _swept_params(metadata: dict) -> list[str]:

    params = metadata.get('yaml_reference', {}).get('sfd', {}).get('params', {})
    if not isinstance(params, dict):
        return []
    return [k for k, v in params.items() if isinstance(v, list) and len(v) > 1]


def _metric_columns(df: 'object', swept: list[str]) -> list[str]:

    return [c for c in df.columns if c not in _NON_METRIC_COLUMNS and c not in swept]


def _default_metric(df: 'object') -> str | None:

    for candidate in _PREFERRED_METRICS:
        if candidate in df.columns and df[candidate].drop_nulls().len() > 0:
            return candidate
    return None
