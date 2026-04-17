'''CLI entry point for running a Loop payload through UEL.

Invocation:
    python -m limen.sfd.loop.run <payload.json> --out <experiment_dir> --n <N>

The runner owns the experiment directory: it creates the dir, copies the
payload there as an audit trail, then hands the dir to UEL which writes all
its standard artifacts (results.csv, checkpoint.json, audit.jsonl, etc.).
The progress callback writes progress.json on every round so a polling
backend can serve live updates.

NOTE: This module is part of the temporary `limen.sfd.loop` subpackage that
will be removed when RFC-1005 (YAML compiler) lands.
'''

import argparse
import inspect
import json
import logging
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import polars as pl

from limen import UniversalExperimentLoop
from limen.data import HistoricalData
from limen.experiment.param_domain import ParamDomain
from limen.experiment.param_search import RandomStrategy
from limen.sfd.loop.loop_sfd import LoopSFD
from limen.sfd.loop.meta import DATA_SOURCE_REGISTRY
from limen.sfd.loop.progress import make_progress_callback


logger = logging.getLogger(__name__)


_DATA_API_AUTH_TOKEN_ENV = 'DATA_API_AUTH_TOKEN'  # noqa: S105 — env var name, not a secret
_LOOP_ENV = 'LOOP_ENV'


def _annotation_accepts_int(annotation: Any) -> bool:

    '''
    Compute whether a type annotation accepts `int` values.

    Handles plain `int`, `int | None`, `Optional[int]`, any Union containing
    int, and string annotations produced by `from __future__ import annotations`
    (e.g. `'int'`, `'int | None'`).

    Args:
        annotation: The annotation object from inspect.Parameter.annotation

    Returns:
        bool: True if int is a valid value for this annotation

    '''

    if annotation is int:
        return True
    if isinstance(annotation, str):
        return 'int' in [part.strip() for part in annotation.split('|')]
    args = getattr(annotation, '__args__', None)
    if args is None:
        return False
    return int in args


def _coerce_string_params_by_signature(method: Callable,
                                       params: dict[str, Any]) -> dict[str, Any]:

    '''
    Compute a params dict with string values coerced to int when the method
    signature annotates the parameter as int.

    Loop web-form inputs come through as strings (e.g. '3600'), but
    HistoricalData methods expect typed values (e.g. kline_size: int). Coerce
    at the edge so the rest of the stack sees correct types, and fail loudly
    with a clear message when a string value cannot be converted.

    Args:
        method (Callable): The HistoricalData method about to be called.
            Its annotations drive the coercion
        params (dict): Raw params as pulled from the payload

    Returns:
        dict: Same keys as params, with string values coerced to int where
            appropriate

    Raises:
        ValueError: When a string value cannot be converted to the annotated
            type. The message includes the param name and offending value

    '''

    try:
        sig = inspect.signature(method)
    except (TypeError, ValueError):
        return dict(params)

    coerced: dict[str, Any] = {}
    for pname, pvalue in params.items():
        if not isinstance(pvalue, str):
            coerced[pname] = pvalue
            continue
        param_sig = sig.parameters.get(pname)
        if param_sig is None:
            coerced[pname] = pvalue
            continue
        if not _annotation_accepts_int(param_sig.annotation):
            coerced[pname] = pvalue
            continue
        try:
            coerced[pname] = int(pvalue)
        except ValueError as exc:
            raise ValueError(
                f"Cannot coerce data source param {pname}={pvalue!r} to int "
                f"(method {method.__name__} expects "
                f"{param_sig.annotation}): {exc}"
            ) from exc
    return coerced


def _fetch_data_from_payload(payload: dict) -> pl.DataFrame:

    '''
    Compute the raw DataFrame for an experiment from the payload data source.

    Creates a HistoricalData instance with auth_token sourced from the
    DATA_API_AUTH_TOKEN environment variable (None when unset), then calls
    the method named in payload.inputData.selectedItems[0].params.method
    with the remaining params. HistoricalData methods set self.data rather
    than returning, so we read it back from the instance.

    When LOOP_ENV=test, falls back to HistoricalData._get_data_for_test to
    keep local smoke tests runnable without real credentials.

    Args:
        payload (dict): Loop web UI experiment design payload

    Returns:
        pl.DataFrame: Raw data ready to pass to UniversalExperimentLoop

    Raises:
        ValueError: If the payload has no data_source selectedItem, the
            method key is missing, or the method name is not in
            DATA_SOURCE_REGISTRY

    '''

    if os.getenv(_LOOP_ENV) == 'test':
        hd = HistoricalData()
        hd._get_data_for_test()
        return hd.data

    items = (payload.get('inputData') or {}).get('selectedItems') or []
    ds_item = next(
        (i for i in items if (i or {}).get('name') == 'data_source'),
        None,
    )
    if ds_item is None:
        raise ValueError(
            "payload.inputData.selectedItems has no entry with name='data_source'"
        )

    ds_params = dict(ds_item.get('params') or {})
    method_name = ds_params.pop('method', None)
    if method_name is None:
        raise ValueError(
            "data_source params missing required key: 'method'"
        )
    if method_name not in DATA_SOURCE_REGISTRY:
        raise ValueError(
            f"Unknown data source method '{method_name}'. "
            f"Allowed: {sorted(DATA_SOURCE_REGISTRY)}"
        )

    fetch_params = {k: v for k, v in ds_params.items() if v is not None}

    auth_token = os.getenv(_DATA_API_AUTH_TOKEN_ENV)
    hd = HistoricalData(auth_token=auth_token)
    method = getattr(hd, method_name)

    # Loop web-form values arrive as strings; coerce to int where the
    # method signature requires it (e.g. kline_size, n_rows) so the
    # ClickHouse query assembly downstream gets typed values.
    fetch_params = _coerce_string_params_by_signature(method, fetch_params)

    method(**fetch_params)

    return hd.data


def run_experiment(payload_path: Path,
                   experiment_dir: Path,
                   n_permutations: int = 100,
                   seed: int = 42) -> None:

    '''
    Compute and execute a Loop experiment from a JSON payload file.

    Args:
        payload_path (Path): Path to the Loop payload JSON file
        experiment_dir (Path): Directory to write all experiment artifacts to.
            Created if missing
        n_permutations (int): Number of parameter combinations to try
        seed (int): Random seed for the search strategy

    '''

    payload_path = Path(payload_path)
    experiment_dir = Path(experiment_dir)

    experiment_dir.mkdir(parents=True, exist_ok=True)

    payload_text = payload_path.read_text()
    payload = json.loads(payload_text)

    # Audit copy — survives the run, helps reproduce later
    (experiment_dir / 'payload.json').write_text(payload_text)

    # Fetch data explicitly using the payload's data source config so that
    # auth_token (from DATA_API_AUTH_TOKEN env) can be injected into
    # HistoricalData at instance construction time. The fetched DataFrame is
    # passed to UEL via data=, which bypasses manifest.fetch_data_for_env.
    raw_data = _fetch_data_from_payload(payload)

    sfd = LoopSFD(payload)

    domain = ParamDomain(sfd.params())
    strategy = RandomStrategy(domain, seed=seed)

    progress_file = experiment_dir / 'progress.json'
    progress_callback = make_progress_callback(progress_file, total=n_permutations)

    uel = UniversalExperimentLoop(
        sfd=sfd,
        data=raw_data,
        search_strategy=strategy,
        feedback_interval=1,
        experiment_dir=experiment_dir,
        intra_callback=progress_callback,
    )

    logger.info(
        'Starting Loop experiment: payload=%s out=%s n=%d',
        payload_path, experiment_dir, n_permutations,
    )

    uel.run(
        experiment_name=str(experiment_dir / 'run'),
        n_permutations=n_permutations,
    )

    logger.info('Experiment complete: %s', experiment_dir)


def main(argv: list[str] | None = None) -> int:

    '''
    Compute and execute a Loop experiment from command-line arguments.

    Args:
        argv (list | None): Command-line arguments excluding the program name

    Returns:
        int: Exit code (0 on success, 1 on failure)

    '''

    parser = argparse.ArgumentParser(
        description='Run a Loop web UI experiment payload through Limen UEL.',
    )
    parser.add_argument(
        'payload',
        type=Path,
        help='Path to the Loop payload JSON file',
    )
    parser.add_argument(
        '--out',
        type=Path,
        required=True,
        help='Experiment directory for all output artifacts',
    )
    parser.add_argument(
        '--n',
        type=int,
        default=100,
        help='Number of parameter combinations to try (default: 100)',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for the search strategy (default: 42)',
    )
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)',
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    )

    try:
        run_experiment(
            payload_path=args.payload,
            experiment_dir=args.out,
            n_permutations=args.n,
            seed=args.seed,
        )
    except Exception:
        logger.exception('Experiment failed')
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
