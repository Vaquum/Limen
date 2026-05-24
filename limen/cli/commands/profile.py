from pathlib import Path

import click

from limen.yaml.compiler import CompiledSFD
from limen.yaml.parser import parse
from limen.yaml.profiler import ProfileResult
from limen.yaml.profiler import profile
from limen.yaml.validator import validate


_COMPLEXITY_COLOURS = {
    'low': 'green',
    'medium': 'yellow',
    'high': 'red',
    'extreme': 'red',
}


def run_profile(yaml_path: Path) -> bool:

    '''
    Validate, compile, and profile a YAML experiment file.

    Args:
        yaml_path (Path): Path to the YAML experiment file

    Returns:
        bool: True on success, False on validation failure or profile error

    '''

    click.echo(f"Profiling {yaml_path} ...")

    yaml_dict, parse_errors = parse(yaml_path)
    if parse_errors:
        for e in parse_errors:
            location = f' (line {e.line})' if e.line else ''
            click.secho(f'  PARSE ERROR{location}: {e.message}', fg='red')
        return False

    result = validate(yaml_dict)
    for e in result.errors:
        location = f' (line {e.line})' if e.line else ''
        path = f'  [{e.path}]' if e.path else ''
        suggestion = f'\n    → {e.suggestion}' if e.suggestion else ''
        click.secho(f'  ERROR{path}{location}: {e.message}{suggestion}', fg='red')
    for w in result.warnings:
        location = f' (line {w.line})' if w.line else ''
        path = f'  [{w.path}]' if w.path else ''
        click.secho(f'  WARN{path}{location}: {w.message}', fg='yellow')

    if not result.valid:
        click.secho(f'  ✗ {len(result.errors)} validation error(s) — aborting', fg='red')
        return False

    click.secho('  ✓ Valid', fg='green')

    try:
        sfd = CompiledSFD(yaml_dict)
        prof = profile(sfd)
    except Exception as exc:  # noqa: BLE001
        click.secho(f'  ✗ Profile failed: {exc}', fg='red')
        return False

    _print_profile(prof)
    return True


def _print_profile(prof: ProfileResult) -> None:

    click.echo('')
    _print_permutation_space(prof)
    _print_runtime_sampling(prof)

    for w in prof.warnings:
        click.secho(f'  WARN: {w}', fg='yellow')
    for e in prof.errors:
        click.secho(f'  ERROR: {e}', fg='red')


def _print_permutation_space(prof: ProfileResult) -> None:

    colour = _COMPLEXITY_COLOURS.get(prof.complexity_rating, 'white')
    rating_label = click.style(f'[{prof.complexity_rating}]', fg=colour)
    n_params = len(prof.param_cardinalities)
    click.echo("  Permutation space")
    click.echo(f"    Total:       {_format_space(prof.total_permutations)}  {rating_label}")
    click.echo(f"    Parameters:  {n_params}")

    sorted_params = sorted(
        prof.param_cardinalities.items(),
        key=lambda kv: kv[1],
        reverse=True,
    )
    for name, card in sorted_params:
        click.echo(f"      {name:<24} {card} value{'s' if card != 1 else ''}")


def _print_runtime_sampling(prof: ProfileResult) -> None:

    click.echo('')
    attempted = prof.sample_permutations_attempted
    completed = prof.sample_permutations_completed

    if attempted == 0:
        click.echo('  Runtime sampling')
        click.echo('    Skipped — no test_data_source configured')
        return

    click.echo(f"  Runtime sampling  ({completed} of {attempted} completed)")

    if prof.sample_time_seconds_per_permutation is not None:
        t = prof.sample_time_seconds_per_permutation
        click.echo(f"    Per permutation: {t:.3f}s")
        estimated_total = t * prof.total_permutations
        click.echo(f"    Estimated total: {_format_duration(estimated_total)}  "
                   f"({_format_space(prof.total_permutations)} permutations)")
    else:
        click.echo('    Per permutation: —  (no completed runs)')


_SECONDS_PER_MINUTE = 60
_SECONDS_PER_HOUR = 3600

_SPACE_UNITS = [
    (10 ** 12, 'T'),
    (10 ** 9, 'B'),
    (10 ** 6, 'M'),
    (10 ** 3, 'K'),
]


def _format_space(n: int) -> str:

    if n >= 10 ** 15:
        return f"{float(n):.2e}"
    for threshold, suffix in _SPACE_UNITS:
        if n >= threshold:
            return f"{n / threshold:.1f}{suffix}"
    return str(n)


def _format_duration(seconds: float) -> str:

    if seconds < _SECONDS_PER_MINUTE:
        return f"{seconds:.1f}s"
    if seconds < _SECONDS_PER_HOUR:
        return f"{seconds / _SECONDS_PER_MINUTE:.1f}m"
    return f"{seconds / _SECONDS_PER_HOUR:.1f}h"
