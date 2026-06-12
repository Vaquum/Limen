import ast
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent

PYTEST_ONLY_TESTS = {
    'test_historical_data': {
        'test_get_arrow_file_zero_copy_single_batch',
        'test_get_arrow_file_date_range_is_zero_copy_slice',
        'test_get_arrow_file_date_range_on_datetime_only_file',
        'test_get_arrow_file_rejects_compressed',
        'test_get_arrow_file_rejects_non_single_batch',
        'test_get_arrow_file_rejects_row_limit_with_closed_date_window',
        'test_get_arrow_file_row_count_limit_zero_copy',
        'test_get_arrow_file_view_outlives_call',
    },
    'test_tabpfn': {'test_tabpfn'},
}


def _module_test_functions(module_path: Path) -> dict[str, ast.FunctionDef]:
    '''
    Collect the top-level test functions defined in one test module.

    Args:
        module_path (Path): The test module to parse.

    Returns:
        dict[str, ast.FunctionDef]: Test function nodes keyed by name.
    '''

    tree = ast.parse(module_path.read_text(encoding='utf-8'))

    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith('test_')
    }


def _registered_test_names() -> dict[str, set[str]]:
    '''
    Collect per-module test names registered in the tests/run.py suite list.

    A name counts as registered only when it is imported from a test module
    and appears (under its imported alias) in the module-level `tests` list.

    Returns:
        dict[str, set[str]]: Original function names keyed by module stem.
    '''

    tree = ast.parse((TESTS_DIR / 'run.py').read_text(encoding='utf-8'))

    listed: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.List):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == 'tests':
                listed = {
                    element.id
                    for element in node.value.elts
                    if isinstance(element, ast.Name)
                }

    registered: dict[str, set[str]] = {}
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module is None or not node.module.startswith('tests.test_'):
            continue
        module = node.module.removeprefix('tests.')
        for alias in node.names:
            if (alias.asname or alias.name) in listed:
                registered.setdefault(module, set()).add(alias.name)

    return registered


def _names_referenced_by(
    functions: dict[str, ast.FunctionDef],
    roots: set[str],
) -> set[str]:
    '''
    Collect test names referenced inside the bodies of the given root tests.

    A registered aggregate test that invokes sibling test functions (the
    tests/test_indicators_vs_talib.py pattern) covers those siblings.

    Args:
        functions (dict[str, ast.FunctionDef]): Test function nodes by name.
        roots (set[str]): Names of registered functions whose bodies to scan.

    Returns:
        set[str]: Defined test names referenced from any root body.
    '''

    referenced: set[str] = set()
    for root in roots:
        if root not in functions:
            continue
        for node in ast.walk(functions[root]):
            if isinstance(node, ast.Name) and node.id in functions:
                referenced.add(node.id)

    return referenced


def test_every_test_function_is_registered_in_run():
    '''
    Every top-level test function must run in CI: it is in the tests/run.py
    suite list, or invoked by a registered test in its module, or explicitly
    allowlisted as pytest-only.
    '''

    registered = _registered_test_names()

    unaccounted: dict[str, list[str]] = {}
    for module_path in sorted(TESTS_DIR.glob('test_*.py')):
        functions = _module_test_functions(module_path)
        module = module_path.stem
        covered = registered.get(module, set())
        covered = covered | _names_referenced_by(functions, covered)
        covered = covered | PYTEST_ONLY_TESTS.get(module, set())
        gap = sorted(set(functions) - covered)
        if gap:
            unaccounted[module] = gap

    assert not unaccounted, (
        f'test_run_registration found tests neither registered in tests/run.py '
        f'nor in the PYTEST_ONLY_TESTS allowlist: {unaccounted}'
    )


def test_pytest_only_allowlist_is_current():
    '''
    Every pytest-only allowlist entry must still exist in its module and
    must not also be registered in the tests/run.py suite list.
    '''

    registered = _registered_test_names()

    stale: dict[str, list[str]] = {}
    for module, names in PYTEST_ONLY_TESTS.items():
        functions = _module_test_functions(TESTS_DIR / f'{module}.py')
        gone = sorted(names - set(functions))
        double = sorted(names & registered.get(module, set()))
        if gone or double:
            stale[module] = gone + double

    assert not stale, (
        f'test_run_registration found stale pytest-only allowlist entries: '
        f'{stale}'
    )
