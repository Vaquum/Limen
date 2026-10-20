import ast
from pathlib import Path


def validate_test_reference(root: Path, reference: str) -> None:
    parts = reference.split('::')
    path = root / parts[0]
    roots = (root / 'tests', root / 'docs/examples/tests')
    if len(parts) not in (2, 3) or not path.name.startswith('test_') or path.suffix != '.py' or not path.is_file() or not any(path.resolve().is_relative_to(directory.resolve()) for directory in roots):
        raise ValueError('Unknown documentation proof test: ' + reference)
    tree = ast.parse(path.read_text())
    scope = tree.body
    decorators = [node.value for node in scope if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'pytestmark' for target in node.targets)]
    if len(parts) == 3:
        classes = [node for node in scope if isinstance(node, ast.ClassDef) and node.name == parts[1] and node.name.startswith('Test')]
        if len(classes) != 1:
            raise ValueError('Unknown documentation proof test class: ' + reference)
        decorators += classes[0].decorator_list
        scope = classes[0].body
    functions = [node for node in scope if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == parts[-1] and node.name.startswith('test_')]
    if len(functions) != 1:
        raise ValueError('Unknown documentation proof test: ' + reference)
    decorators += functions[0].decorator_list
    if any(isinstance(node, ast.Attribute) and node.attr in ('skip', 'skipif', 'xfail') for decorator in decorators for node in ast.walk(decorator)):
        raise ValueError('Skipped documentation proof test: ' + reference)
