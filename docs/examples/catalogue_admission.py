"""Resolve declared exports to their exact reference contracts without importing backends."""
import ast
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=512)
def _source_tree(source: str):
    return ast.parse(source)


def module_path(root: Path, module: str) -> Path:
    path = root.joinpath(*module.split('.'))
    return path.with_suffix('.py') if path.with_suffix('.py').is_file() else path / '__init__.py'


def resolve(root, module, symbol, seen=()):
    if (module, symbol) in seen:
        raise ValueError('Circular public export: ' + module + '.' + symbol)
    source = module_path(root, module)
    if not source.is_file():
        raise ValueError('Public export has no package source: ' + module + '.' + symbol)
    tree = _source_tree(source.read_text())
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == symbol:
            return source, node
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        if any(isinstance(target, ast.Name) and target.id == symbol for target in targets):
            return source, node
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for alias in node.names:
                if (alias.asname or alias.name) == symbol:
                    child = module_path(root, node.module + '.' + alias.name)
                    if node.module == module and child.is_file():
                        return child, None
                    return resolve(root, node.module, alias.name, (*seen, (module, symbol)))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == '_LAZY_EXPORTS' for target in node.targets):
            exports = ast.literal_eval(node.value)
            if symbol in exports:
                if exports[symbol][1] is None:
                    return module_path(root, exports[symbol][0]), None
                return resolve(root, exports[symbol][0], exports[symbol][1], (*seen, (module, symbol)))
    child = module_path(root, module + '.' + symbol)
    if child.is_file():
        return child, None
    raise ValueError('Unresolved public export: ' + module + '.' + symbol)


def declared_contracts(root: Path):
    contracts = {}
    for source in sorted((root / 'limen').rglob('*.py')):
        if any(part.startswith('_') and part != '__init__.py' for part in source.relative_to(root).parts):
            continue
        tree = _source_tree(source.read_text())
        declarations = [node for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == '__all__' for target in node.targets)]
        if not declarations:
            continue
        module = '.'.join((source.parent if source.name == '__init__.py' else source.with_suffix('')).relative_to(root).parts)
        for name in ast.literal_eval(declarations[0].value):
            target, node = resolve(root, module, name)
            public = module + '.' + name
            if node is None or name == '__version__':
                contracts[public] = {'source': str(target.relative_to(root)), 'symbol': name, 'role': 'module' if node is None else 'metadata'}
                continue
            symbol = node.name if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) else name
            role = 'signature' if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) else 'definition'
            contracts[public] = {'source': str(target.relative_to(root)), 'symbol': symbol, 'role': role}
            if isinstance(node, ast.ClassDef):
                for member in node.body:
                    if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)) and (not member.name.startswith('_') or member.name in ('__init__', '__call__')):
                        # A property getter is the public member; its setter has the same name.
                        setter = any(isinstance(decorator, ast.Attribute) and decorator.attr == 'setter' for decorator in member.decorator_list)
                        if not setter:
                            contracts[public + '.' + member.name] = {'source': str(target.relative_to(root)), 'symbol': symbol + '.' + member.name, 'role': 'signature'}
                    elif isinstance(member, ast.ImportFrom) and member.module and not member.level:
                        for alias in member.names:
                            method = alias.asname or alias.name
                            if not method.startswith('_'):
                                authority, function = resolve(root, member.module, alias.name)
                                if isinstance(function, ast.FunctionDef):
                                    contracts[public + '.' + method] = {'source': str(authority.relative_to(root)), 'symbol': function.name, 'role': 'signature'}
    return contracts


def validate_admission(root, recorded):
    if recorded != declared_contracts(root):
        raise ValueError('Public export/member admission differs from current source')
    from check_evidence import all_reference_bindings
    references = {(item['path'], item['region']) for item in all_reference_bindings(root)}
    for public, entry in recorded.items():
        if entry['role'] not in ('module', 'metadata') and (entry['source'], entry['role'] + ':' + entry['symbol']) not in references:
            raise ValueError('Public entry has no exact reference contract: ' + public)


def cli_contracts():
    from limen.cli.main import cli
    import click
    result = {}
    for name, command in [('limen', cli), *[('limen ' + key, value) for key, value in sorted(cli.commands.items())]]:
        context = click.Context(command)
        help_option = command.get_help_option(context)
        params = [*command.params, *([help_option] if help_option is not None else [])]
        entries = []
        for parameter in params:
            entries.append({'name': parameter.name, 'kind': parameter.param_type_name,
                            'spellings': parameter.opts, 'type': parameter.type.to_info_dict(),
                            'default': None if parameter.required or parameter.param_type_name == 'argument' else parameter.get_default(context), 'required': parameter.required,
                            'nargs': parameter.nargs, 'multiple': getattr(parameter, 'multiple', False),
                            'help': getattr(parameter, 'help', None)})
        result[name] = entries
    return result


def validate_cli_contracts(root):
    import json
    if json.loads((root / 'docs/examples/cli-contracts.json').read_text()) != cli_contracts():
        raise ValueError('CLI argument, flag, default or type contract differs from current Click declarations')
