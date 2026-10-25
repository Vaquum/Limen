from pathlib import Path

__all__: list[str] = []


def admit_output(prefix: Path, expanded: str) -> Path:
    '''Select a direct-YAML destination strictly inside the resolved mode root.'''
    relative = Path(expanded)
    if relative.is_absolute():
        raise ValueError('uel.output_path must be relative to the results directory')
    root = prefix.resolve()
    destination = (root / relative).resolve()
    if destination == root or not destination.is_relative_to(root):
        raise ValueError('uel.output_path must resolve strictly inside the results directory')
    return destination
