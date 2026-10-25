from collections.abc import Mapping

from limen.yaml.schema import UEL_YAML_REJECTIONS

__all__: list[str] = []


def callback_error(yaml_reference: Mapping[str, object]) -> str | None:
    '''Admit saved CLI YAML without revalidating native scientific declarations.'''
    uel = yaml_reference.get('uel')
    if isinstance(uel, Mapping) and 'intra_callback' in uel:
        return 'uel.intra_callback: ' + UEL_YAML_REJECTIONS['intra_callback']
    return None
