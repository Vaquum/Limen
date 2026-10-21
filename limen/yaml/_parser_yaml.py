from collections.abc import Callable
from typing import cast

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap
from ruamel.yaml.constructor import ConstructorError
from ruamel.yaml.constructor import RoundTripConstructor
from ruamel.yaml.nodes import MappingNode
from ruamel.yaml.nodes import Node
from typing_extensions import override

from limen.yaml.config import RoundTripYAML


__all__ = ['_parser_yaml']


class _ManifestConstructor(RoundTripConstructor):

    @override
    def construct_mapping(self, node: MappingNode, maptyp: CommentedMap, deep: bool = False) -> None:
        pairs = cast(list[tuple[Node, Node]], getattr(node, 'value'))
        for key_node, _ in pairs:
            if isinstance(key_node, MappingNode):
                raise ConstructorError(
                    'while constructing a manifest mapping', getattr(node, 'start_mark'),
                    'mappings cannot be YAML keys', getattr(key_node, 'start_mark'),
                )
        construct = cast(Callable[[MappingNode, CommentedMap, bool], None], getattr(super(), 'construct_mapping'))
        construct(node, maptyp, deep)


def _parser_yaml() -> RoundTripYAML:
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.Constructor = _ManifestConstructor
    return yaml
