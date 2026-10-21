from typing import cast
from typing import Protocol

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap
from ruamel.yaml.constructor import ConstructorError
from ruamel.yaml.constructor import RoundTripConstructor
from ruamel.yaml.nodes import MappingNode
from typing_extensions import override

from limen.yaml.config import RoundTripYAML


class _MappingNode(Protocol):
    value: list[tuple[object, object]]
    start_mark: object


class _Constructor(Protocol):
    def construct_mapping(self, node: MappingNode, maptyp: CommentedMap, deep: bool = False) -> None: ...


class _ManifestConstructor(RoundTripConstructor):

    @override
    def construct_mapping(self, node: MappingNode, maptyp: CommentedMap, deep: bool = False) -> None:
        mapping = cast(_MappingNode, node)
        for key_node, _ in mapping.value:
            if isinstance(key_node, MappingNode):
                raise ConstructorError(
                    'while constructing a manifest mapping', mapping.start_mark,
                    'mappings cannot be YAML keys', cast(_MappingNode, key_node).start_mark,
                )
        cast(_Constructor, super()).construct_mapping(node, maptyp, deep)


def parser_yaml() -> RoundTripYAML:
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.Constructor = _ManifestConstructor
    return yaml
