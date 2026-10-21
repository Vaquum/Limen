from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap
from ruamel.yaml.constructor import ConstructorError
from ruamel.yaml.constructor import RoundTripConstructor
from ruamel.yaml.nodes import MappingNode

from limen.yaml.config import RoundTripYAML


class _ManifestConstructor(RoundTripConstructor):

    def construct_mapping(self, node: MappingNode, maptyp: CommentedMap, deep: bool = False) -> None:
        for key_node, _ in node.value:
            if isinstance(key_node, MappingNode):
                raise ConstructorError(
                    'while constructing a manifest mapping', node.start_mark,
                    'mappings cannot be YAML keys', key_node.start_mark,
                )
        super().construct_mapping(node, maptyp, deep=deep)


def _parser_yaml() -> RoundTripYAML:
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.Constructor = _ManifestConstructor
    return yaml
