from typing import Protocol, cast

from ruamel.yaml import MappingNode, SequenceNode, YAML
from ruamel.yaml.constructor import ConstructorError, RoundTripConstructor
from typing_extensions import override

from limen.yaml.config import RoundTripYAML, round_trip_yaml


class _Node(Protocol):
    value: object
    start_mark: object


class _Constructor(Protocol):
    def construct_document(self, node: object) -> object: ...


class _ManifestConstructor(RoundTripConstructor):

    @override
    def construct_document(self, node: object) -> object:
        pending, seen = [node], set[int]()
        while pending:
            current = pending.pop()
            if id(current) not in seen:
                seen.add(id(current))
                if isinstance(current, MappingNode):
                    for key, value in cast(list[tuple[object, object]], cast(_Node, current).value):
                        if isinstance(key, MappingNode):
                            raise ConstructorError(None, None, 'mappings cannot be YAML keys', cast(_Node, key).start_mark)
                        pending.extend((key, value))
                elif isinstance(current, SequenceNode):
                    pending.extend(cast(list[object], cast(_Node, current).value))
        return cast(_Constructor, super()).construct_document(node)


def parser_yaml() -> RoundTripYAML:
    yaml = cast(YAML, round_trip_yaml(preserve_quotes=True))
    yaml.Constructor = _ManifestConstructor
    return yaml
