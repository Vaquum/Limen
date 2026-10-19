import copy
from functools import lru_cache

from limen.yaml.parser import parse


@lru_cache(maxsize=8)
def _parsed(text: str):
    config, errors = parse(text)
    assert not errors
    return config


def yaml_config(text: str):
    return copy.deepcopy(_parsed(text))
