import copy
from functools import lru_cache

from limen.yaml.parser import parse


@lru_cache(maxsize=8)
def _parsed(text: str):
    return parse(text)


def parse_cached(text: str):
    return copy.deepcopy(_parsed(text))
