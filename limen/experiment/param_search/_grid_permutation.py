import hashlib
import json
import math
import random
from collections.abc import Mapping

__all__: list[str] = []

_SHUFFLE_VERSION = 2
GOLDEN_RATIO_CONJUGATE = 0.6180339887498949


def domain_digest(params: Mapping[str, list[object]]) -> str:
    return hashlib.sha256(json.dumps(dict(params), sort_keys=True, default=str).encode()).hexdigest()


def _permuted_index(index: int, n: int, seed: int) -> int:
    if not 0 <= index < n:
        raise ValueError('Grid permutation index is outside its domain')
    if n == 1:
        return 0
    half = ((n - 1).bit_length() + 1) // 2
    mask, width = (1 << half) - 1, (half + 7) // 8
    key = b'Limen.GridStrategy.v2\0' + str(seed).encode() + b'\0' + str(n).encode() + b'\0'
    value = index
    while True:
        left, right = value >> half, value & mask
        for round_index in range(8):
            raw = key + bytes([round_index]) + right.to_bytes(width, 'big')
            mixed = int.from_bytes(hashlib.shake_256(raw).digest(width), 'big') & mask
            left, right = right, left ^ mixed
        value = (left << half) | right
        if value < n:
            return value


def _lcg_params(n: int, seed: int | None) -> tuple[int, int]:

    '''
    Compute LCG parameters for a full-period permutation of [0, n).

    Finds a multiplier coprime to n and a seeded increment for
    a linear congruential generator that visits every index exactly once.

    Args:
        n (int): Size of the space to permute
        seed (int | None): Seed for choosing increment

    Returns:
        tuple[int, int]: (multiplier, increment) for the LCG
    '''

    multiplier = _find_coprime(n)
    rng = random.Random(seed)
    increment = rng.randint(0, n - 1)
    return multiplier, increment


def _find_coprime(n: int) -> int:

    '''Find a value coprime to n that provides good mixing.'''

    candidate = max(2, int(n * GOLDEN_RATIO_CONJUGATE))
    while math.gcd(candidate, n) != 1:
        candidate += 1
    return candidate


def _lcg_map(logical_index: int, n: int,
             multiplier: int, increment: int) -> int:

    '''Map logical index to physical index via LCG.'''

    return (multiplier * logical_index + increment) % n
