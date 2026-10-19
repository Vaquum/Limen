from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import import_module
from typing import cast

import polars as pl

from limen.backtest.trade_contract import JsonValue, TradeInputs, TradePolicy, contract_digest, json_value, source_binding

def component_identity(value: object) -> dict[str, JsonValue]:
    identity = cast(Callable[[object], dict[str, JsonValue]], vars(import_module('limen.sfd.reference_architecture.direction_sizing'))['_identity'])
    return identity(value)


@dataclass(frozen=True)
class AllocationConfig:
    method: Callable[..., dict[str, object]]
    params: Mapping[str, object]

    def __post_init__(self) -> None:
        _ = component_identity(self.method)


@dataclass(frozen=True)
class AllocationBatch:
    batch_id: str
    partition_start_ns: int
    partition_end_ns: int
    row_ids: tuple[str, ...]
    available_at_ns: tuple[int, ...]
    primary_inputs: Mapping[str, object]
    meta_features: pl.DataFrame
    stop_loss_bps: tuple[float | None, ...]
    barrier_ready: tuple[bool, ...]

    def digest(self) -> str:
        identities = pl.DataFrame({'row_id': self.row_ids, 'available_at_ns': self.available_at_ns,
                                   'stop_loss_bps': self.stop_loss_bps, 'barrier_ready': self.barrier_ready})
        if identities.height != len(self.stop_loss_bps) or identities.height != len(self.barrier_ready) or identities.height != self.meta_features.height or len(set(self.row_ids)) != len(self.row_ids):
            raise ValueError('Allocation batch requires complete unique ordered membership')
        features = source_binding(self.meta_features, self.batch_id, self.partition_start_ns, self.partition_end_ns, 1, 'allocation_features').checksum
        primary = {}
        for key, value in self.primary_inputs.items():
            if isinstance(value, pl.DataFrame):
                primary[key] = source_binding(value, key, self.partition_start_ns, self.partition_end_ns, 1, 'allocation_primary_inputs').checksum
            else:
                primary[key] = contract_digest({'value': json_value(value)})
        return contract_digest({'batch_id': self.batch_id, 'start': self.partition_start_ns, 'end': self.partition_end_ns,
                                'rows': json_value(identities.to_dicts()), 'features': features, 'primary': primary})


@dataclass(frozen=True)
class OpportunityLabels:
    rows: pl.DataFrame
    features: pl.DataFrame
    binding: Mapping[str, object]


@dataclass(frozen=True)
class AllocationContext:
    manifest: object
    raw_train: pl.DataFrame
    round_params: Mapping[str, object]
    policy: TradePolicy
    batches: Mapping[str, AllocationBatch]
    barriers: pl.DataFrame | None
    initial_equity: float
    train_inputs: TradeInputs


__all__: list[str] = []


def _helper(name: str) -> Callable[..., object]:
    return cast(Callable[..., object], getattr(import_module('limen.experiment._allocation_folds'), name))

def admit(manifest: object, params: Mapping[str, object]) -> None:
    if getattr(manifest, 'allocation_config', None) is None:
        return
    _ = _helper('admit')(manifest, params)

def attach(manifest: object, raw: pl.DataFrame, params: Mapping[str, object], data: dict[str, object]) -> dict[str, object]:
    if getattr(manifest, 'allocation_config', None) is None:
        return data
    return cast(dict[str, object], _helper('attach')(manifest, raw, params, data))

def run(manifest: object, data: dict[str, object], params: Mapping[str, object]) -> dict[str, object]:
    return cast(dict[str, object], _helper('run')(manifest, data, params))
