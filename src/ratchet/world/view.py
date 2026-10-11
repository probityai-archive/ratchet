"""A read-only, JSON-shaped view of the whole world, for guards and graders."""

from __future__ import annotations

import copy
from types import MappingProxyType
from typing import Any

from ratchet.world.system import JSON


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, MappingProxyType):
        return {k: _thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_thaw(v) for v in value]
    return value


class StateView:
    """Immutable snapshot: `view["payments"]["charges"]`. Mappings are read-only, lists are tuples.

    Top-level keys: `step` (agent calls so far), `now` (simulated UTC time), one key per system with its
    state, and `audit` (per-system list of every change, with actor and time).
    """

    __slots__ = ("_data",)
    _data: MappingProxyType[str, Any]

    def __init__(self, data: JSON) -> None:
        object.__setattr__(self, "_data", _freeze(copy.deepcopy(data)))

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("StateView is read-only")

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def keys(self) -> list[str]:
        return list(self._data.keys())

    def to_json(self) -> JSON:
        """A fresh mutable deep copy, for sending to a guard process."""
        thawed: JSON = _thaw(self._data)
        return thawed

    @property
    def step(self) -> int:
        step: int = self._data["step"]
        return step

    @property
    def now(self) -> str:
        now: str = self._data["now"]
        return now
