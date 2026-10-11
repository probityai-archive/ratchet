"""Make the pinned `tool_sandbox` package importable under the versions this repository locks.

Two facts about its imports, each measured against the pinned commit, need handling and nothing else is
touched:

- `tool_sandbox.common.tool_conversion` imports `langchain_core.pydantic_v1`, a module langchain-core
  removed in 0.3 (the lock holds 1.x), and uses only its `BaseModel` as an `issubclass` test on tool
  argument types. Pydantic's own `BaseModel` answers that test the same way, so it stands in under that name.
- `holidays` 0.51, pinned by `tool_sandbox`, warns about its own 1.0 versioning at import. The warning is
  about a future release, no code path can fix it, and it is silenced for that one import only.
"""

from __future__ import annotations

import sys
import types
import warnings

_LOADED = False


def load() -> None:
    """Idempotent: install the `pydantic_v1` stand-in and import `holidays` once, with its warning scoped."""
    global _LOADED
    if _LOADED:
        return
    if "langchain_core.pydantic_v1" not in sys.modules:
        import pydantic

        shim = types.ModuleType("langchain_core.pydantic_v1")
        shim.BaseModel = pydantic.BaseModel  # type: ignore[attr-defined]
        sys.modules["langchain_core.pydantic_v1"] = shim
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=Warning, module="holidays")
        import holidays  # noqa: F401
    _LOADED = True
