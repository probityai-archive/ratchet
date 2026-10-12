"""Load an agent given as `module:function`.

`function(messages, tools)` returns one OpenAI-style assistant message.
"""

from __future__ import annotations

import importlib
from typing import Any, cast

from ratchet_cases.run import Act


class AdapterError(ValueError):
    """The named callable cannot be loaded."""


def load(spec: str) -> Act:
    """Import `module:function` and return the function. The module must be importable from `sys.path`."""
    module_name, sep, attr = spec.partition(":")
    if not sep or not module_name or not attr:
        raise AdapterError(f"expected module:function, got {spec!r}")
    try:
        module = importlib.import_module(module_name)
    except ImportError as err:
        raise AdapterError(f"cannot import {module_name!r}: {err}") from err
    target: Any = module
    for part in attr.split("."):
        if not hasattr(target, part):
            raise AdapterError(f"{module_name!r} has no attribute {attr!r}")
        target = getattr(target, part)
    if not callable(target):
        raise AdapterError(f"{spec!r} is not callable")
    return cast(Act, target)
