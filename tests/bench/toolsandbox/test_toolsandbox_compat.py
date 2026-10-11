from __future__ import annotations

import sys

from ratchet.bench.toolsandbox import _compat


def test_loading_twice_changes_nothing() -> None:
    _compat.load()
    shim = sys.modules["langchain_core.pydantic_v1"]
    _compat.load()
    assert sys.modules["langchain_core.pydantic_v1"] is shim


def test_the_stand_in_answers_the_subclass_test_the_benchmark_makes() -> None:
    import pydantic

    shim = sys.modules["langchain_core.pydantic_v1"]
    assert shim.BaseModel is pydantic.BaseModel


def test_the_benchmark_imports_under_warnings_as_errors() -> None:
    import tool_sandbox.common.tool_conversion as conversion

    assert vars(conversion)["BaseModel"] is sys.modules["langchain_core.pydantic_v1"].BaseModel
