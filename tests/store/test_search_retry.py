"""One unreachable answer from search-index management is retried; any other error is not."""

from __future__ import annotations

import pytest
from pymongo.errors import OperationFailure

from ratchet.store.atlas_setup import _retry_unreachable


def test_retries_unreachable_then_returns() -> None:
    calls: list[int] = []

    def flaky() -> str:
        calls.append(1)
        if len(calls) < 3:
            raise OperationFailure("Error connecting to Search Index Management service.", code=508)
        return "ok"

    assert _retry_unreachable(flaky, timeout_s=5.0, poll_s=0.0) == "ok"
    assert len(calls) == 3


def test_other_errors_are_raised_at_once() -> None:
    calls: list[int] = []

    def denied() -> None:
        calls.append(1)
        raise OperationFailure("not authorized", code=13)

    with pytest.raises(OperationFailure):
        _retry_unreachable(denied, timeout_s=5.0, poll_s=0.0)
    assert len(calls) == 1


def test_deadline_raises_the_last_unreachable_error() -> None:
    def down() -> None:
        raise OperationFailure("unreachable", code=508)

    with pytest.raises(OperationFailure):
        _retry_unreachable(down, timeout_s=0.0, poll_s=0.0)
