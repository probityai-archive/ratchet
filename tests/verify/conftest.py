from __future__ import annotations

import pytest

from ratchet.improve.receipt import Signer


@pytest.fixture(scope="module")
def signer() -> Signer:
    return Signer.generate()


@pytest.fixture(scope="module")
def other_signer() -> Signer:
    return Signer.generate()
