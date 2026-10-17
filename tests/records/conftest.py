from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "records: reads signed decision records from RATCHET_RECORDS_DIR; opt in with -m records",
    )
