"""Point tau2-bench at the pinned task data before any test module imports it, fetching it once."""

from __future__ import annotations

from ratchet.bench.tau2.data import data_dir, fetch_data, is_complete

if not is_complete(data_dir()):
    fetch_data(data_dir())
