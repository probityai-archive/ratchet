"""tau2-bench (retail, airline, telecom) as a Ratchet world.

tau2-bench reads `TAU2_DATA_DIR` once, when its first module is imported, so this package points it at the
data of the pinned commit before any submodule imports tau2-bench. An explicit `TAU2_DATA_DIR` wins.
"""

from __future__ import annotations

import os

from ratchet.bench.tau2.data import default_data_dir

os.environ.setdefault("TAU2_DATA_DIR", str(default_data_dir()))
