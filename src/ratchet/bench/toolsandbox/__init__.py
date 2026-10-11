"""A stateful tool-use benchmark as a Ratchet world: its scenarios, tools, user simulator and milestone
grader, unchanged.

Importing this package prepares the pinned `tool_sandbox` dependency (see `_compat`), so every module below
imports it directly.
"""

from ratchet.bench.toolsandbox import _compat

_compat.load()
