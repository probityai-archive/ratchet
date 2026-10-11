"""Cross-model transfer: does a harness admitted on one model help a model it was never derived on?

Guards are code, so a guard admitted on the derivation model should block the same failure whatever model
proposes the call. This package runs the seed harness and an admitted harness FRESH and paired on the
held-out and audit splits with another model, and reports counts, denominators, run-to-run noise and the
discordant pairs, never a single headline number.
"""

from ratchet.bench.transfer.analysis import (
    ArmStats,
    CaseRow,
    SplitSummary,
    mcnemar_exact,
    paired_rows,
    summarize,
)

__all__ = ["ArmStats", "CaseRow", "SplitSummary", "mcnemar_exact", "paired_rows", "summarize"]
