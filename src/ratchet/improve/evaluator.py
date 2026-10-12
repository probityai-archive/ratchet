"""The evaluator is frozen before any proposal: cases, graders, fault schedules and model configuration are
each digested, and the evaluator version is the digest of those digests. If any component changes after
held-out outcomes were seen, the version changes, and the loop refuses to mix the two."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from typing import Any

from ratchet_cases.types import canonical_json, sha256_hex


class EvaluatorChangedError(RuntimeError):
    """The evaluator changed mid-run; results from the two versions must never be compared."""


def _plain(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    return value


def _digest(value: Any) -> str:
    return sha256_hex(canonical_json(value))


def evaluator_identity(
    cases: Sequence[Any], model_config: Mapping[str, Any], suite_digest: str | None = None
) -> dict[str, Any]:
    """Component digests and the evaluator version. `model_config` never carries a key."""
    ordered = sorted(cases, key=lambda c: c.id)
    body = [
        {k: _plain(v) for k, v in _plain(c).items() if k not in ("grader", "faults")} if is_dataclass(c)
        else {"id": c.id, "split": c.split, "kind": c.kind, "must_succeed": c.must_succeed}
        for c in ordered
    ]  # fmt: skip
    components = {
        "cases": _digest(body),
        "graders": _digest([[c.id, _plain(getattr(c, "grader", None))] for c in ordered]),
        "faults": _digest([[c.id, _plain(getattr(c, "faults", None))] for c in ordered]),
        "model_config": _digest(dict(model_config)),
        "suite_file": suite_digest,
    }
    return {
        "version": _digest(components)[:12],
        "components": components,
        "model_config": dict(model_config),
        "cases": len(ordered),
    }


def require_same(frozen: Mapping[str, Any], current: Mapping[str, Any]) -> None:
    if frozen["version"] != current["version"]:
        changed = sorted(k for k, v in frozen["components"].items() if current["components"].get(k) != v)
        raise EvaluatorChangedError(
            f"evaluator changed from {frozen['version']} to {current['version']} ({', '.join(changed)}); "
            "start a new evaluator version instead of comparing across it"
        )
