"""The referee's results file: the per-case held-out tables a driver may not see, signed with the decision
key, and the verification bundle behind each decision.

The referee writes, under its own results directory:

    bundles/<id>/receipt.dsse            the decision's DSSE envelope
    bundles/<id>/manifest.json           which file is which report, trace list and harness
    bundles/<id>/reports/<role>.json     each evaluation report exactly as its digest was signed
    bundles/<id>/traces/<role>.jsonl     the trace documents of that report, one per line
    bundles/<id>/harnesses/<role>.json   the parent and candidate harness documents
    cohorts/<schedule>/arms/<arm>.json   a v3 cohort's execution records per arm, as their digests were signed
    cohorts/<schedule>/traces/<arm>.jsonl  its traces, appended as each execution completes
    entries.jsonl                        one line per decision, appended as each decision is signed
    <name>.dsse                          the signed results document, written by `publish`

A bundle is written the moment its decision is signed, so a stopped run keeps every decision it made. The
results document lists every decision with its per-case held-out table and the digest of its receipt, plus
the run-level tables (repeatability, parent discordance, identity controls). `ratchet.verify.results`
checks the signature, verifies each receipt from its bundle (re-grading every held-out run from the case
documents) and recomputes each table from the bundled reports.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ratchet_cases.types import sha256_hex

from ratchet.canonical import canonical_json

RESULTS_TYPE = "application/vnd.ratchet.referee-results+json"  # read by ratchet.verify.results
RESULTS_VERSION = 1


def bundle_id(decision_id: str) -> str:
    """A file-system-safe name for a decision's bundle."""
    return sha256_hex(decision_id.encode("utf-8"))[:16]


def _safe(name: str) -> str:
    return name.replace(":", "_").replace("/", "_")


def _dump(path: Path, doc: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, sort_keys=True, default=str), encoding="utf-8")


def _jsonl(path: Path, docs: Sequence[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(d, sort_keys=True) + "\n" for d in docs), encoding="utf-8")


class Ledger:
    """What the referee records for its results file. Without a directory it keeps the entries in memory and
    `publish` is refused, so a referee that was given no results directory cannot claim to have one."""

    def __init__(self, root: Path | None) -> None:
        self.root = None if root is None else Path(root)
        self.entries: list[dict[str, Any]] = []
        self.runs: dict[str, Any] = {}
        self._lock = threading.Lock()
        if self.root is not None:
            self.root.mkdir(parents=True, exist_ok=True)

    def bundle(
        self,
        decision_id: str,
        envelope: bytes,
        reports: Mapping[str, Any],
        traces: Mapping[str, Sequence[Any]],
        harnesses: Mapping[str, Any],
        *,
        cohort: str | None = None,
        arms: Sequence[str] = (),
    ) -> str | None:
        """Write one decision's verification bundle; returns its path relative to the results directory. A
        v3 decision names its cohort directory and the arms it compares instead of evaluation reports."""
        if self.root is None:
            return None
        rel = f"bundles/{bundle_id(decision_id)}"
        base = self.root / rel
        (base / "receipt.dsse").parent.mkdir(parents=True, exist_ok=True)
        (base / "receipt.dsse").write_bytes(envelope)
        manifest: dict[str, Any] = {"decision_id": decision_id, "reports": {}, "traces": {}, "harnesses": {}}
        if cohort is not None:
            # Relative to the results directory: a cohort's records are shared by its decisions.
            manifest["arms"] = {a: f"{cohort}/arms/{_safe(a)}.json" for a in arms}
            manifest["arm_traces"] = {a: f"{cohort}/traces/{_safe(a)}.jsonl" for a in arms}
        for role, doc in reports.items():
            name = _safe(role)
            _dump(base / "reports" / f"{name}.json", doc)
            manifest["reports"][role] = f"reports/{name}.json"
        for role, docs in traces.items():
            name = _safe(role)
            _jsonl(base / "traces" / f"{name}.jsonl", docs)
            manifest["traces"][role] = f"traces/{name}.jsonl"
        for role, doc in harnesses.items():
            _dump(base / "harnesses" / f"{role}.json", doc)
            manifest["harnesses"][role] = f"harnesses/{role}.json"
        _dump(base / "manifest.json", manifest)
        return rel

    def cohort(self, schedule_digest: str) -> str | None:
        """The directory a v3 cohort's records go to, relative to the results directory."""
        if self.root is None:
            return None
        rel = f"cohorts/{schedule_digest[:16]}"
        (self.root / rel / "traces").mkdir(parents=True, exist_ok=True)
        return rel

    def append_trace(self, cohort: str | None, arm: str, doc: Any) -> None:
        if self.root is None or cohort is None:
            return
        line = json.dumps(doc, sort_keys=True) + "\n"
        with (
            self._lock,
            (self.root / cohort / "traces" / f"{_safe(arm)}.jsonl").open("a", encoding="utf-8") as f,
        ):
            f.write(line)

    def cohort_records(self, cohort: str | None, records: Mapping[str, Sequence[Any]]) -> None:
        if self.root is None or cohort is None:
            return
        for arm, docs in records.items():
            _dump(self.root / cohort / "arms" / f"{_safe(arm)}.json", list(docs))

    def add(self, entry: Mapping[str, Any]) -> None:
        with self._lock:
            self.entries.append(dict(entry))
            if self.root is not None:
                with (self.root / "entries.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, sort_keys=True, default=str) + "\n")

    def run_traces(self, name: str, docs: Sequence[Any]) -> None:
        """The traces behind a run-level table (the repeatability probe's two runs), for recomputing it."""
        if self.root is not None:
            _jsonl(self.root / "runs" / f"{_safe(name)}.jsonl", docs)

    def note(self, key: str, value: Any) -> None:
        """A run-level table (repeatability, discordance, identity controls) for the results document."""
        with self._lock:
            self.runs[key] = value
            if self.root is not None:
                _dump(self.root / "runs" / f"{key}.json", value)

    def document(self, name: str, keyid: str, evaluator: Mapping[str, Any] | None) -> dict[str, Any]:
        return {
            "type": "ratchet-referee-results",
            "version": RESULTS_VERSION,
            "name": name,
            "signer_keyid": keyid,
            "evaluator": None if evaluator is None else evaluator.get("version"),
            "runs": dict(self.runs),
            "decisions": list(self.entries),
        }

    def publish(self, name: str, signer: Any, evaluator: Mapping[str, Any] | None) -> dict[str, Any]:
        """Sign the results document with the decision key and write it as `<name>.dsse`. Returns what a
        driver may record: the path, the envelope's SHA-256 and the counts."""
        if self.root is None:
            raise ValueError("the referee has no results directory")
        if not name or "/" in name or name.startswith("."):
            raise ValueError(f"results name {name!r} is not a plain file name")
        doc = self.document(name, signer.keyid, evaluator)
        env: bytes = signer.sign(RESULTS_TYPE, canonical_json(doc))
        path = self.root / f"{name}.dsse"
        path.write_bytes(env)
        return {
            "path": str(path),
            "sha256": sha256_hex(env),
            "decisions": len(self.entries),
            "admitted": sum(1 for e in self.entries if e.get("admitted")),
            "signer_keyid": signer.keyid,
        }
