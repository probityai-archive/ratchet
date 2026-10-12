"""Harness value types: guards, procedures and content-addressed harness versions.

Calls, results, events, outcomes and traces are shared with the case suite and live in `ratchet_cases.types`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal

from ratchet_cases.types import canonical_json, sha256_hex

Author = Literal["seed", "improver", "human"]


@dataclass(frozen=True)
class Guard:
    id: str
    description: str
    source: str
    origin_failure_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Procedure:
    id: str
    name: str
    when: str
    steps: list[str]
    origin_failure_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Harness:
    """A complete, content-addressed harness version. Build new ones with `Harness.build`."""

    version: str
    parent: str | None
    system_prompt: str
    procedures: list[Procedure]
    guards: list[Guard]
    tool_allowlist: list[str]
    created_by: Author
    digest: str

    @staticmethod
    def content_of(
        parent: str | None,
        system_prompt: str,
        procedures: list[Procedure],
        guards: list[Guard],
        tool_allowlist: list[str],
        created_by: Author,
    ) -> dict[str, Any]:
        return {
            "parent": parent,
            "system_prompt": system_prompt,
            "procedures": [asdict(p) for p in procedures],
            "guards": [asdict(g) for g in guards],
            "tool_allowlist": sorted(tool_allowlist),
            "created_by": created_by,
        }

    @classmethod
    def build(
        cls,
        *,
        parent: str | None,
        system_prompt: str,
        procedures: list[Procedure],
        guards: list[Guard],
        tool_allowlist: list[str],
        created_by: Author,
    ) -> Harness:
        content = cls.content_of(parent, system_prompt, procedures, guards, tool_allowlist, created_by)
        digest = sha256_hex(canonical_json(content))
        return cls(
            version=digest[:12],
            parent=parent,
            system_prompt=system_prompt,
            procedures=list(procedures),
            guards=list(guards),
            tool_allowlist=sorted(tool_allowlist),
            created_by=created_by,
            digest=digest,
        )

    def verify_digest(self) -> bool:
        content = self.content_of(
            self.parent,
            self.system_prompt,
            self.procedures,
            self.guards,
            self.tool_allowlist,
            self.created_by,
        )
        digest = sha256_hex(canonical_json(content))
        return digest == self.digest and self.version == digest[:12]

    def child(self, **changes: Any) -> Harness:
        """A new version whose parent is this one. Fields not named keep their values."""
        base = {f.name: getattr(self, f.name) for f in fields(self) if f.name not in ("version", "digest")}
        base.update(changes)
        base["parent"] = self.version
        return Harness.build(**base)
