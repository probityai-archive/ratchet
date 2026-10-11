"""The policy service: versioned documents. A new version can publish at any moment and applies at once."""

from __future__ import annotations

import copy

from ratchet.world.system import JSON, System, ToolDef, ToolError, params, string


def current_version(doc: JSON, at: str) -> JSON | None:
    live = [v for v in doc["versions"] if v["published_at"] <= at]
    return max(live, key=lambda v: v["version"]) if live else None


def citation(doc_id: str, version: int) -> str:
    return f"{doc_id}@v{version}"


class Policy(System):
    name = "policy"

    @staticmethod
    def default_state() -> JSON:
        return {"docs": {}}

    def tools(self) -> list[ToolDef]:
        return [
            ToolDef(
                "policy_get",
                "Read the version of a policy document in force right now: version number, publication "
                "time, text, machine-readable rules and a citation string (doc@vN) to quote in ticket "
                "notes. A new version may publish at any time and replaces the old one immediately.",
                params({"doc_id": string("Policy document id, for example refund-policy.")}),
                self._get,
            ),
        ]

    def _get(self, args: JSON, actor: str) -> JSON:
        doc = self.row("docs", args["doc_id"], "policy document")
        version = current_version(doc, self.ctx.now())
        if version is None:
            raise ToolError(f"no version of {args['doc_id']} is in force")
        out = copy.deepcopy(version)
        out["doc_id"] = args["doc_id"]
        out["citation"] = citation(args["doc_id"], version["version"])
        return {"policy": out}

    def publish(self, doc_id: str, version: JSON) -> None:
        """Publish a new version now (a fault operation, not a tool)."""
        doc = self.row("docs", doc_id, "policy document")
        entry = copy.deepcopy(version)
        entry["published_at"] = self.ctx.now()
        doc["versions"].append(entry)
        self.record("policy-team", "publish", {"doc_id": doc_id, "version": version["version"]}, {})
