"""Building blocks shared by every simulated system: tool definitions, argument checks, refusals."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

JSON = dict[str, Any]
Handler = Callable[[JSON, str], JSON]
ToolSpec = JSON


class ToolError(Exception):
    """A call the system refuses. The message is returned to the caller as `ToolResult.error`."""


class Context(Protocol):
    """What a system may ask of the world that hosts it."""

    def now(self) -> str: ...

    def next_id(self, prefix: str) -> str: ...

    def record(self, system: str, actor: str, op: str, args: JSON, result: JSON) -> None: ...


@dataclass(frozen=True)
class ToolDef:
    name: str
    description: str
    parameters: JSON
    handler: Handler

    def spec(self) -> ToolSpec:
        """The OpenAI function-tool schema for this tool."""
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }


def string(description: str, enum: list[str] | None = None) -> JSON:
    schema: JSON = {"type": "string", "description": description}
    if enum is not None:
        schema["enum"] = enum
    return schema


def integer(description: str) -> JSON:
    return {"type": "integer", "description": description}


def boolean(description: str) -> JSON:
    return {"type": "boolean", "description": description}


def array(description: str, items: JSON) -> JSON:
    return {"type": "array", "description": description, "items": items}


def obj(properties: JSON, required: list[str], description: str = "") -> JSON:
    schema: JSON = {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }
    if description:
        schema["description"] = description
    return schema


def params(required: JSON, optional: JSON | None = None) -> JSON:
    """A parameters object: every key of `required` is required, every key of `optional` is not."""
    return obj({**required, **(optional or {})}, sorted(required))


_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
}


def check_args(schema: JSON, value: Any, path: str = "args") -> str | None:
    """Validate `value` against the small JSON-schema subset the tools use. Returns the first problem."""
    kind = schema.get("type")
    allowed = _TYPES.get(str(kind))
    if allowed is None:
        return None
    if not isinstance(value, allowed) or (kind == "integer" and isinstance(value, bool)):
        return f"{path} must be {kind}"
    if "enum" in schema and value not in schema["enum"]:
        return f"{path} must be one of {schema['enum']}"
    if isinstance(value, list):
        for i, item in enumerate(value):
            problem = check_args(schema["items"], item, f"{path}[{i}]")
            if problem:
                return problem
    if isinstance(value, dict):
        props: JSON = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                return f"{path}.{key} is required"
        for key, item in value.items():
            if key not in props:
                return f"{path}.{key} is not a known field"
            problem = check_args(props[key], item, f"{path}.{key}")
            if problem:
                return problem
    return None


class System:
    """One simulated service. It owns `state[name]` and writes every change to its own audit log."""

    name = ""

    def __init__(self, state: JSON, ctx: Context) -> None:
        self.state = state
        self.ctx = ctx

    @staticmethod
    def default_state() -> JSON:
        return {}

    def tools(self) -> list[ToolDef]:
        raise NotImplementedError

    def record(self, actor: str, op: str, args: JSON, result: JSON) -> None:
        self.ctx.record(self.name, actor, op, args, result)

    def row(self, table: str, key: str, label: str) -> JSON:
        found = self.state[table].get(key)
        if found is None:
            raise ToolError(f"{label} not found: {key}")
        result: JSON = found
        return result
