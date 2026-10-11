"""Run the transfer experiment and the endpoint proofs against an OpenAI-compatible endpoint.

Every run is FRESH: `evaluate_many` gives each run a new run id, and the two harnesses share one bounded
pool per repeat, so both arms see the same endpoint load at the same time (a paired design). The result
file records the endpoint's own account of what it served (`/v1/models`, `/version`), the harness versions
and digests, the case-suite digest and every per-case result, so the report can be rebuilt from it alone.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any

import httpx

from ratchet.agent.model import ChatModel, ModelClient, ModelConfig
from ratchet.bench.transfer.analysis import summarize
from ratchet.improve.evaluate import EvalReport, RunCase, evaluate_many
from ratchet.types import Guard, Harness, Procedure, Trace

PROBE_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
}
PROBE_MESSAGES = [
    {"role": "system", "content": "You answer by calling tools when a tool can answer."},
    {"role": "user", "content": "What is the weather in Gainesville right now? Use the tool."},
]


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def harness_from(raw: dict[str, Any]) -> Harness:
    """Rebuild a stored harness and refuse one whose content does not match its digest."""
    h = Harness(
        version=raw["version"],
        parent=raw["parent"],
        system_prompt=raw["system_prompt"],
        procedures=[Procedure(**p) for p in raw["procedures"]],
        guards=[Guard(**g) for g in raw["guards"]],
        tool_allowlist=list(raw["tool_allowlist"]),
        created_by=raw["created_by"],
        digest=raw["digest"],
    )
    if not h.verify_digest():
        raise ValueError(f"harness {h.version} does not match its digest")
    return h


def endpoint_info(
    base_url: str, api_key: str, transport: httpx.BaseTransport | None = None
) -> dict[str, Any]:
    """What the server says it serves: model ids, their root checkpoint and context, and its version."""
    headers = {"Authorization": f"Bearer {api_key}"}
    root = base_url.rstrip("/")
    with httpx.Client(timeout=30.0, transport=transport) as http:
        models = http.get(f"{root}/models", headers=headers)
        models.raise_for_status()
        served = [
            {"id": m.get("id"), "root": m.get("root"), "max_model_len": m.get("max_model_len")}
            for m in models.json().get("data", [])
        ]
        server = root.removesuffix("/v1")
        version: str | None = None
        resp = http.get(f"{server}/version", headers=headers)
        if resp.is_success:
            version = resp.json().get("version")
    return {"base_url": root, "served": served, "server_version": version}


def probe_tool_call(model: ChatModel) -> dict[str, Any]:
    """One tool-calling request; reports whether the endpoint returned a parsed tool call."""
    started = time.monotonic()
    turn = model.complete(PROBE_MESSAGES, [PROBE_TOOL])
    return {
        "latency_s": round(time.monotonic() - started, 3),
        "finish_reason": turn.finish_reason,
        "tool_calls": [{"name": c.name, "args": c.args} for c in turn.tool_calls],
        "arg_errors": turn.arg_errors,
        "input_tokens": turn.input_tokens,
        "output_tokens": turn.output_tokens,
        "content": turn.content,
        "tool_call_ok": any(c.name == "get_weather" for c in turn.tool_calls) and not turn.arg_errors,
    }


def trace_summary(trace: Trace, wall_s: float) -> dict[str, Any]:
    o = trace.outcome
    kinds: dict[str, int] = {}
    for e in trace.events:
        kinds[e.kind] = kinds.get(e.kind, 0) + 1
    final = next((e.payload for e in reversed(trace.events) if e.kind == "final"), {})
    return {
        "run_id": trace.run_id,
        "case_id": trace.case_id,
        "harness_version": trace.harness_version,
        "model": trace.model,
        "passed": o.passed,
        "reasons": o.reasons,
        "final_reason": final.get("reason"),
        "events": kinds,
        "cost": {
            "model_calls": o.cost.model_calls,
            "input_tokens": o.cost.input_tokens,
            "output_tokens": o.cost.output_tokens,
            "tool_calls": o.cost.tool_calls,
            "wall_s": o.cost.wall_s,
        },
        "wall_s": round(wall_s, 3),
    }


def run_transfer(
    base: Harness,
    cand: Harness,
    cases: Sequence[Any],
    model: Any,
    run_case: RunCase,
    *,
    repeats: int,
    workers: int,
    store: Any = None,
    log: Callable[[str], None] = lambda _m: None,
) -> tuple[list[EvalReport], list[EvalReport]]:
    """`repeats` independent fresh repeats; each repeat runs both harnesses on every case in one pool."""
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    base_reports: list[EvalReport] = []
    cand_reports: list[EvalReport] = []
    for r in range(repeats):
        log(f"repeat {r + 1}/{repeats}: {len(cases)} cases x 2 harnesses, {workers} workers")
        b, c = evaluate_many([base, cand], cases, run_case, model, store=store, workers=workers)
        base_reports.append(b)
        cand_reports.append(c)
        log(f"repeat {r + 1}: base {len(b.passed())}/{b.count()}, cand {len(c.passed())}/{c.count()}")
    return base_reports, cand_reports


def _cost(reports: Sequence[EvalReport]) -> dict[str, Any]:
    cs = [r.cost() for r in reports]
    return {
        "model_calls": sum(c.model_calls for c in cs),
        "input_tokens": sum(c.input_tokens for c in cs),
        "output_tokens": sum(c.output_tokens for c in cs),
        "tool_calls": sum(c.tool_calls for c in cs),
        "model_wall_s": round(sum(c.wall_s for c in cs), 3),
    }


def result_payload(
    meta: dict[str, Any],
    base_reports: Sequence[EvalReport],
    cand_reports: Sequence[EvalReport],
    splits: Sequence[str],
) -> dict[str, Any]:
    return {
        "type": "transfer",
        "meta": meta,
        "summaries": {s: summarize(base_reports, cand_reports, s).to_payload() for s in splits},
        "cost": {"base": _cost(base_reports), "cand": _cost(cand_reports)},
        "base_reports": [r.to_payload() for r in base_reports],
        "cand_reports": [r.to_payload() for r in cand_reports],
    }


def default_client(base_url: str, api_key: str, model_name: str) -> ModelClient:
    config = ModelConfig(base_url=base_url.rstrip("/"), model=model_name, api_key=api_key, timeout_s=300.0)
    return ModelClient(config)
