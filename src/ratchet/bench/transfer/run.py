"""Run the transfer experiment and the endpoint proofs against an OpenAI-compatible endpoint.

Every run is FRESH: `evaluate_many` gives each run a new run id, and the two harnesses share one bounded
pool per repeat, so both arms see the same endpoint load at the same time (a paired design). The result
file records the endpoint's own account of what it served (`/v1/models`, `/version`), the harness versions
and digests, the case-suite digest and every per-case result, so the report can be rebuilt from it alone.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

import httpx

from ratchet.agent.model import ChatModel, Message, ModelClient, ModelConfig, ModelTurn, ToolSpec
from ratchet.bench.transfer.analysis import summarize
from ratchet.improve.evaluate import EvalReport, RunCase, evaluate_many
from ratchet.types import Guard, Harness, Procedure, Trace, canonical_json, sha256_hex, to_jsonable

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


class SingleCallModel:
    """Keep only the first tool call of a turn, exactly what vLLM does for `parallel_tool_calls: false`.

    Llama 3.x's own chat template refuses an assistant message carrying more than one tool call ("This model
    only supports single tool-calls at once!"), yet the llama3_json parser will return several when the model
    writes them, so the next request of that run fails with HTTP 400. Applied to both arms of a comparison,
    so the pairing is unaffected; the count of truncated turns is recorded.
    """

    def __init__(self, inner: ChatModel) -> None:
        self._inner = inner
        self._lock = threading.Lock()
        self.truncated_turns = 0

    @property
    def name(self) -> str:
        return self._inner.name

    def complete(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> ModelTurn:
        turn = self._inner.complete(messages, tools)
        if len(turn.tool_calls) <= 1:
            return turn
        with self._lock:
            self.truncated_turns += 1
        first = turn.tool_calls[0]
        raw = turn.message.get("tool_calls") or []
        message = {**turn.message, "tool_calls": raw[:1]}
        errors = {k: v for k, v in turn.arg_errors.items() if k == first.call_id}
        return replace(turn, message=message, tool_calls=[first], arg_errors=errors)


def default_client(base_url: str, api_key: str, model_name: str) -> ModelClient:
    config = ModelConfig(base_url=base_url.rstrip("/"), model=model_name, api_key=api_key, timeout_s=300.0)
    return ModelClient(config)


def persist(db: Any, payload: dict[str, Any], reports: Sequence[EvalReport]) -> dict[str, Any]:
    """Write every trace and the summary into an evaluator-only database (held-out traces never go where
    the improver can read). Plain inserts: no schema or search-index side effects on a shared cluster."""
    summary = {k: v for k, v in payload.items() if k not in ("base_reports", "cand_reports")}
    eval_id = "transfer-" + sha256_hex(canonical_json(to_jsonable(summary)))[:16]
    docs = []
    for rep in reports:
        for trace in rep.traces:
            body = to_jsonable(trace)
            docs.append({**body, "_id": f"{eval_id}:{trace.run_id}", "eval_id": eval_id})
    if docs:
        db["transfer_traces"].insert_many(docs, ordered=False)
    db["transfer_evals"].replace_one(
        {"_id": eval_id}, {**to_jsonable(payload), "_id": eval_id, "stored_at": now_iso()}, upsert=True
    )
    return {"eval_id": eval_id, "traces": len(docs)}
