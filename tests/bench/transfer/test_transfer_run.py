from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import httpx
import pytest

from ratchet.agent.model import ScriptedModel, scripted_turn
from ratchet.agent.seed import seed_harness
from ratchet.bench.transfer import __main__ as cli
from ratchet.bench.transfer.render import margin_met, render
from ratchet.bench.transfer.run import (
    endpoint_info,
    harness_from,
    probe_tool_call,
    result_payload,
    run_transfer,
    trace_summary,
)
from ratchet.cases.registry import by_split
from ratchet.types import Cost, Event, Guard, Harness, Outcome, Trace


def candidate() -> Harness:
    base = seed_harness()
    guard = Guard("g1", "block", "def check(call, view, history):\n    return None\n", ["f1"])
    return Harness.build(
        parent=base.version, system_prompt=base.system_prompt, procedures=[], guards=[guard],
        tool_allowlist=base.tool_allowlist, created_by="improver",
    )  # fmt: skip


def fake_run_case(harness: Harness, case: Any, model: Any, store: Any, *, run_id: str) -> Trace:
    passed = bool(harness.guards) or case.kind == "benign_twin"
    if case.id.endswith("x"):
        raise RuntimeError("never")
    events = [Event(0, "model", {}, 0.0), Event(1, "final", {"reason": "answered"}, 0.0)]
    return Trace(
        run_id, case.id, harness.version, "m", events, Outcome(passed, [], "d", Cost(1, 10, 5, 0, 0.1))
    )


def test_harness_round_trip_and_digest_check() -> None:
    h = candidate()
    assert harness_from(asdict(h)) == h
    bad = {**asdict(h), "system_prompt": "changed"}
    with pytest.raises(ValueError):
        harness_from(bad)


def test_endpoint_info_reads_models_and_version() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.headers["authorization"] == "Bearer k"
        if req.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "m", "root": "org/m", "max_model_len": 4096}]})
        assert req.url.path == "/version"
        return httpx.Response(200, json={"version": "0.26.0"})

    info = endpoint_info("http://h:1/v1/", "k", transport=httpx.MockTransport(handler))
    assert info == {
        "base_url": "http://h:1/v1",
        "served": [{"id": "m", "root": "org/m", "max_model_len": 4096}],
        "server_version": "0.26.0",
    }

    def no_version(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/v1/models":
            return httpx.Response(200, json={"data": []})
        return httpx.Response(404)

    assert (
        endpoint_info("http://h:1/v1", "k", transport=httpx.MockTransport(no_version))["server_version"]
        is None
    )


def test_probe_tool_call_reports_parsed_call() -> None:
    ok = probe_tool_call(
        ScriptedModel([scripted_turn(tool_calls=[("get_weather", {"city": "Gainesville"})])])
    )
    assert ok["tool_call_ok"] and ok["tool_calls"] == [
        {"name": "get_weather", "args": {"city": "Gainesville"}}
    ]
    assert ok["input_tokens"] == 10 and ok["output_tokens"] == 5
    no = probe_tool_call(ScriptedModel([scripted_turn("sunny")]))
    assert not no["tool_call_ok"] and no["content"] == "sunny"


def test_run_transfer_pairs_fresh_repeats_and_renders() -> None:
    cases = by_split("heldout")[:4]
    base, cand = seed_harness(), candidate()
    logs: list[str] = []
    b, c = run_transfer(base, cand, cases, object(), fake_run_case, repeats=2, workers=2, log=logs.append)
    assert len(b) == len(c) == 2 and len(logs) == 4
    assert all(r.modes()["FRESH"] == 4 for r in [*b, *c])
    assert len({r.run_id for rep in [*b, *c] for r in rep.results}) == 16
    with pytest.raises(ValueError):
        run_transfer(base, cand, cases, object(), fake_run_case, repeats=0, workers=1)
    meta = {
        "model": "m", "endpoint": {"served": [{"id": "m", "root": "org/m", "max_model_len": 1}],
                                   "server_version": "0.26.0"},
        "base": {"version": base.version, "digest": base.digest},
        "cand": {"version": cand.version, "digest": cand.digest, "guards": 1, "procedures": 0},
        "suite_digest": "s", "splits": ["heldout"], "case_counts": {"heldout": 4}, "repeats": 2,
        "workers": 2, "started_at": "t0", "finished_at": "t1", "commit": "abc",
    }  # fmt: skip
    payload = result_payload(meta, b, c, ["heldout"])
    s = payload["summaries"]["heldout"]
    assert s["cand"]["strict_passes"] == 4
    assert payload["cost"]["cand"]["model_calls"] == 8
    md = render([(payload, "f" * 64)], {"Note": "x"})
    assert "| heldout | 4 |" in md and "- Note: x" in md and "transfer margin" in md
    with pytest.raises(ValueError):
        render([], {})


def test_margin_uses_base_noise() -> None:
    s = {"cand": {"strict_passes": 9}, "base": {"strict_passes": 6, "noise_d": 3}}
    assert margin_met(s) == (False, 3, 4)
    s["cand"]["strict_passes"] = 10
    assert margin_met(s) == (True, 4, 4)


def test_trace_summary() -> None:
    t = fake_run_case(candidate(), by_split("heldout")[0], None, None, run_id="r")
    s = trace_summary(t, 1.23456)
    assert s["passed"] and s["final_reason"] == "answered" and s["events"] == {"model": 1, "final": 1}
    assert s["wall_s"] == 1.235 and s["cost"]["input_tokens"] == 10


def test_cli_run_and_render_with_candidate_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cand_file = tmp_path / "cand.json"
    cand_file.write_text(json.dumps({"harness": asdict(candidate())}), encoding="utf-8")
    monkeypatch.delenv("RATCHET_MONGODB_URI", raising=False)
    monkeypatch.setattr(cli, "endpoint_info", lambda *_a, **_k: {"served": [], "server_version": None})
    monkeypatch.setattr("ratchet.agent.worker.run_case", fake_run_case)
    out = tmp_path / "res.json"
    rc = cli.main([
        "run", "--base-url", "http://h/v1", "--model", "m", "--candidate-file", str(cand_file),
        "--splits", "audit", "--repeats", "1", "--workers", "2", "--out", str(out),
    ])  # fmt: skip
    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    n = len(by_split("audit"))
    assert data["summaries"]["audit"]["cases"] == n and data["meta"]["trace_db"] is None
    md = tmp_path / "t.md"
    assert cli.main(["render", str(out), "--out", str(md), "--note", "K=V"]) == 0
    assert f"{n}/{n}" in md.read_text(encoding="utf-8")
    with pytest.raises(SystemExit):
        cli.main(["run", "--base-url", "u", "--model", "m", "--out", str(out)])
