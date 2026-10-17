from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import httpx
import pytest
from ratchet_cases.cases.registry import by_split
from ratchet_cases.types import Cost, Event, Outcome, Trace

from ratchet.agent.model import ScriptedModel, scripted_turn
from ratchet.agent.seed import seed_harness
from ratchet.bench.transfer import __main__ as cli
from ratchet.bench.transfer.render import margin_met, render
from ratchet.bench.transfer.run import (
    endpoint_info,
    harness_from,
    persist,
    probe_tool_call,
    result_payload,
    run_transfer,
    trace_docs,
    trace_summary,
)
from ratchet.types import Guard, Harness


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
    only = render([], {"Status": "waiting"}, [(payload, "a" * 64)])
    assert "No transfer run" in only and "## A/A control: m, checkpoint default sampling" in only


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
    assert cli.main(["render", str(out), "--out", str(md), "--note", "K=V", "--control", str(out)]) == 0
    text = md.read_text(encoding="utf-8")
    assert f"{n}/{n}" in text and "## A/A control: m, temperature 0.0" in text
    with pytest.raises(SystemExit):
        cli.main(["run", "--base-url", "u", "--model", "m", "--out", str(out)])


class FakeCollection:
    def __init__(self) -> None:
        self.docs: dict[str, dict[str, Any]] = {}

    def bulk_write(self, ops: list[Any], ordered: bool = True) -> None:
        for op in ops:
            self.docs[op._filter["_id"]] = op._doc

    def replace_one(self, flt: dict[str, Any], doc: dict[str, Any], upsert: bool = False) -> None:
        self.docs[flt["_id"]] = doc


def test_persist_writes_every_trace_and_one_eval() -> None:
    cases = by_split("audit")[:3]
    b, c = run_transfer(seed_harness(), candidate(), cases, object(), fake_run_case, repeats=1, workers=1)
    payload = {"type": "transfer", "summaries": {}, "base_reports": [], "cand_reports": []}
    db = {"transfer_traces": FakeCollection(), "transfer_evals": FakeCollection()}
    out = persist(db, payload, trace_docs([*b, *c]))
    assert out["traces"] == 6 and len(db["transfer_traces"].docs) == 6
    (ev,) = db["transfer_evals"].docs.values()
    assert ev["_id"] == out["eval_id"] and ev["type"] == "transfer"
    assert all(d["eval_id"] == out["eval_id"] for d in db["transfer_traces"].docs.values())
    empty = persist(db, payload, [])
    assert empty["traces"] == 0


def test_derived_on_prefers_the_frozen_evaluator() -> None:
    assert cli.derived_on({"evaluator": {"model_config": {"model": "a"}}, "context": {"model": "b"}}) == "a"
    assert cli.derived_on({"context": {"model": "b"}}) == "b"
    assert cli.derived_on({}) is None


class FakeStore:
    def __init__(self, decisions: list[dict[str, Any]], harness: Harness | None) -> None:
        self._d = decisions
        self._h = harness

    def decisions(self, flt: dict[str, Any]) -> list[dict[str, Any]]:
        return [d for d in self._d if d.get("admitted") == flt.get("admitted")]

    def get_harness(self, version: str) -> Harness | None:
        return self._h

    def close(self) -> None:
        return None


def test_candidate_from_store_takes_latest_admitted(tmp_path: Path) -> None:
    import argparse

    h = candidate()
    decisions = [
        {"_id": "d1", "candidate": "old", "admitted": True},
        {"_id": "d2", "candidate": h.version, "admitted": True, "evaluator": {"version": "e1",
         "model_config": {"model": "gpt-5"}}},
        {"_id": "d3", "candidate": "rej", "admitted": False},
    ]  # fmt: skip
    args = argparse.Namespace(candidate_file=None, candidate="latest")
    got, extra = cli._candidate(args, FakeStore(decisions, h))
    assert got == h and extra["decision_id"] == "d2" and extra["derived_on"] == "gpt-5"
    assert extra["admitted_decisions_in_store"] == 2 and extra["evaluator_version"] == "e1"
    args.candidate = "missing"
    with pytest.raises(SystemExit):
        cli._candidate(args, FakeStore(decisions, h))
    args.candidate = h.version
    with pytest.raises(SystemExit):
        cli._candidate(args, FakeStore(decisions, None))
    with pytest.raises(SystemExit):
        cli._candidate(args, None)


class _Ctx:
    def __init__(self, model: ScriptedModel) -> None:
        self.model = model

    def __enter__(self) -> ScriptedModel:
        return self.model

    def __exit__(self, *exc: object) -> None:
        return None


def test_cli_probe_writes_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    turns = [scripted_turn(tool_calls=[("get_weather", {"city": "x"})])]
    monkeypatch.setattr(cli, "endpoint_info", lambda *_a, **_k: {"served": [], "server_version": "v"})
    monkeypatch.setattr(cli, "default_client", lambda *_a: _Ctx(ScriptedModel(turns)))
    monkeypatch.setattr(
        "ratchet.agent.worker.run_case",
        lambda h, case, model: fake_run_case(h, case, model, None, run_id="p"),
    )
    out = tmp_path / "probe.json"
    assert (
        cli.main(["probe", "--base-url", "u", "--model", "m", "--case", "ordinary.v1", "--out", str(out)])
        == 0
    )
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["tool_call"]["tool_call_ok"] and data["run_case"]["case_id"] == "ordinary.v1"
    monkeypatch.setattr(cli, "default_client", lambda *_a: _Ctx(ScriptedModel([scripted_turn("no")])))
    assert cli.main(["probe", "--base-url", "u", "--model", "m", "--case", "ordinary.v1"]) == 1


def test_cli_run_persists_when_a_store_uri_is_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import pymongo

    import ratchet.store.atlas

    written: dict[str, Any] = {}

    class FakeClient:
        def __init__(self, uri: str, **_kw: Any) -> None:
            written["uri"] = uri

        def __getitem__(self, name: str) -> dict[str, FakeCollection]:
            written["db"] = name
            written["colls"] = {"transfer_traces": FakeCollection(), "transfer_evals": FakeCollection()}
            return written["colls"]  # type: ignore[no-any-return]

        def close(self) -> None:
            written["closed"] = True

    h = candidate()
    monkeypatch.setenv("RATCHET_MONGODB_URI", "mongodb://fake")
    monkeypatch.setattr(pymongo, "MongoClient", FakeClient)
    stored = [{"_id": "d", "candidate": h.version, "admitted": True}]
    monkeypatch.setattr(ratchet.store.atlas, "Store", lambda *_a: FakeStore(stored, h))
    monkeypatch.setattr(cli, "endpoint_info", lambda *_a, **_k: {"served": [], "server_version": None})
    monkeypatch.setattr("ratchet.agent.worker.run_case", fake_run_case)
    out = tmp_path / "r.json"
    args = [
        "run",
        "--base-url",
        "u",
        "--model",
        "m",
        "--splits",
        "audit",
        "--repeats",
        "1",
        "--out",
        str(out),
    ]
    assert cli.main(args) == 0
    n = len(by_split("audit"))
    assert written["db"] == "ratchet_heldout" and written["closed"]
    assert len(written["colls"]["transfer_traces"].docs) == 2 * n
    assert json.loads(out.read_text(encoding="utf-8"))["meta"]["trace_db"] == "ratchet_heldout"


def test_single_call_model_keeps_first_call_only() -> None:
    from dataclasses import replace

    from ratchet.bench.transfer.run import SingleCallModel

    multi = scripted_turn(tool_calls=[("a", {"x": 1}), ("b", {})])
    multi = replace(multi, arg_errors={"c0": "bad", "c1": "bad"})
    single = scripted_turn(tool_calls=[("a", {})])
    model = SingleCallModel(ScriptedModel([multi, single, scripted_turn("done")], name="inner"))
    assert model.name == "inner"
    t = model.complete([], [])
    assert [c.name for c in t.tool_calls] == ["a"] and len(t.message["tool_calls"]) == 1
    assert t.arg_errors == {"c0": "bad"} and t.output_tokens == multi.output_tokens
    assert model.complete([], []) is single
    assert model.complete([], []).content == "done"
    assert model.truncated_turns == 1


def test_cli_run_single_tool_call_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cand_file = tmp_path / "cand.json"
    cand_file.write_text(json.dumps(asdict(candidate())), encoding="utf-8")
    monkeypatch.delenv("RATCHET_MONGODB_URI", raising=False)
    monkeypatch.setattr(cli, "endpoint_info", lambda *_a, **_k: {"served": [], "server_version": None})
    monkeypatch.setattr("ratchet.agent.worker.run_case", fake_run_case)
    out = tmp_path / "res.json"
    args = ["run", "--base-url", "http://h/v1", "--model", "m", "--candidate-file", str(cand_file)]
    args += ["--splits", "audit", "--repeats", "1", "--single-tool-call", "--out", str(out)]
    assert cli.main(args) == 0
    meta = json.loads(out.read_text(encoding="utf-8"))["meta"]
    assert meta["single_tool_call"] is True and meta["truncated_turns"] == 0


def test_candidate_file_carries_its_decision(tmp_path: Path) -> None:
    import argparse

    h = candidate()
    f = tmp_path / "c.json"
    decision = {
        "_id": "d9",
        "decided_at": "t",
        "evaluator": {"version": "e", "model_config": {"model": "luna"}},
    }
    f.write_text(
        json.dumps({"harness": asdict(h), "decision": decision, "store": "atlas/db"}), encoding="utf-8"
    )
    got, extra = cli._candidate(argparse.Namespace(candidate_file=str(f), candidate="latest"), None)
    assert got == h
    assert extra["decision_id"] == "d9" and extra["derived_on"] == "luna" and extra["store"] == "atlas/db"


def test_sampling_follows_flags_then_derivation_then_default() -> None:
    from ratchet.bench.transfer.run import sampling_for

    d = {"evaluator": {"model_config": {"temperature": 0.0, "seed": 7}}}
    assert sampling_for(None, None, d) == {
        "temperature": 0.0, "temperature_source": "derivation", "seed": 7, "seed_source": "derivation",
    }  # fmt: skip
    assert sampling_for(0.5, 3, d)["temperature_source"] == "flag"
    assert sampling_for(0.5, 3, d)["seed"] == 3
    assert sampling_for(None, None, None) == {
        "temperature": 0.0, "temperature_source": "default", "seed": None, "seed_source": "none",
    }  # fmt: skip


def test_seed_transport_adds_seed_to_chat_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    from ratchet.bench.transfer.run import SeedTransport, default_client

    seen: list[httpx.Request] = []

    def fake_handle(_self: httpx.HTTPTransport, req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(200)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", fake_handle)
    t = SeedTransport(7)
    t.handle_request(httpx.Request("POST", "http://h/v1/chat/completions", json={"model": "m"}))
    t.handle_request(httpx.Request("GET", "http://h/v1/models"))
    assert json.loads(seen[0].content) == {"model": "m", "seed": 7}
    assert seen[0].headers["content-length"] == str(len(seen[0].content))
    assert seen[1].method == "GET"
    with default_client("http://h/v1", "k", "m", temperature=0.0, seed=1) as c:
        assert c.name == "m"
    with default_client("http://h/v1", "k", "m") as c:
        assert c.name == "m"


def test_store_retries_and_keeps_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import pymongo
    from pymongo.errors import AutoReconnect

    calls = {"n": 0}

    class Flaky:
        def __init__(self, uri: str, **_kw: Any) -> None:
            pass

        def __getitem__(self, name: str) -> dict[str, FakeCollection]:
            calls["n"] += 1
            if calls["n"] < 2:
                raise AutoReconnect("no primary")
            return {"transfer_traces": FakeCollection(), "transfer_evals": FakeCollection()}

        def close(self) -> None:
            pass

    cand_file = tmp_path / "cand.json"
    cand_file.write_text(json.dumps(asdict(candidate())), encoding="utf-8")
    monkeypatch.setenv("RATCHET_MONGODB_URI", "mongodb://fake")
    monkeypatch.setattr(pymongo, "MongoClient", Flaky)
    monkeypatch.setattr(cli, "STORE_BACKOFF_S", 0.0)
    monkeypatch.setattr(cli, "endpoint_info", lambda *_a, **_k: {"served": [], "server_version": None})
    monkeypatch.setattr("ratchet.agent.worker.run_case", fake_run_case)
    out = tmp_path / "r.json"
    args = ["run", "--base-url", "u", "--model", "m", "--candidate-file", str(cand_file), "--splits", "audit"]
    assert cli.main([*args, "--repeats", "1", "--out", str(out)]) == 0
    assert calls["n"] == 2 and out.exists()
    lines = out.with_suffix(".traces.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 * len(by_split("audit"))
    calls["n"] = -10
    monkeypatch.setattr(cli, "STORE_ATTEMPTS", 2)
    assert cli.main(["store", str(out)]) == 3
    calls["n"] = 5
    assert cli.main(["store", str(out)]) == 0
    monkeypatch.delenv("RATCHET_MONGODB_URI")
    with pytest.raises(SystemExit):
        cli.main(["store", str(out)])


def _chat_handler(vary: bool) -> Any:
    count = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        count["n"] += 1
        is_target = body["messages"][-1]["content"].startswith("What is the weather")
        args = '{"city": "G"}' if not (vary and is_target and count["n"] % 2) else '{"city": "H"}'
        msg = {"role": "assistant", "content": None, "reasoning_content": "think",
               "tool_calls": [{"id": f"id{count['n']}", "type": "function",
                               "function": {"name": "get_weather", "arguments": args}}]}  # fmt: skip
        return httpx.Response(200, json={"choices": [{"message": msg, "finish_reason": "tool_calls"}],
                                         "usage": {"completion_tokens": 3}})  # fmt: skip

    return handler


def test_determinism_probe_detects_differences() -> None:
    from ratchet.bench.transfer.run import determinism_probe

    same = determinism_probe("http://h/v1", "k", "m", repeats=3, background=2,
                             transport=httpx.MockTransport(_chat_handler(False)))  # fmt: skip
    assert same["identical"] and same["distinct_outputs"] == 1 and same["seed"] == 7
    diff = determinism_probe("http://h/v1", "k", "m", repeats=4, background=1, seed=None,
                             transport=httpx.MockTransport(_chat_handler(True)))  # fmt: skip
    assert not diff["identical"] and diff["distinct_outputs"] == 2


def test_cli_determinism_uses_a_case_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[dict[str, Any]] = []

    def fake_probe(*_a: Any, **kw: Any) -> dict[str, Any]:
        seen.append(kw)
        return {"identical": True, "outputs": [{"x": 1}]}

    monkeypatch.setattr(cli, "determinism_probe", fake_probe)
    out = tmp_path / "d.json"
    args = ["determinism", "--base-url", "u", "--model", "m", "--case", "ordinary.v1", "--out", str(out)]
    assert cli.main(args) == 0
    assert seen[0]["messages"][1]["content"].startswith("Ticket ") and len(seen) == 2
    assert any(t["function"]["name"] == "read_procedure" for t in seen[0]["tools"])
    monkeypatch.setattr(cli, "determinism_probe", lambda *_a, **_k: {"identical": False, "outputs": [{}]})
    assert cli.main(["determinism", "--base-url", "u", "--model", "m"]) == 1
