from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from ratchet_cases.types import sha256_hex

from ratchet.canonical import canonical_json
from ratchet.improve import endpoints as ep

DET = {"endpoints": ["http://a/v1", "http://b/v1"], "model": "m", "temperature": 0.0, "seed": 7,
       "batch_size": 1, "prefix_cache": False}  # fmt: skip


def _write(tmp_path: Path, doc: Any, name: str = "ep.json") -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(doc, indent=3))
    return p


def test_the_digest_is_over_canonical_json_so_whitespace_does_not_change_it(tmp_path: Path) -> None:
    loaded = ep.load(_write(tmp_path, DET))
    assert loaded.digest == sha256_hex(canonical_json(DET))
    assert ep.load(tmp_path / "ep.json", loaded.digest).doc == DET
    assert loaded.model_config() == {**DET, "endpoint_config_digest": loaded.digest}
    assert ep.resolve(loaded.reference(tmp_path / "ep.json")).digest == loaded.digest


def test_a_changed_file_is_refused_against_the_named_digest(tmp_path: Path) -> None:
    path = _write(tmp_path, DET)
    named = ep.load(path).digest
    _write(tmp_path, {**DET, "seed": 8})
    with pytest.raises(ep.EndpointsError, match="is not the named"):
        ep.resolve({"file": str(path), "digest": named})


@pytest.mark.parametrize(
    "doc, match",
    [
        ([], "not a JSON object"),
        ({"model": "m"}, "endpoints"),
        ({"endpoints": [], "model": "m"}, "endpoints"),
        ({"endpoints": ["u"], "model": ""}, "model"),
        ({"endpoints": ["u"], "model": "m", "seed": "7"}, "seed must"),
        ({"endpoints": ["u"], "model": "m", "seed_base": True}, "seed_base must"),
        ({"endpoints": ["u"], "model": "m", "api_key": "sk"}, "names a key"),
    ],
)
def test_malformed_configurations_are_refused(doc: Any, match: str) -> None:
    with pytest.raises(ep.EndpointsError, match=match):
        ep.parse(doc)


def test_unreadable_files_and_bad_references_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ep.EndpointsError, match="cannot read"):
        ep.load(tmp_path / "missing.json")
    with pytest.raises(ep.EndpointsError, match="names a file and a digest"):
        ep.resolve({"file": "x"})


class _Client:
    def __init__(self, url: str, seed: int) -> None:
        self.url, self.seed, self.closed = url, seed, False

    def close(self) -> None:
        self.closed = True


def test_round_robin_sends_every_case_to_the_next_endpoint_with_the_fixed_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = ep.parse(DET)
    monkeypatch.setattr(ep.Endpoints, "client", lambda self, url, seed: _Client(url, seed))
    seen: list[tuple[str, int]] = []

    def run_case(h: Any, case: Any, model: _Client, store: Any, *, run_id: str) -> Any:
        seen.append((model.url, model.seed))
        return run_id

    rr = ep.RoundRobin(loaded, run_case)
    for i in range(3):
        rr(None, None, None, None, run_id=f"r{i}")
    assert seen == [("http://b/v1", 7), ("http://a/v1", 7), ("http://b/v1", 7)]
    rr.close()
    assert all(c.closed for c in rr.clients)
    with pytest.raises(ep.EndpointsError, match="names a seed"):
        ep.RoundRobin(ep.parse({"endpoints": ["u"], "model": "m"}), run_case)


def test_per_repetition_clients_share_one_seed_schedule_and_follow_the_replicas_file(tmp_path: Path) -> None:
    loaded = ep.parse({"endpoints": ["http://a/v1"], "model": "m", "seed_base": 1000})
    replicas = tmp_path / "replicas"
    pr = ep.PerRepetition(loaded, replicas, make=_Client)
    url, c = pr.client_for(0, 3)
    assert (url, c.seed) == ("http://a/v1", 1003) and pr.client_for(5, 3)[1] is c
    replicas.write_text("http://r1/v1\nhttp://r2/v1\n")
    assert [pr.client_for(i, 0)[0] for i in range(2)] == ["http://r1/v1", "http://r2/v1"]
    replicas.write_text("\n")
    assert pr.live() == ["http://a/v1"]
    pr.close()
    with pytest.raises(ep.EndpointsError, match="seed_base"):
        ep.PerRepetition(ep.parse(DET))


def test_a_real_client_takes_its_key_from_the_named_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EP_TEST_KEY", "k1")
    loaded = ep.parse({**DET, "api_key_env": "EP_TEST_KEY"})
    client = loaded.client("http://a/v1", 7)
    try:
        assert client._config.api_key == "k1" and client._config.temperature == 0.0
    finally:
        client.close()


def test_an_unseeded_round_robin_needs_no_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ep.Endpoints, "client", lambda self, url, seed: _Client(url, seed))
    rr = ep.RoundRobin(
        ep.parse({"endpoints": ["u1", "u2"], "model": "m"}), lambda *a, **k: None, seeded=False
    )
    assert [(c.url, c.seed) for c in rr.clients] == [("u1", None), ("u2", None)]
