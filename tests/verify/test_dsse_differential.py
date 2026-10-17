"""The DSSE binding and the independent Python reader agree on every envelope, honest, malformed or forged."""

from __future__ import annotations

import ast
import base64
import json
from collections.abc import Callable
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from ratchet.verify import dsse

from . import python_dsse

TYPE = "application/vnd.ratchet.decision+json"
BODY = b'{"admitted":true,"candidate":"abc"}'


def _edit(raw: bytes, change: Callable[[dict[str, object]], None]) -> bytes:
    doc = json.loads(raw)
    change(doc)
    return json.dumps(doc).encode()


def _flip_sig(doc: dict[str, object]) -> None:
    entry = doc["signatures"][0]  # type: ignore[index]
    sig = bytearray(base64.b64decode(entry["sig"]))
    sig[0] ^= 1
    entry["sig"] = base64.b64encode(bytes(sig)).decode()


def _swap_payload(doc: dict[str, object]) -> None:
    doc["payload"] = base64.b64encode(b'{"admitted":false,"candidate":"abc"}').decode()


def _swap_type(doc: dict[str, object]) -> None:
    doc["payloadType"] = "application/vnd.ratchet.human-approval+json"


def _rename_keyid(doc: dict[str, object]) -> None:
    doc["signatures"][0]["keyid"] = "0" * 64  # type: ignore[index]


MUTATIONS = {
    "honest": lambda doc: None,
    "flipped signature": _flip_sig,
    "payload swapped": _swap_payload,
    "type swapped": _swap_type,
    "keyid renamed": _rename_keyid,
}


def _binding(raw: bytes, trusted: dict[str, object]) -> tuple[str | None, str | None]:
    try:
        result = dsse.verify_signature(dsse.parse(raw), trusted)  # type: ignore[arg-type]
    except dsse.EnvelopeError:
        return None, "malformed"
    return result.keyid, result.reason


def _python(raw: bytes, trusted: dict[str, object]) -> tuple[str | None, str | None]:
    try:
        result = python_dsse.verify_signature(python_dsse.parse(raw), trusted)  # type: ignore[arg-type]
    except python_dsse.EnvelopeError:
        return None, "malformed"
    return result.keyid, result.reason


@pytest.mark.parametrize("signed_by", ["binding", "python"])
@pytest.mark.parametrize("mutation", sorted(MUTATIONS))
def test_both_readers_reach_the_same_result(signed_by: str, mutation: str) -> None:
    key = Ed25519PrivateKey.generate()
    raw = dsse.sign(TYPE, BODY, [key]) if signed_by == "binding" else python_dsse.sign(TYPE, BODY, key)
    raw = _edit(raw, MUTATIONS[mutation])
    trusted = {dsse.keyid_of(key.public_key()): key.public_key()}
    assert _binding(raw, trusted) == _python(raw, trusted)
    if mutation == "honest":
        assert _binding(raw, trusted) == (dsse.keyid_of(key.public_key()), None)


def test_both_sign_the_same_bytes() -> None:
    key = Ed25519PrivateKey.generate()  # Ed25519 is deterministic, so the two envelopes are equal bytes
    assert dsse.sign(TYPE, BODY, [key]) == python_dsse.sign(TYPE, BODY, key)
    assert dsse.pae(TYPE, BODY) == python_dsse.pae(TYPE, BODY)


MALFORMED = [
    b"[]",
    b"{",
    b'{"payloadType":"t","payload":"!","signatures":[{"keyid":"k","sig":"AA=="}]}',
    b'{"payloadType":"t","payload":"AA==","signatures":[]}',
]


@pytest.mark.parametrize("raw", MALFORMED)
def test_both_refuse_the_same_malformed_envelopes(raw: bytes) -> None:
    assert _binding(raw, {}) == _python(raw, {}) == (None, "malformed")


def test_threshold_needs_two_distinct_trusted_keys() -> None:
    a, b = Ed25519PrivateKey.generate(), Ed25519PrivateKey.generate()
    trusted = {dsse.keyid_of(k.public_key()): k.public_key() for k in (a, b)}
    both = dsse.parse(dsse.sign(TYPE, BODY, [a, b]))
    one = dsse.parse(dsse.sign(TYPE, BODY, [a]))
    assert set(dsse.verify_signature(both, trusted, 2).keyids) == set(trusted)
    assert dsse.verify_signature(one, trusted, 2).reason == "untrusted_key"
    forged = dsse.parse(_edit(dsse.sign(TYPE, BODY, [a, b]), _flip_sig))
    assert dsse.verify_signature(forged, trusted, 2).reason == "signature_invalid"


def _imports(path: Path) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module:
            out |= {f"{node.module}.{a.name}" for a in node.names}
        elif isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
    return out


def test_the_python_reader_shares_nothing_with_the_binding() -> None:
    """The oracle reads envelopes with the standard library and `cryptography` alone."""
    imported = _imports(Path(python_dsse.__file__))
    assert {m for m in imported if m.startswith("ratchet")} == {"ratchet_cases.types.sha256_hex"}
    stdlib = {"__future__", "base64", "binascii", "json", "collections", "dataclasses", "ratchet_cases"}
    assert {m.split(".")[0] for m in imported} - stdlib == {"cryptography"}


def test_the_package_dsse_module_leaves_every_signature_to_the_crate() -> None:
    imported = _imports(Path(dsse.__file__))
    assert "jcs_admit.dsse" in imported
    assert "cryptography.exceptions.InvalidSignature" not in imported
