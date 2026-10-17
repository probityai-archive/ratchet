"""RFC 8785 canonical JSON for every record Ratchet signs or digests, through the `probityai.jcs` binding.

`canonical_json` admits the JSON text Python writes and returns its RFC 8785 form from the `jcs-admit` crate:
members sorted by UTF-16 code units, numbers in ECMAScript form (`1.0` is `1`, `1e-07` is `1e-7`), no
whitespace. NaN and infinities are refused, as RFC 8785 requires. An integer beyond 2**53 is written as the
nearest double, so a value that needs every digit is carried as a string.

Every record that carries a digest names the canonicaliser it was computed with (`CANONICALIZATION`). A
record written before RFC 8785 names none: its digests and signatures are over Python's sorted-keys JSON, and
a digest can be recomputed only with the function that produced it, so `canonicalizer(None)` returns that
form, for reading those records alone. Nothing new is written with it.

The case suite (`ratchet_cases`) digests its own case definitions and world states with sorted-keys JSON and
keeps no runtime dependency; those digests reach Ratchet as values and are never recomputed here.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from probityai import jcs

CANONICALIZATION = "rfc8785"


def canonical_json(obj: Any) -> bytes:
    """RFC 8785 canonical JSON bytes of `obj`, the input to every digest and signature."""
    text = json.dumps(obj, ensure_ascii=False, allow_nan=False).encode("utf-8")
    return jcs.admit_with(text, max_bytes=None)


def _sorted_keys_json(obj: Any) -> bytes:
    """Python's sorted-keys JSON, which every record written before RFC 8785 was digested and signed over."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def canonicalizer(name: str | None) -> Callable[[Any], bytes]:
    """The canonicaliser a record declares: `rfc8785`, or `None` for a record written before RFC 8785."""
    if name == CANONICALIZATION:
        return canonical_json
    if name is None:
        return _sorted_keys_json
    raise ValueError(f"unknown canonicalization {name!r}")
