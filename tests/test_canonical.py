"""`ratchet.canonical` is RFC 8785, and reads records written before it with the form they were signed in."""

from __future__ import annotations

import json
import math

import jcs_admit
import pytest

from ratchet.canonical import CANONICALIZATION, canonical_json, canonicalizer


def test_numbers_take_their_ecmascript_form() -> None:
    assert canonical_json([1.0, 0.0, -0.0, 100.0, 1e-7, 1e21, 123.456, 2**53]) == (
        b"[1,0,0,100,1e-7,1e+21,123.456,9007199254740992]"
    )


def test_members_sort_by_utf16_code_units() -> None:
    # U+1F600 is the surrogate pair D83D DE00: before U+FFFF in UTF-16, after it by code point.
    value = {chr(0xFFFF): 2, chr(0x1F600): 1, "a": 0}
    want = ("{" + '"a":0,"' + chr(0x1F600) + '":1,"' + chr(0xFFFF) + '":2}').encode("utf-8")
    assert canonical_json(value) == want


def test_rfc8785_section_3_2_2_example() -> None:
    text = "".join(map(chr, [0x20AC, 0x24, 0x0F, 0x0A, 0x41, 0x27, 0x42, 0x22, 0x5C, 0x5C, 0x22, 0x2F]))
    value = {
        "numbers": [333333333.33333329, 1e30, 4.50, 2e-3, 0.000000000000000000000000001],
        "string": text,
        "literals": [None, True, False],
    }
    want = (
        '{"literals":[null,true,false],"numbers":[333333333.3333333,1e+30,4.5,0.002,1e-27],'
        '"string":"' + chr(0x20AC) + "$" + "\\u000f\\nA'B" + '\\"' + "\\\\\\\\" + '\\"' + '/"}'
    )
    assert canonical_json(value) == want.encode("utf-8")


def test_output_is_canonical_and_stable() -> None:
    once = canonical_json({"b": [1, {"d": 1.5, "c": "x"}], "a": None})
    assert jcs_admit.is_canonical(once)
    assert canonical_json(json.loads(once)) == once


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_numbers_are_refused(bad: float) -> None:
    with pytest.raises(ValueError):
        canonical_json({"x": bad})


def test_a_record_written_before_rfc8785_is_read_with_sorted_keys_json() -> None:
    value = {"b": 1.0, "a": [2.5, 1e-07]}
    assert canonicalizer(None)(value) == b'{"a":[2.5,1e-07],"b":1.0}'
    assert canonicalizer(CANONICALIZATION)(value) == b'{"a":[2.5,1e-7],"b":1}'
    with pytest.raises(ValueError, match="unknown canonicalization"):
        canonicalizer("json")
