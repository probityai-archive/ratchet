"""The release contract: one version, one suite digest, recorded before release and checked on import."""

from __future__ import annotations

import json
import re
from pathlib import Path

import ratchet_cases
from ratchet_cases.cases.registry import DIGESTS_PATH, released_digests

PACKAGE = Path(__file__).resolve().parents[1]


def test_the_shipped_cases_match_the_digest_recorded_for_this_version() -> None:
    assert ratchet_cases.released_digest(ratchet_cases.__version__) == ratchet_cases.SUITE_DIGEST


def test_every_recorded_digest_is_a_sha256_and_versions_never_repeat_a_digest_change_silently() -> None:
    digests = released_digests()
    assert digests and all(len(d) == 64 and int(d, 16) >= 0 for d in digests.values())
    assert list(json.loads(DIGESTS_PATH.read_text(encoding="utf-8"))) == list(digests)


def test_the_package_has_no_runtime_dependencies_and_supports_python_310() -> None:
    text = (PACKAGE / "pyproject.toml").read_text(encoding="utf-8")
    assert re.search(r"^dependencies = \[\]$", text, re.M)
    assert re.search(r'^requires-python = ">=3\.10"$', text, re.M)
