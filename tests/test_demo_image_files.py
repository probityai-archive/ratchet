"""The demo image's health check and the CI job that runs it on both architectures."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_the_image_declares_a_healthcheck_that_calls_the_store_script() -> None:
    dockerfile = (ROOT / "docker/demo.Dockerfile").read_text(encoding="utf-8")
    assert re.search(r'HEALTHCHECK[^\n]*\\\n\s*CMD \["/usr/local/bin/ratchet-store", "health"\]', dockerfile)


def test_the_store_script_has_a_health_command_that_starts_nothing() -> None:
    script = (ROOT / "docker/demo/ratchet-store").read_text(encoding="utf-8")
    health = re.search(r"\n    health\)\n(.*?)\n        ;;", script, re.S)
    assert health is not None
    assert health.group(1).split() == ["probe"]


def test_the_reviewer_check_waits_for_the_healthcheck_before_running_the_demo() -> None:
    check = (ROOT / "docker/demo/check.sh").read_text(encoding="utf-8")
    assert "State.Health.Status" in check
    assert check.index('step 0 "^healthy$" wait_healthy') < check.index("ratchet demo double-refund")


def test_ci_runs_the_demo_image_on_x86_and_arm() -> None:
    ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    job = ci[ci.index("\n  demo-image:") :]
    assert "arch: amd64" in job and "runner: ubuntu-24.04\n" in job
    assert "arch: arm64" in job and "runner: ubuntu-24.04-arm" in job
    assert "runs-on: ${{ matrix.runner }}" in job


def test_the_reviewer_check_requires_every_check_recomputed_on_both_receipts() -> None:
    check = (ROOT / "docker/demo/check.sh").read_text(encoding="utf-8")
    verifies = [
        i for i, line in enumerate(check.splitlines()) if "ratchet verify" in line and "--pubkey-file" in line
    ]
    lines = check.splitlines()
    assert len(verifies) == 2
    assert all(lines[i + 1] == 'also "RECOMPUTED \\\\(10 of 10\\\\)"' for i in verifies)
