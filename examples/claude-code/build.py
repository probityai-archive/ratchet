"""Build the worked example: prove the guard on transcripts, sign the decision, export the hook.

    uv run python examples/claude-code/build.py

1. Replay `no_force_push_main.py` over every session in `transcripts/` (synthetic, invented for this
   example) through the sandboxed guard runtime, and require that it fires on exactly the tool calls in
   `expected_fires.txt` with no guard error.
2. Record the result as a Ratchet decision and sign it (DSSE, Ed25519). The key comes from
   RATCHET_SIGNING_KEY when set; otherwise a fresh key is made for this build and only its public half is
   kept, in `public_key.txt`.
3. Export the admitted guard as a Claude Code PreToolUse hook in `hooks/`, plus the `settings.json`
   fragment that installs it.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from ratchet.agent.guards import GuardRuntime
from ratchet.export.claude_code import export_hook, settings_entry
from ratchet.export.replay_transcripts import discover, replay_transcripts
from ratchet.improve.admit import Check, Decision
from ratchet.improve.receipt import SIGNING_KEY_ENV, Signer, public_key_bytes, receipt
from ratchet.types import Guard, Harness

HERE = Path(__file__).resolve().parent
GUARD_ID = "no-force-push-main"
HOOK = HERE / "hooks" / "ratchet_no_force_push_main.py"


def main() -> int:
    guard = Guard(
        GUARD_ID,
        "Never force-push to, or delete, main or master.",
        (HERE / "no_force_push_main.py").read_text(encoding="utf-8"),
    )
    expected = (HERE / "expected_fires.txt").read_text(encoding="utf-8").split()
    with GuardRuntime(workers=2) as runtime:
        report = replay_transcripts(guard, discover([HERE / "transcripts"]), runtime, expect=expected)
    (HERE / "replay.json").write_text(json.dumps(report.to_payload(), indent=2) + "\n", encoding="utf-8")
    if not report.passed:
        print(json.dumps(report.to_payload(), indent=2), file=sys.stderr)
        return 1

    parent = Harness.build(
        parent=None, system_prompt="", procedures=[], guards=[], tool_allowlist=[], created_by="human"
    )
    candidate = parent.child(guards=[guard])
    decision = Decision(
        parent=parent.version,
        candidate=candidate.version,
        candidate_digest=candidate.digest,
        admitted=True,
        checks=[Check("replay", True, report.to_payload())],
        context={
            "admitted_by": "human",
            "corpus": "examples/claude-code/transcripts, synthetic sessions written for this example",
            "rule": "transcript replay only: the held-out, no-regression and authority gates belong to the "
            "simulated-company loop and were not run for this coding-agent guard",
        },
        decided_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    seed = os.environ.get(SIGNING_KEY_ENV)
    signer = Signer.from_seed_hex(seed) if seed else Signer.generate()
    envelope = receipt(decision, signer)
    (HERE / "harness.json").write_text(json.dumps(asdict(candidate), indent=2) + "\n", encoding="utf-8")
    (HERE / "receipt.json").write_bytes(envelope)
    (HERE / "public_key.txt").write_text(public_key_bytes(signer.public_key).hex() + "\n", encoding="utf-8")

    provenance = export_hook(candidate, GUARD_ID, envelope, {signer.keyid: signer.public_key}, HOOK)
    entry = settings_entry("python3", "${CLAUDE_PROJECT_DIR}/.claude/hooks/" + HOOK.name, "Bash")
    (HERE / "settings.json").write_text(json.dumps(entry, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"replay": report.summary(), "provenance": asdict(provenance)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
