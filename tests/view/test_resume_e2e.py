"""Kill a real run mid-case, then `ratchet resume` it from the Atlas checkpoint.

The case is a refund whose reply is lost after it commits. A child process runs it with a scripted model
that kills the process (SIGKILL) at the third model request, after the refund committed and was
checkpointed. The CLI then resumes the run with the rest of the script and must not repeat the refund.
"""

from __future__ import annotations

import io
import signal
import subprocess
import sys
import textwrap

import pytest
from rich.console import Console

from ratchet.agent.model import ModelTurn, ScriptedModel, scripted_turn
from ratchet.cli import LiveBackend, main
from ratchet.store import Store

pytestmark = pytest.mark.integration
CASE = "refund_lost_reply.v1"
REFUND = {
    "charge_id": "ch_3MoK8",
    "amount_cents": 3200,
    "idempotency_key": "rf-1",
    "reason": "cracked kettle",
}
TICKET = "T-1042"

CHILD = textwrap.dedent(
    """
    import os, signal, sys
    from ratchet.agent.model import ModelTurn, ScriptedModel, scripted_turn
    from ratchet.agent.seed import seed_harness
    from ratchet.agent.worker import run_case
    from ratchet.cases import get
    from ratchet.store import Store

    uri, db, run_id = sys.argv[1:4]
    store = Store(uri, db)

    class KilledAtThirdRequest(ScriptedModel):
        def complete(self, messages, tools):
            if len(self.requests) == 2:
                os.kill(os.getpid(), signal.SIGKILL)
            return super().complete(messages, tools)

    refund = {"charge_id": "ch_3MoK8", "amount_cents": 3200, "idempotency_key": "rf-1", "reason": "cracked"}
    model = KilledAtThirdRequest([
        scripted_turn(tool_calls=[("tickets_get", {"ticket_id": "T-1042"})], id_prefix="a"),
        scripted_turn(tool_calls=[("payments_refund", refund)], id_prefix="b"),
    ])
    store.put_harness(seed_harness())
    run_case(seed_harness(), get("refund_lost_reply.v1"), model, store, run_id=run_id)
    """
)


class ResumeBackend(LiveBackend):
    """The real store, worker and case suite; the model is the rest of the script."""

    def __init__(self, env: dict[str, str], turns: list[ModelTurn]) -> None:
        super().__init__(env)
        self.turns = turns

    def model(self) -> object:
        return ScriptedModel(self.turns, name="scripted")


def kill_a_run(uri: str, db: str, run_id: str) -> None:
    done = subprocess.run(
        [sys.executable, "-I", "-c", CHILD, uri, db, run_id], capture_output=True, timeout=120
    )
    assert done.returncode == -signal.SIGKILL, done.stderr.decode()[-500:]


def finish_script() -> list[ModelTurn]:
    return [
        scripted_turn(tool_calls=[("payments_list_refunds", {"charge_id": "ch_3MoK8"})], id_prefix="c"),
        scripted_turn(
            tool_calls=[
                (
                    "mail_send",
                    {
                        "to": "maya.okafor@fastmail.example",
                        "template": "refund_issued",
                        "reference_id": "ch_3MoK8",
                        "body": "Your 32.00 refund was issued.",
                        "amount_cents": 3200,
                    },
                ),
                ("tickets_add_note", {"ticket_id": TICKET, "body": "Refund of 32.00 on ch_3MoK8 confirmed."}),
                ("tickets_set_status", {"ticket_id": TICKET, "status": "solved"}),
            ],
            id_prefix="d",
        ),
        scripted_turn("Your 32.00 refund went through."),
    ]


def flat(text: str) -> str:
    """The panel's words on one line: no borders, no wrapping."""
    return " ".join(text.replace("│", " ").split())


def resume(uri: str, db: str, run_id: str, turns: list[ModelTurn]) -> tuple[int, str]:
    buf = io.StringIO()
    env = {"RATCHET_MONGODB_URI": uri, "RATCHET_DB": db}
    code = main(["resume", run_id], backend=ResumeBackend(env, turns), console=Console(file=buf, width=150))
    return code, buf.getvalue()


def test_a_killed_run_resumes_and_the_committed_refund_is_not_repeated(
    atlas_db: tuple[str, str, Store],
) -> None:
    uri, db, store = atlas_db
    kill_a_run(uri, db, "run-killed")
    saved = store.resume("run-killed")
    assert saved is not None
    assert saved["status"] == "running"
    assert [c["name"] for c in saved["executed"]] == ["tickets_get", "payments_refund"]

    code, out = resume(uri, db, "run-killed", finish_script())
    assert code == 0, out
    assert "RESUMING" in out
    assert "payments_refund" in out
    assert "reply lost (timeout)" in out
    assert "NOT REPEATED" in out
    assert "the resume did not send it again" in flat(out)
    assert "RE-ISSUED" not in out
    assert "case graded  PASS" in out

    trace = store.get_trace("run-killed", CASE)
    assert trace is not None
    assert trace.outcome.passed
    refunds = [
        e for e in trace.events if e.kind == "tool_call" and e.payload["call"]["name"] == "payments_refund"
    ]
    assert len(refunds) == 1  # one refund in the whole run: the one before the kill
    final = store.resume("run-killed")
    assert final is not None
    assert final["status"] == "final"
    code, out = resume(uri, db, "run-killed", finish_script())
    assert code == 1
    assert "already finished" in out


def test_a_model_that_repeats_the_committed_refund_after_resume_is_called_out(
    atlas_db: tuple[str, str, Store],
) -> None:
    uri, db, _ = atlas_db
    kill_a_run(uri, db, "run-killed-2")
    again = [
        scripted_turn(tool_calls=[("payments_refund", {**REFUND, "reason": "cracked"})], id_prefix="e"),
        scripted_turn("Done."),
    ]
    code, out = resume(uri, db, "run-killed-2", again)
    assert code == 0, out
    body = flat(out)
    assert "RE-ISSUED BY THE MODEL" in body
    assert "the model asked for it again and it ran again" in body
    assert "NOT REPEATED" not in body
