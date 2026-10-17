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
from dataclasses import dataclass
from typing import Any

import pytest
from ratchet_cases.types import Event, Trace
from rich.console import Console

from ratchet.agent.model import ModelTurn, ScriptedModel, scripted_turn
from ratchet.cli import LiveBackend, main
from ratchet.store.atlas import Store

pytestmark = pytest.mark.integration


@dataclass(frozen=True)
class Scenario:
    case: str
    ticket: str
    charge: str
    cents: int
    to: str

    @property
    def refund(self) -> dict[str, Any]:
        return {
            "charge_id": self.charge,
            "amount_cents": self.cents,
            "idempotency_key": "rf-1",
            "reason": "returned",
        }


LOST_REPLY = Scenario("refund_lost_reply.v1", "T-1042", "ch_3MoK8", 3200, "maya.okafor@fastmail.example")
TIMEOUT_FIRST = Scenario("refund_lost_reply.v2", "T-1077", "ch_7LbQ2", 12400, "lars.becker@posteo.example")

CHILD = textwrap.dedent(
    """
    import os, signal, sys
    from ratchet.agent.model import ModelTurn, ScriptedModel, scripted_turn
    from ratchet.agent.seed import seed_harness
    from ratchet.agent.worker import run_case
    from ratchet_cases.cases import get
    from ratchet.store.atlas import Store

    uri, db, run_id, case_id, ticket, charge, cents = sys.argv[1:8]
    store = Store(uri, db)

    class KilledAtThirdRequest(ScriptedModel):
        def complete(self, messages, tools):
            if len(self.requests) == 2:
                os.kill(os.getpid(), signal.SIGKILL)
            return super().complete(messages, tools)

    refund = {"charge_id": charge, "amount_cents": int(cents), "idempotency_key": "rf-1",
              "reason": "returned"}
    model = KilledAtThirdRequest([
        scripted_turn(tool_calls=[("tickets_get", {"ticket_id": ticket})], id_prefix="a"),
        scripted_turn(tool_calls=[("payments_refund", refund)], id_prefix="b"),
    ])
    store.put_harness(seed_harness())
    run_case(seed_harness(), get(case_id), model, store, run_id=run_id)
    """
)


class ResumeBackend(LiveBackend):
    """The real store, worker and case suite; the model is the rest of the script."""

    def __init__(self, env: dict[str, str], turns: list[ModelTurn]) -> None:
        super().__init__(env)
        self.turns = turns

    def model(self) -> object:
        return ScriptedModel(self.turns, name="scripted")


def kill_a_run(uri: str, db: str, run_id: str, sc: Scenario) -> None:
    argv = [sys.executable, "-I", "-c", CHILD, uri, db, run_id, sc.case, sc.ticket, sc.charge, str(sc.cents)]
    done = subprocess.run(argv, capture_output=True, timeout=120)
    assert done.returncode == -signal.SIGKILL, done.stderr.decode()[-500:]


def finish(sc: Scenario, *, retry_refund: bool = False) -> list[ModelTurn]:
    """The rest of the run: check the provider (or retry the refund), tell the customer, close the ticket."""
    first = (
        scripted_turn(tool_calls=[("payments_refund", sc.refund)], id_prefix="c")
        if retry_refund
        else scripted_turn(tool_calls=[("payments_list_refunds", {"charge_id": sc.charge})], id_prefix="c")
    )
    mail = {
        "to": sc.to,
        "template": "refund_issued",
        "reference_id": sc.charge,
        "body": "Your refund was issued.",
        "amount_cents": sc.cents,
    }
    close: list[tuple[str, dict[str, Any]]] = [
        ("mail_send", mail),
        ("tickets_add_note", {"ticket_id": sc.ticket, "body": f"Refund on {sc.charge} confirmed."}),
        ("tickets_set_status", {"ticket_id": sc.ticket, "status": "solved"}),
    ]
    return [first, scripted_turn(tool_calls=close, id_prefix="d"), scripted_turn("Your refund went through.")]


def flat(text: str) -> str:
    """The panel's words on one line: no borders, no wrapping."""
    return " ".join(text.replace("│", " ").split())


def resume(uri: str, db: str, run_id: str, turns: list[ModelTurn]) -> tuple[int, str]:
    buf = io.StringIO()
    env = {"RATCHET_MONGODB_URI": uri, "RATCHET_DB": db}
    code = main(["resume", run_id], backend=ResumeBackend(env, turns), console=Console(file=buf, width=150))
    return code, buf.getvalue()


def refund_calls(trace: Trace) -> list[Event]:
    return [
        e for e in trace.events if e.kind == "tool_call" and e.payload["call"]["name"] == "payments_refund"
    ]


def test_commit_then_timeout_the_audit_log_shows_it_committed_and_the_resume_does_not_repeat_it(
    atlas_db: tuple[str, str, Store],
) -> None:
    uri, db, store = atlas_db
    sc = LOST_REPLY
    kill_a_run(uri, db, "run-killed", sc)
    saved = store.resume("run-killed")
    assert saved is not None
    assert saved["status"] == "running"
    assert [c["name"] for c in saved["executed"]] == ["tickets_get", "payments_refund"]

    code, out = resume(uri, db, "run-killed", finish(sc))
    assert code == 0, out
    assert "RESUMING" in out
    assert "reply lost (timeout)" in out
    assert "NOT REPEATED" in out
    assert "the resume did not send it again" in flat(out)
    assert "NOT COMMITTED" not in out
    assert "RE-ISSUED" not in out
    assert "case graded  PASS" in out

    trace = store.get_trace("run-killed", sc.case)
    assert trace is not None
    assert trace.outcome.passed
    assert len(refund_calls(trace)) == 1  # one refund in the whole run: the one before the kill
    final = store.resume("run-killed")
    assert final is not None
    assert final["status"] == "final"
    code, out = resume(uri, db, "run-killed", finish(sc))
    assert code == 1
    assert "already finished" in out


def test_timeout_before_commit_is_labelled_not_committed_and_a_retry_is_not_a_repeat(
    atlas_db: tuple[str, str, Store],
) -> None:
    uri, db, store = atlas_db
    sc = TIMEOUT_FIRST
    kill_a_run(uri, db, "run-timeout", sc)
    code, out = resume(uri, db, "run-timeout", finish(sc, retry_refund=True))
    assert code == 0, out
    body = flat(out)
    assert "timed out, not committed" in body
    assert "reply lost" not in body
    assert "NOT COMMITTED" in body
    assert "the audit log shows no change, so a retry is not a repeat" in body
    assert "the model retried it: ok" in body
    assert "RE-ISSUED" not in body
    assert "NOT REPEATED" not in body
    assert "case graded PASS" in body

    trace = store.get_trace("run-timeout", sc.case)
    assert trace is not None
    assert trace.outcome.passed
    assert len(refund_calls(trace)) == 2  # the one that timed out, and the retry that landed


def test_a_model_that_repeats_the_committed_refund_after_resume_is_called_out(
    atlas_db: tuple[str, str, Store],
) -> None:
    uri, db, _ = atlas_db
    sc = LOST_REPLY
    kill_a_run(uri, db, "run-killed-2", sc)
    again = [
        scripted_turn(tool_calls=[("payments_refund", sc.refund)], id_prefix="e"),
        scripted_turn("Done."),
    ]
    code, out = resume(uri, db, "run-killed-2", again)
    assert code == 0, out
    body = flat(out)
    assert "RE-ISSUED BY THE MODEL" in body
    assert "the model asked for it again and it ran again" in body
    assert "NOT REPEATED" not in body
