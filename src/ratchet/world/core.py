"""The simulated company: every system behind one tool surface, with deterministic faults."""

from __future__ import annotations

import copy
from datetime import UTC, datetime, timedelta
from typing import Protocol

from ratchet.types import ToolCall, ToolResult, canonical_json, sha256_hex
from ratchet.world.approvals import Approvals
from ratchet.world.billing import Billing
from ratchet.world.catalog import Catalog
from ratchet.world.customers import Customers
from ratchet.world.faults import Fault
from ratchet.world.mail import Mail
from ratchet.world.orders import Orders
from ratchet.world.payments import Payments
from ratchet.world.policy import Policy
from ratchet.world.subscriptions import Subscriptions
from ratchet.world.system import JSON, System, ToolDef, ToolError, ToolSpec, check_args
from ratchet.world.tickets import Tickets
from ratchet.world.view import StateView

AGENT = "agent"
SYSTEMS = (
    "customers",
    "payments",
    "subscriptions",
    "catalog",
    "orders",
    "tickets",
    "approvals",
    "policy",
    "mail",
    "billing",
)
_ISO = "%Y-%m-%dT%H:%M:%SZ"


def parse_time(text: str) -> datetime:
    return datetime.strptime(text, _ISO).replace(tzinfo=UTC)


def format_time(moment: datetime) -> str:
    return moment.strftime(_ISO)


class CaseLike(Protocol):
    @property
    def initial_state(self) -> JSON: ...

    @property
    def faults(self) -> list[Fault]: ...


class World:
    """A deterministic simulated company.

    The clock starts at `initial_state["clock"]["start"]` and advances one minute per agent call. Every
    change is written to the owning system's audit log. Faults fire at declared points of the agent's call
    sequence. Two worlds built from the same case and fed the same calls have the same digest.
    """

    def __init__(self, initial_state: JSON, faults: list[Fault] | None = None) -> None:
        state = copy.deepcopy(initial_state)
        self._start = parse_time(state.pop("clock")["start"])
        unknown = sorted(set(state) - set(SYSTEMS))
        if unknown:
            raise ValueError(f"unknown systems in initial state: {unknown}")
        self._step = 0
        self._seq = 0
        self._ids: dict[str, int] = {}
        self._tool_counts: dict[str, int] = {}
        self._faults = list(faults or [])
        self._fired: list[str] = []
        self._scheduled: list[tuple[int, Fault]] = []
        self._audit: dict[str, list[JSON]] = {name: [] for name in (*SYSTEMS, "world")}
        self._state: dict[str, JSON] = {}
        catalog = Catalog(self._section(state, Catalog), self)
        self._tickets = Tickets(self._section(state, Tickets), self)
        self._approvals = Approvals(self._section(state, Approvals), self)
        self._policy = Policy(self._section(state, Policy), self)
        built: list[System] = [
            Customers(self._section(state, Customers), self),
            Payments(self._section(state, Payments), self),
            Subscriptions(self._section(state, Subscriptions), self),
            catalog,
            Orders(self._section(state, Orders), self, catalog),
            self._tickets,
            self._approvals,
            self._policy,
            Mail(self._section(state, Mail), self),
            Billing(self._section(state, Billing), self),
        ]
        self._tools: dict[str, ToolDef] = {t.name: t for s in built for t in s.tools()}

    def _section(self, state: JSON, cls: type[System]) -> JSON:
        section = cls.default_state()
        section.update(state.get(cls.name, {}))
        self._state[cls.name] = section
        return section

    @classmethod
    def from_case(cls, case: CaseLike) -> World:
        return cls(case.initial_state, list(case.faults))

    # Context for systems -------------------------------------------------------------------------

    def now(self) -> str:
        return format_time(self._start + timedelta(minutes=self._step))

    def next_id(self, prefix: str) -> str:
        self._ids[prefix] = self._ids.get(prefix, 0) + 1
        return f"{prefix}_sim{self._ids[prefix]:04d}"

    def record(self, system: str, actor: str, op: str, args: JSON, result: JSON) -> None:
        self._seq += 1
        self._audit[system].append(
            {
                "seq": self._seq,
                "step": self._step,
                "at": self.now(),
                "actor": actor,
                "op": op,
                "args": copy.deepcopy(args),
                "result": copy.deepcopy(result),
            }
        )

    # Public surface ------------------------------------------------------------------------------

    @property
    def step(self) -> int:
        return self._step

    def tools(self) -> list[ToolSpec]:
        """OpenAI function-tool schemas for every tool, sorted by name."""
        return [self._tools[name].spec() for name in sorted(self._tools)]

    def tool_names(self) -> list[str]:
        return sorted(self._tools)

    def call(self, call: ToolCall, actor: str = AGENT) -> ToolResult:
        """Execute one tool call. Only agent calls advance the clock and trigger faults."""
        tool = self._tools.get(call.name)
        if tool is None:
            result = ToolResult(call.call_id, False, None, f"unknown tool: {call.name}")
        else:
            result = self._execute(tool, call, actor)
        if actor == AGENT:
            self._advance(call.name)
        return result

    def _execute(self, tool: ToolDef, call: ToolCall, actor: str) -> ToolResult:
        problem = check_args(tool.parameters, call.args)
        if problem is not None:
            return ToolResult(call.call_id, False, None, f"invalid_arguments: {problem}")
        timeout = self._timeout_for(tool.name) if actor == AGENT else None
        if timeout is not None and timeout.kind == "timeout_before_commit":
            self._fault_record(timeout, {"tool": tool.name, "committed": False})
            return ToolResult(call.call_id, False, None, "timeout")
        try:
            data = tool.handler(copy.deepcopy(call.args), actor)
        except ToolError as err:
            return ToolResult(call.call_id, False, None, str(err))
        if timeout is not None:
            self._fault_record(timeout, {"tool": tool.name, "committed": True})
            return ToolResult(call.call_id, False, None, "timeout")
        return ToolResult(call.call_id, True, data, None)

    def _timeout_for(self, tool_name: str) -> Fault | None:
        count = self._tool_counts.get(tool_name, 0) + 1
        for fault in self._faults:
            if fault.kind != "mutate" and fault.tool == tool_name and fault.nth == count:
                return fault
        return None

    def _advance(self, tool_name: str) -> None:
        self._tool_counts[tool_name] = self._tool_counts.get(tool_name, 0) + 1
        self._step += 1
        for fault in self._faults:
            if fault.kind != "mutate" or fault.id in self._fired:
                continue
            if fault.after_step == self._step:
                self._fire(fault)
            elif fault.tool == tool_name and fault.nth == self._tool_counts[tool_name]:
                self._scheduled.append((self._step + fault.delay, fault))
        due = [f for at, f in self._scheduled if at == self._step and f.id not in self._fired]
        for fault in due:
            self._fire(fault)

    def _fire(self, fault: Fault) -> None:
        self._fired.append(fault.id)
        self._fault_record(fault, {"ops": len(fault.ops)})
        for op in fault.ops:
            self._apply(op)

    def _fault_record(self, fault: Fault, detail: JSON) -> None:
        self.record("world", "fault", fault.kind, {"id": fault.id}, detail)

    def _apply(self, op: JSON) -> None:
        kind = op["op"]
        if kind == "call":
            self.call(
                ToolCall(op["tool"], copy.deepcopy(op["args"]), f"{op['actor']}-{self._seq}"), op["actor"]
            )
        elif kind == "approval_revoke":
            self._approvals.revoke(op["approval_id"], op["by"], op["reason"])
        elif kind == "policy_publish":
            self._policy.publish(op["doc_id"], op["version"])
        elif kind == "customer_message":
            self._tickets.customer_message(op["ticket_id"], op["body"], op.get("requests", []))
        else:
            raise ValueError(f"unknown fault op: {kind}")

    def view(self) -> StateView:
        """A read-only snapshot of every system, the clock and the audit logs."""
        return StateView(self._snapshot())

    def _snapshot(self) -> JSON:
        data: JSON = {"step": self._step, "now": self.now()}
        data.update(self._state)
        data["audit"] = self._audit
        data["faults_fired"] = list(self._fired)
        return data

    def digest(self) -> str:
        """SHA-256 of the canonical JSON of all state, the audit logs and the clock."""
        return sha256_hex(canonical_json(self._snapshot()))
