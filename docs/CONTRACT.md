# Ratchet: build contract

Ratchet is an agent harness that turns each failure into an executable guard or procedure, proves the change
against the recorded history before it adopts it, and keeps every version, trace and proof in MongoDB Atlas.
A ratchet moves one way: a change is admitted only when it fixes failures and breaks nothing that worked.

This file is the interface every lane builds against. Change it only through the integrator (main session).

## Package layout (Python 3.12+, `uv`, package `ratchet`, src layout)

```
src/ratchet/
  world/      lane WORLD   simulated company: billing provider, tickets, approvals, policy, mail
  cases/      lane WORLD   case families, variants, train/held-out split, deterministic graders
  agent/      lane AGENT   harness bundle, worker loop, guard runtime, model client
  store/      lane STORE   Atlas persistence, vector search, checkpoints, change streams
  improve/    lane IMPROVE diagnosis, candidate synthesis, replay validation, admission, receipts
  cli.py      integrator   `ratchet run|evolve|replay|show|verify`
tests/        one test module per source module, plus tests/integration/
```

## Shared types (`src/ratchet/types.py`, owned by the integrator, frozen dataclasses, JSON round-trip)

- `ToolCall(name: str, args: dict, call_id: str)`
- `ToolResult(call_id: str, ok: bool, data: dict | None, error: str | None)`; an error that hides a
  committed side effect (timeout after commit) is `ok=False, error="timeout"` and the world still changed.
- `Event(seq: int, kind: Literal["model","tool_call","tool_result","guard_block","procedure","checkpoint","final"], payload: dict, ts: float)`
- `Trace(run_id, case_id, harness_version, model, events: list[Event], outcome: Outcome)`
- `Outcome(passed: bool, reasons: list[str], final_state_digest: str, cost: Cost)`
- `Cost(model_calls: int, input_tokens: int, output_tokens: int, tool_calls: int, wall_s: float)`
- `Guard(id, description, source: str, origin_failure_ids: list[str])`. `source` defines
  `def check(call: ToolCall, view: StateView, history: list[Event]) -> str | None`; a string is the block
  reason shown to the model. Guards run in the guard runtime, never with `exec` in the worker process
  (see Safety).
- `Procedure(id, name, when: str, steps: list[str], origin_failure_ids: list[str])`
- `Harness(version: str, parent: str | None, system_prompt: str, procedures: list[Procedure], guards: list[Guard], tool_allowlist: list[str], created_by: Literal["seed","improver","human"], digest: str)`.
  `digest` is the SHA-256 of the canonical JSON of every other field; the version id is its first 12 hex chars.

## Lane interfaces

- WORLD: `World.from_case(case) -> World`; `World.tools() -> list[ToolSpec]` (OpenAI tool-schema JSON);
  `World.call(ToolCall) -> ToolResult`; `World.view() -> StateView` (read-only); `World.digest() -> str`.
  `Case(id, family, variant, split: Literal["train","heldout"], must_succeed: bool, customer_message, initial_state, faults, grader)`;
  `grade(case, world, trace) -> Outcome`. Graders check the FINAL STATE of every system (provider ledger,
  ticket, mail outbox, approvals), never the agent's words alone.
- AGENT: `run_case(harness, case, model: ModelClient, store: Store | None) -> Trace`. Every guard decision is
  an event. `ModelClient` is OpenAI-compatible; base URL, key and model come from env
  (`RATCHET_BASE_URL`, `RATCHET_API_KEY`, `RATCHET_MODEL`), nothing hard-coded.
- STORE: `Store(uri, db)`; `put_harness`, `get_harness`, `lineage()`, `put_trace`, `traces(filter)`,
  `put_eval`, `similar_failures(text, k)` (Atlas Vector Search, Voyage embeddings), `checkpoint(run_id, state)`,
  `resume(run_id)`, `watch(collection)` (change stream). Local dev runs against the `mongodb/mongodb-atlas-local`
  image, so vector search works offline; the demo runs against the hackathon Atlas sandbox.
- IMPROVE: `diagnose(traces) -> list[FailureCluster]`; `propose(cluster, harness) -> list[Candidate]`;
  `replay(guard, traces) -> ReplayReport` (fires on each target failure at the step that caused it, and on no
  passing trace); `evaluate(candidate, cases) -> EvalReport`; `admit(parent, candidate, reports) -> Decision`;
  `receipt(decision) -> bytes` (signed DSSE envelope over the canonical JSON of the decision and its evidence).

## Admission rule (the ratchet)

A candidate harness is admitted only when all hold, and the decision records each check:
1. Replay: every new guard fires on its target failures and on zero previously passing traces.
2. Held-out: pass count on held-out cases strictly rises, measured on cases the improver never saw.
3. No regression: no `must_succeed` case that passed before fails now (refusing everything cannot win).
4. Authority does not grow: `tool_allowlist` is a subset of the parent's, and no guard is removed, unless a
   human-signed change says so. The improver can never grant itself access.
5. Cost is recorded and reported. It is not a gate.

## Safety

- Guard source is untrusted model output. It runs in a separate process with no network, a CPU and memory
  limit, a time limit per call, and only the `ToolCall`, `StateView` and `history` passed in as JSON.
- No real payment, mail or ticket system is touched. The world is simulated.
- No secret is written to the repository, a trace, a log or Atlas. Keys come from the environment.

## Commit discipline

Each lane owns its directories and their tests. Commit each unit the moment its tests pass (conventional
commits, plain ASCII, imperative, no AI attribution, no internal IDs). Never edit another lane's files; ask the
integrator to change a shared type. Zero warnings from `ruff`, `mypy --strict`, `pytest -W error`.
