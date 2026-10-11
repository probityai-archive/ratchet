# Ratchet: build contract

Ratchet turns observed agent failures into executable harness changes. Before it adopts a change, it checks the
proposed guard against recorded actions, runs fresh evaluations, prevents any growth of authority, and publishes a
signed, independently checkable admission record. Every version, trace, evaluation and decision lives in MongoDB
Atlas. Admitted changes satisfy the recorded-history and evaluation gates; every decision states exactly what
evidence was checked, and every rejection stays in the record.

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

## Admission rule (the ratchet), adjudicated 2026-10-11

Freeze before proposing: cases, graders, fault schedules, model configuration, evaluator version (digests recorded).
Changing any of them after held-out outcomes were seen starts a new evaluator version.

A candidate is admitted only when ALL hold, and the decision records each check with counts and ids:
1. Replay (guards only): over the declared, digest-frozen trace snapshot, using each call's recorded PRE-CALL
   view, the guard fires on every target failure at or before the causing step, and on zero passing traces and
   zero benign-twin traces. Replay may run similarity-ordered and stop at the first counterexample (REJECTED,
   with "k/N traces checked"); clearance requires an exhaustive pass ("N/N checked"). Replay proves the
   predicate's behaviour on recorded inputs only, not that the counterfactual continuation succeeds.
2. Fresh held-out evaluation: run the parent twice fresh; D = cases whose outcome differs between the two runs.
   The candidate's held-out pass count must exceed the better parent count by at least max(2, D+1). Then one
   more fresh candidate run must confirm it. Procedures and prompt edits are judged by this gate only.
3. No regression: zero failures on must_succeed cases that passed in either parent run; zero unwarranted
   interventions on benign twins.
4. Overfitting screen: no literal case, customer, charge or order ids, names or amounts from training traces.
5. Authority does not grow: tool_allowlist is a subset of the parent's and no guard is removed, unless a
   human-signed change says so. This is an enforced policy, not a safety proof.
6. Cost is recorded and reported. It is not a gate.

Evaluation modes are distinct and labelled everywhere: FRESH (counts for admission), RECORDED (a stored fresh
result shown again, with its timestamp), CACHED (byte-identical request replay; debugging only, never admission).

Held-out secrecy: held-out cases and their traces live in collections the improver's database identity cannot
read; the improver process never inherits evaluator or admin credentials. A final audit cohort is never used for
any decision until the harness is frozen.

Level 2 (the improver's own harness): versions i0 and i1 are frozen, get the same training-visible failure
material, and are scored on separately held-out proposal tasks with fixed slots and budget, counting invalid,
duplicate, rejected and empty proposals. The claim is exactly what that comparison shows.

The receipt (DSSE) names: corpus snapshot digest, parent, candidate, evaluator version, every report digest,
exceptions. It is verified by the pinned public verifier (`probity-verify`) and an independently pinned key.

## Safety

- Guard source is untrusted model output. It runs in a separate process with no network, a CPU and memory
  limit, a time limit per call, and only the `ToolCall`, `StateView` and `history` passed in as JSON.
- No real payment, mail or ticket system is touched. The world is simulated.
- No secret is written to the repository, a trace, a log or Atlas. Keys come from the environment.

## Commit discipline

Each lane owns its directories and their tests. Commit each unit the moment its tests pass (conventional
commits, plain ASCII, imperative, no AI attribution, no internal IDs). Never edit another lane's files; ask the
integrator to change a shared type. Zero warnings from `ruff`, `mypy --strict`, `pytest -W error`.
