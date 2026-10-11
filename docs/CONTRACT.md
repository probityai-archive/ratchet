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
  `digest` is the SHA-256 of the canonical JSON of every field EXCEPT `digest` and `version` (exactly
  `Harness.content_of`); `version` is the first 12 hex chars of `digest`. Every read and every verification
  recomputes both and rejects a mismatch.

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

## Admission rule (the ratchet), ruling 2

Admission and activation are separate operations. Search scores, Pareto
membership, and descendant productivity never authorize activation.

### Frozen comparison

Before proposals are evaluated, freeze and digest the admission policy,
case manifests, graders, fault schedules, model configuration, runtime,
view projection, replay manifest, and proposal/retry budgets.

Freeze each candidate before protected evaluation. Any candidate change
creates a new digest and requires evaluation of that exact bundle.

Changes to the policy or evaluator create a new version. Previously
observed results remain historical evidence and are not relabeled as
prospective validation under the new version.

### Structural and authority gates

Automatic candidates may append guards, procedures, and prompt sections,
and may narrow the tool allowlist.

Existing guards and procedures remain unchanged and in order. Existing
prompt bytes remain a prefix. IDs are unique. Tool permissions are a
subset of the comparison parent's permissions.

The runtime, evaluator, credentials, guard projection, and enforcement
mechanism are not candidate-editable.

Human-authorized exceptions are separately signed, name the exact
candidate and policy exception, and are not ordinary automatic admissions.

### Replay gate

The replay manifest fixes all eligible recorded inputs and explicitly
excludes the untouched final audit. Protected admission-history traces
may be included only under evaluator access.

Replay uses the exact recorded runtime guard inputs: call, permitted
pre-call view, and history prefix. The projection schema and runtime
versions are evidence artifacts.

Each added guard has nonempty declared target evidence. It must fire on
every declared target failure at or before its identified causing step,
and on no passing or benign-twin trace in the eligible snapshot.

Evaluate added guards individually and evaluate the composed guard
runtime to detect interactions. Complete the declared replay even after
a counterexample. Record target hits, all false fires, coverage, and
runtime errors. Errors or incomplete coverage cannot produce clearance.

Procedure-trigger replay applies only to deterministic triggers actually
used by the runtime. Required and forbidden trigger contexts are frozen
before evaluation. A procedure firing on a passing trace is not inherently
a false fire. Natural-language triggers have replay marked NOT APPLICABLE.

Replay establishes predicate behavior on recorded inputs, not successful
counterfactual continuation. All candidate types require fresh evaluation.

### Fresh worker-evaluation gate

Run the comparison parent twice freshly, P1 and P2, and the immutable
candidate twice freshly, C1 and C2, on the same declared case manifest.

Parent reports may be shared within one frozen comparison cohort with
the same parent digest and evaluator configuration.

For each case, outcomes are pass or fail.

Stable recoveries are cases failing in both P1 and P2 and passing in both
C1 and C2. Require at least two stable recoveries.

Protected cases are cases passing in either P1 or P2. Every protected case
must pass in both C1 and C2. Thus every parent-discordant case must pass in
both candidate runs.

Require zero must-succeed regressions and zero forbidden benign-twin
interventions in either candidate run.

Record all four per-case outcome vectors, stable-recovery and regression
sets, parent discordance D, and costs. D is reported, not used as a
net-count threshold.

The confirmation uses the same candidate digest. Additional retries may
not replace unfavorable completed runs. Infrastructure retry rules are
predeclared; missing or invalid evidence cannot satisfy admission.

### Overfitting screen

Reject fixture-specific case/customer/charge/order identifiers and copied
fixture-specific names or amounts in executable or instructional changes.
Provenance metadata is not executable logic.

Shared policy constants require a documented source independent of the
training fixture. Record the screen version and result. Passing this
screen does not prove absence of semantic memorization.

### Evidence access

The evaluator owns protected cases, traces, grades, caches, and detailed
reports. Neither the improver nor candidate code receives evaluator or
admin credentials.

The proposer may receive complete training replay results. Protected
feedback is limited to the frozen sanitized report schema: no protected
case IDs, raw inputs, per-case outcomes, guard-returned strings, or
exceptions containing protected data.

Retrieval uses training-visible material only. The final audit is unused
until the selected harness is frozen and is never fed back into that
experiment's selection loop.

### Level 2

One admission entry point supports separate worker and improver reports.

Freeze i0 and i1 before evaluation on proposal tasks excluded from their
instruction tuning. Both receive identical task inputs, starting worker
harnesses, downstream evaluation rules, and fixed proposal budgets.

Count invalid, duplicate, rejected, and empty slots. Count distinct
independently admitted worker candidates and report their yield, cluster
coverage, stable recoveries, and costs.

For automatic i1 promotion, require more distinct admitted candidates
than i0, no loss of covered evaluation clusters, and no reduction in
aggregate stable recoveries under the same declared budget. Every counted
candidate must independently satisfy the worker admission gates.

Descendant productivity is an additional search/reporting metric. Its
formula, horizon, attribution, and denominator are versioned. It does
not replace this comparison or a worker admission gate.

### Receipts and activation

Persist immutable evidence and a signed DSSE admission receipt before
activation. The receipt binds the candidate, comparison parent, policy
and evaluator versions, manifests, reports, and any exceptions.

The independently runnable verification bundle pins the DSSE verifier,
Ratchet predicate adapter, schemas, and trusted key configuration. It
reports signature checks, digest checks, recomputed predicates/replay,
and authenticated-only evaluator assertions separately.

Activation uses an idempotent MongoDB transaction that conditionally
updates the expected active version and revision, validates the evidence
and receipt binding, and records activation. Missing receipts or stale
active references prevent activation.

An exploratory child of an older ancestor cannot directly replace the
active harness. Rebase its delta onto the active harness and evaluate
the resulting immutable candidate against that active parent.

Telemetry and external exports are outside the activation transaction.

FRESH executions count as evaluation evidence. RECORDED displays add no
new evidence. CACHED response playback never counts as a fresh run.

Costs are recorded and reported, not used as a worker admission gate.

### Fresh worker-evaluation gate: stochastic policy v3

This policy is a tolerance-based empirical evaluation policy for
nondeterministic executions. It does not establish zero regression,
statistical significance, or universal reliability.

Before evaluation, freeze the policy, evaluator configuration, case
manifest, runtime, endpoint configuration, execution schedule, proposal
budget, and infrastructure retry rules.

Every arm contains 40 fresh executions of every declared case, divided
prospectively into two panels of 20. Panel membership is fixed before
outcomes are observed.

Interleave arms using a predeclared randomized schedule. Each execution
uses isolated world state. Model sessions and other mutable state must
not carry information between executions.

When supported, use a predeclared seed schedule shared across arms.
A shared seed does not establish deterministic or independent outcomes.
Record the actual model, endpoint, batching, and concurrency configuration.

For case i and arm A, let s(A, i) be its number of passes out of 40.
Let s(A, i, j) be its number of passes out of 20 in panel j.

A case is high-pass when s(A, i) is at least 30.
A case is low-pass when s(A, i) is at most 10.
All other cases are intermediate. Intermediate cases remain in every
denominator and in the per-case degradation checks.

A stable recovery is a case for which:
- the parent is low-pass;
- the candidate is high-pass; and
- in each panel, the candidate has at least eight more passes than
  the parent.

Require at least two stable recoveries.

A stable regression is a case that is high-pass under the parent and
low-pass under the candidate. Require zero stable regressions.

Additionally, for every case, candidate pass count must be no more than
eight below the parent pass count out of 40.

The candidate's unweighted mean case pass rate must be no more than
five percentage points below the parent's, both across all 40 executions
and separately in each panel.

Must-succeed cases receive all of these checks and are reported as a
separate subset. This policy does not require every must-succeed execution
to pass. Any separately declared hard safety invariant remains absolute
and cannot be relaxed by these tolerances.

Require zero forbidden benign-twin interventions in every candidate
execution.

Before evaluating candidates in a comparison cohort, qualify the cohort
with a fresh parent reference arm and two fresh identity-control arms.
Each identity-control arm executes exactly the parent's effective harness,
without instructional or executable changes, under the same evaluator
configuration and schedule rules.

Each identity-control arm must:
- pass every non-regression check against the parent reference; and
- have fewer than two stable recoveries against that reference.

An identity-control failure makes the cohort UNQUALIFIED. No candidate
may be admitted using that cohort. Retain and report the failed controls.
Do not replace unfavorable completed control runs or adjust tolerances
within the cohort.

A candidate with no change to the effective system prompt, procedures,
guards, or tool permissions is ineligible for improvement admission.
Changes only to identifiers, provenance, parent references, or timestamps
do not constitute an effective harness change.

The identity-control empirical improvement check is required even though
identity candidates are structurally ineligible for admission.

Parent references may be shared only within the qualified frozen cohort.
Both identity controls are calibration evidence, not additional parent
runs from which a favorable reference may be selected.

Every candidate is immutable before its protected evaluation begins.
The two panels evaluate the same candidate digest. Neither panel may
be replaced by an additional favorable completed run.

Missing or invalid executions cannot satisfy the gate. Infrastructure
retries follow only the predeclared rules and retain attempt records.
Candidate-caused runtime failures are not infrastructure retry events.

Reports retain all execution outcomes, panel assignments, pass counts,
classification changes, stable recoveries, stable regressions, per-case
declines, aggregate differences, control results, costs, and retry records.

Protected detailed reports remain evaluator-only. Proposer feedback
continues to use the frozen sanitized report schema.

Passing this policy authorizes admission only when every other applicable
gate passes. Receipt-first conditional activation remains separate.

Receipts name stochastic policy v3 and its exact thresholds. Existing
ruling-2 decisions remain ruling-2 decisions.

Operator acceptance, 2026-10-11: stochastic policy v3 is adopted for nondeterministic evaluators, including its tolerances (a per-case observed decline of up to 8 of 40 passes and an aggregate decline of up to 5 percentage points). A decision made under v3 never claims zero regressions.

## Public text (operator ruling, 2026-10-11)

No public artifact (README, docs/, code comments, docstrings, commit messages, demo, video, submission form, posts,
pull requests) cites, credits or names any external paper, method, project or person as prior or related work.
Research lineage stays in `private/` only. Two things are not citations and stay: the dependency manifest (pinned
packages and their versions, required by the hackathon's new-work rule) and the name of a public benchmark next to a
result measured on it.

## Safety

- Guard source is untrusted model output. It runs in a separate process with no network, a CPU and memory
  limit, a time limit per call, and only the `ToolCall`, `StateView` and `history` passed in as JSON.
- No real payment, mail or ticket system is touched. The world is simulated.
- No secret is written to the repository, a trace, a log or Atlas. Keys come from the environment.

## Commit discipline

Each lane owns its directories and their tests. Commit each unit the moment its tests pass (conventional
commits, plain ASCII, imperative, no AI attribution, no internal IDs). Never edit another lane's files; ask the
integrator to change a shared type. Zero warnings from `ruff`, `mypy --strict`, `pytest -W error`.
