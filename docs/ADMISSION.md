# The admission rule, in plain words

Ratchet changes an agent's harness only when the change has passed every check below. This page is the
short version; the exact rule is in [CONTRACT.md](CONTRACT.md).

A change is a **guard** (code that blocks a tool call), a **procedure** (written steps), or a **prompt
section**. The agent being improved is the **parent**. The changed version is the **candidate**.

## Before anything runs

The cases, graders, fault schedules, model configuration, evaluator version, replay snapshot and proposal
budgets are frozen and digested. A candidate is frozen before it is evaluated. If any of these changes, that is a new
version, and old results are history, not evidence for the new one.

## The checks

| # | Check | In plain words |
|---|---|---|
| 1 | **Structure and authority** | A change may add guards, procedures and prompt sections, and may shrink the tool list. Existing guards and procedures stay, in order. Existing prompt bytes stay a prefix. Tools can only be removed. A human-signed exception is a separate record, not an ordinary admission. A new tool or an edited procedure is admitted only on an admitted k-session bundle ([SESSIONS.md](SESSIONS.md)) that compares exactly this parent and candidate, and only when the candidate's fresh runs called every added tool and used every edited procedure |
| 2 | **Replay** | Each new guard has declared target failures. Using each call's recorded inputs (the call, the state before it, the history before it), the guard must fire on every target at or before the step that caused it, and on no passing trace and no benign twin. Each guard is replayed alone and all guards composed. The whole snapshot is replayed even after a first counterexample; an error or incomplete coverage cannot clear a guard |
| 3 | **Fresh evaluation** | The parent runs twice and the candidate runs twice on the same held-out cases. A *stable recovery* fails both parent runs and passes both candidate runs: at least two are needed. Every case the parent passed in either run must pass in both candidate runs. No must-succeed regression, and no intervention on a benign twin. All four outcome vectors, the parent's run-to-run disagreement and costs are recorded |
| 4 | **Overfitting screen** | No case, customer, charge or order identifier, and no copied name or amount, from the fixtures may appear in the change |
| 5 | **A signed receipt** | The decision binds the candidate, the parent, the policy and evaluator versions, the manifests and every report, and is signed before the change is activated |

Cost is recorded in every decision. It is not a gate.

## What the evidence does and does not show

- **Replay** shows how a guard behaves on recorded inputs. It does not show that the run would have
  succeeded after the block, so every kind of change also needs fresh evaluation.
- **Fresh evaluation** shows what the evaluated cases did. It is finite, and neither check proves a change safe in general.
- **Tool-list monotonicity** is an enforced policy over named capabilities, not a proof of safety.
- **A signature** shows who signed which decision with which evidence digests. It does not show the evaluator was honest.
- Every decision lists exactly what was checked, with counts and case ids, and rejected decisions are kept.

## Labels you will see

| Label | Meaning |
|---|---|
| FRESH | every model call was answered fresh; the only mode that counts as evidence |
| RECORDED | a stored fresh result shown again, with its timestamp |
| CACHED | a byte-identical response replay, for debugging only; never evidence |

## Who can see what

Held-out cases and their traces live in a database the improver's identity cannot read. The improver receives
a sanitized report schema, never protected case ids, raw inputs or per-case outcomes. A final audit cohort is
unused until the harness is frozen.

## Activation

Admission and activation are separate. Activation is one idempotent database transaction that checks the
receipt and the expected active version, so a stale or missing receipt cannot activate a change.

## The improver's own instructions

The improver's instructions are versioned too. Two frozen versions are scored on separately held-out
proposal tasks with the same budget, and counted for invalid, duplicate, rejected and empty proposals. Each
counted candidate must independently pass the checks above.
