# Running Ratchet

## 0. The demo, with nothing to build

One image holds the tool, a local Atlas store and the scripted double-refund scenario. The agent is a scripted
worker with fixed replies, so the demo needs no model, no key, no GPU and no network once the image exists.
Build it from a checkout, then run the scenario once:

```sh
docker build -f docker/demo.Dockerfile -t ratchet-demo:dev .
docker run --rm ratchet-demo:dev ratchet demo double-refund
```

To read what the demo stored, keep one container running and run each command in it:

```sh
docker run -d --rm --name ratchet-demo ratchet-demo:dev        # starts the store (about 20 s)
docker exec ratchet-demo ratchet demo double-refund                     # one admission end to end; prints the ids below
docker exec ratchet-demo ratchet show decision 13881cee67fe     # the bad fix (block refunds of 50.00 or more): REJECTED
docker exec ratchet-demo ratchet show decision ea3bd812059a     # the good fix (check for an existing refund): ADMITTED
docker exec ratchet-demo ratchet replay 13881cee67fe ea3bd812059a
docker exec ratchet-demo ratchet verify ea3bd812059a --pubkey-file /data/ratchet-demo/ratchet_demo-decision-key.pub
docker rm -f ratchet-demo
```

The candidate ids are content addresses, so they are the same on every run. The demo signs with an Ed25519
key it generates for the run and writes only the public half, to the path it prints; every decision it signs
records that a demo key signed it. `ratchet verify` checks a demo decision against that file. Without
`--pubkey-file` it checks against the pinned decision key, `keys/ratchet-decisions.pub`, and reports INVALID,
because a demo key is never the pinned key. `docker/demo/check.sh` runs this sequence with `--network none` and
fails unless every step gives the answer above; CI runs it on every push.

Inside the image, guard children apply rlimits, seccomp and Landlock but not a network namespace: Docker's
default seccomp profile refuses the user namespace that layer needs. Seccomp still refuses every socket.

From a checkout instead, with the store of section 1 running, `uv run ratchet demo double-refund` uses that store when
`RATCHET_MONGODB_URI` is unset and prints the follow-up commands with the variables they need.

## 1. A store

Ratchet keeps everything in MongoDB Atlas. For development, the local Atlas image runs mongod and the search
service on one node, so Vector Search and change streams work offline:

```sh
docker compose -f docker/compose.yaml up -d --wait
export RATCHET_MONGODB_URI='mongodb://ratchet:ratchet-local-dev@127.0.0.1:27018/?directConnection=true&authSource=admin'
export RATCHET_DB=ratchet
uv run python -m ratchet.store --uri "$RATCHET_MONGODB_URI" --db "$RATCHET_DB"
```

The last command creates the collections, indexes and search indexes. It is safe to run again. The local
credentials above belong to the throwaway container in `docker/compose.yaml`. On an Atlas cluster, put that
cluster's connection string in `RATCHET_MONGODB_URI`.

## 2. Environment

| Variable | Used for |
|---|---|
| `RATCHET_MONGODB_URI`, `RATCHET_DB` | the store (database defaults to `ratchet`) |
| `RATCHET_BASE_URL`, `RATCHET_API_KEY`, `RATCHET_MODEL` | the agent model, through any OpenAI-compatible endpoint |
| `RATCHET_IMPROVER_MODEL` | the model that proposes guards (with `RATCHET_IMPROVER_BASE_URL` and `RATCHET_IMPROVER_API_KEY` for a different endpoint) |
| `RATCHET_SIGNING_KEY` | 32-byte Ed25519 seed as hex; signs decisions. Never stored in the repository |
| `VOYAGE_API_KEY` | embeddings for failure retrieval |
| `RAINDROP_WRITE_KEY` | with `--raindrop`, sends runs and decisions to Raindrop |
| `RATCHET_IMPROVER_URI` | the improver's connection string, a database user with no access to held-out cases |
| `RATCHET_GUARD_REQUIRE` | isolation layers a guard child must apply (default `rlimits,netns,seccomp,landlock`; must keep `netns` or `seccomp`) |
| `RATCHET_DEMO_DIR` | where `ratchet demo` writes its public key (default `.ratchet-demo`) |

Nothing is hard-coded and nothing secret is written to a trace, a log or the store.

## 3. Commands

```sh
uv run ratchet run --split heldout --workers 8      # one harness over a case split (default harness: seed)
uv run ratchet evolve -n 3                          # the loop: diagnose, propose, replay, evaluate, admit
uv run ratchet replay                               # replay the latest generation's guards, no model call
uv run ratchet show lineage                         # every version, admitted and rejected
uv run ratchet show trace RUN_ID/CASE_ID            # one trace as a timeline
uv run ratchet show decision DECISION_ID            # one decision, gate by gate
uv run ratchet show decision DECISION_ID --envelope-out receipt.json
uv run ratchet verify DECISION_ID                   # check the signed receipt
uv run ratchet live                                 # follow the change streams
uv run ratchet report -o ratchet-report.html        # one static page: lineage, replay matrix, results
```

`ratchet replay` takes candidate versions or ids, reveals its matrix row by row on a terminal, and prints
`N traces, X s, 0 model calls` under it. `ratchet run --split all` runs train and held-out; the untouched audit
split runs only when named.

## 4. Verify without Ratchet's code path

`--envelope-out` writes the signed receipt. `python -m ratchet.verify` checks it with the pinned
`probity-verify` package and the pinned key in `keys/ratchet-decisions.pub`. See [VERIFY.md](VERIFY.md).

## 5. The Claude Code hook

An admitted guard exports to a single-file Claude Code `PreToolUse` hook. See
[examples/claude-code/README.md](../examples/claude-code/README.md).

## 6. Tests and checks

```sh
uv sync --all-groups --all-extras
uv run ruff check && uv run ruff format --check && uv run mypy src
uv run pytest
python3 scripts/readme-lint.py README.md
python3 scripts/check-docs.py
```

The store's integration tests need the local Atlas container from step 1.

## 7. Benchmarks

Each benchmark adapter has a module entry point that writes its raw records to a run directory and a `report`
step that renders `docs/results/`:

```sh
uv run python -m ratchet.bench.agentdojo --help
uv run python -m ratchet.bench.tau --help
uv run python -m ratchet.bench.terminal --help
uv run python -m ratchet.bench.transfer --help
```

The results pages say which subset ran, with task counts.
