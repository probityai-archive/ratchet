# Atlas feature probes on the hackathon sandbox

Every feature below was run against the real sandbox before the product claims it. Project
`6acbd00b1b25c55279b6511b`, `Cluster0` (M10, AWS). Dates are 2026-10-11. The cluster was MongoDB 8.0.34 when the
probes started and was upgraded in place to 9.0.4 during the session; both results are recorded. The probe
database is `ratchet_featureprobe`. Raw results (JSON, one file per probe and server version), the captured
documentation pages and their `SHA256SUMS` are in `research/_raw/atlas-probes/` (not committed). Code:
`src/ratchet/atlas_features/`; tests: `tests/atlas_features/` (`-m sandbox` runs the live ones).

## Summary

| # | Feature | 8.0.34 | 9.0.4 | Verdict |
|---|---------|--------|-------|---------|
| 1 | `$rankFusion` | works | works | ship (scoreDetails available) |
| 1 | `$scoreFusion` | code 40324, stage unknown | works | ship on 9.0 |
| 2 | `autoEmbed` vector index (Atlas-run Voyage embeddings) | works | works | ship |
| 3 | Database trigger writing sanitized failure memory | works | works | ship |
| 4 | Agent Engine Policy Engine | NOT RUN | NOT RUN | needs an operator-issued credential |
| 4 | Agent Engine standalone memory | NOT RUN | NOT RUN | same blocker |
| 5 | MongoDB MCP server, restricted identity, one training read | works | works | ship |

Recommendation for retrieval (item 1, numbers below): use vector search alone, with server-side `autoEmbed`,
as the default `similar_failures`; use `$scoreFusion` with `minMaxScaler` normalization and weights 2:1 when the
query may be a bare identifier or an exact phrase. On this set fusion did not beat vector search alone.

## 1. `$rankFusion` and `$scoreFusion`

Set: 32 synthetic failure summaries in 8 families (including 4 ordinary-work distractors) and 30 queries with
one gold document each: 7 lexical, 7 paraphrase, 7 mixed, 4 bare-identifier, 5 ambiguous. Vectors are
`voyage-4-large`, 1024 dimensions, cosine; text is an Atlas Search index on `summary`. Each leg keeps 10
candidates. The metric is recall@3 (gold in the top 3); recall@1 and MRR@3 are alongside. Latency is the median
over the 30 queries from the probe host, so it includes the network round trip.

Exact pipelines (the two legs are `$vectorSearch` with `queryVector`, `numCandidates: 50, limit: 10`, and
`$search` with `text` on `summary` then `$limit: 10`):

```
{$rankFusion: {input: {pipelines: {vec: <vector leg>, txt: <text leg>}},
               combination: {weights: {vec: 1, txt: 1}}, scoreDetails: false}}, {$limit: 3}
{$scoreFusion: {input: {pipelines: {vec: <vector leg>, txt: <text leg>}, normalization: "sigmoid"},
                combination: {weights: {vec: 1, txt: 1}}, scoreDetails: false}}, {$limit: 3}
```

Server response for `$scoreFusion` on 8.0.34, verbatim:
`code 40324 (Location40324): Unrecognized pipeline stage name: '$scoreFusion'`. `$rankFusion` ran on 8.0.34
with identical results to 9.0.4, so it does not need 8.1 or later. After the upgrade `$scoreFusion` returned
`['rl1', 'rl2', 'od4']` for the first query. `scoreDetails: true` on `$rankFusion` returns per-leg ranks and the
formula the server applied: `sum of weight * (1 / (60 + rank))` over the legs that returned the document.

Results on 9.0.4 (identical recall on 8.0.34 where the stage exists):

| Mode | recall@3 | recall@1 | MRR@3 | median ms | misses |
|------|---------:|---------:|------:|----------:|--------|
| vector only | **1.000** | 0.867 | 0.922 | 69 | none |
| text only | 0.900 | 0.667 | 0.772 | 65 | q11, q14, q30 |
| `$rankFusion` 1:1 | 0.900 | 0.767 | 0.828 | 88 | q11, q14, q30 |
| `$rankFusion` vec 2 : txt 1 | 0.933 | 0.800 | 0.861 | 98 | q14, q30 |
| `$rankFusion` vec 1 : txt 2 | 0.900 | 0.733 | 0.811 | 89 | q11, q14, q30 |
| `$scoreFusion` sigmoid 1:1 | 0.900 | 0.767 | 0.828 | 95 | q11, q14, q30 |
| `$scoreFusion` sigmoid 2:1 | 0.900 | 0.767 | 0.828 | 105 | q11, q14, q30 |
| `$scoreFusion` minMaxScaler 2:1 | 0.967 | 0.733 | 0.839 | 137 | q28 |
| rank fusion by hand (`$unionWith`, same formula) | 0.900 | 0.767 | 0.828 | 101 | q11, q14, q30 |

What the numbers say, and their limits. The differences between the fusion modes are one or two queries out of
30, so none of the orderings among them is a finding; they are measurements on a small set. Vector search alone
reached 30/30 at rank 3 on this set because every paraphrase query is semantically close to its gold; the text
leg adds nothing for paraphrases and, in fusion, pulls the two paraphrase misses (q11, q14) and one ambiguous
query (q30) down. The text leg was needed for none of the four bare-identifier queries, because the vector leg
also found them. Both fusion operators therefore cost one extra index and 20 to 60 ms and bought no recall
here. Weighting the vector leg 2:1 recovers one query; `minMaxScaler` normalization recovered the most (0.967)
but lost rank-1 precision (0.733). The hand-built version of the rank fusion matches the operator exactly,
so the operator is a convenience, not a capability gap, on 8.0.

Reproduce: `uv run python -m ratchet.atlas_features.fusion` (writes `fusion.json` and `fusion-<version>.json`).

## 2. Atlas Vector Search automated embedding

Works on this M10 on both server versions. Index definition, stored documents hold text only:

```
{"fields": [{"type": "autoEmbed", "modality": "text", "path": "summary", "model": "voyage-4"}]}
```

Query with text, no vector computed by the client:
`{$vectorSearch: {index: "auto_voyage_4", path: "summary", query: "<text>", numCandidates: 50, limit: 10}}`.
`query` may also be `{text: "<text>"}`; both forms returned the same document and score.

- Index creation to `READY`: 80 s on the first build (8.0.34, three replica members becoming ready one by one),
  under 1 s when the index already existed on the 9.0.4 run.
- Recall@3 on the same 30 queries: `voyage-4` 1.000 and `voyage-4-large` 1.000 (recall@1 0.867 for both),
  matching the client-embedded `voyage-4-large` vector leg. Median query latency 195 ms and 193 ms
  (the query text is embedded on the server at query time, against 69 ms when the client sends a vector).
- Writes: a new document became searchable by meaning 2.4 s to 16.8 s after insert and an update 1.3 s to
  9.9 s after the write (three runs, 9.0.4 the slower), with no client code.
- `$rankFusion` accepts an `autoEmbed` leg: recall@3 0.933 at weights 2:1, median 162 ms.

Trade-off: server-side embedding removes the embedding key and the write-side pipeline from Ratchet and keeps
vectors in sync on update, at about 2.5 times the query latency of a client-supplied vector. For the failure
memory (low query volume, text-only writers) that trade favors `autoEmbed`.

Reproduce: `uv run python -m ratchet.atlas_features.autoembed`.

## 3. Atlas Trigger writing sanitized failure memory

Created entirely through the App Services Admin API with the project API key (login at
`POST /api/admin/v3.0/auth/providers/mongodb-cloud/login`; no operator step was needed). App
`ratchet-featureprobe` is linked to `Cluster0`; function `failureMemory` (source:
`src/ratchet/atlas_features/failure_memory.js`); trigger `failure_to_memory`:

```
{"type": "DATABASE", "config": {"database": "ratchet_featureprobe", "collection": "failures_probe",
  "operation_types": ["INSERT"], "full_document": true, "unordered": false, "skip_catchup_events": false}}
```

The function upserts `_id = "fm:" + failure_id` with `$setOnInsert`, so writing the same failure twice, or
delivering the same change event twice, creates one document. It removes email addresses, provider object ids
(`ch_...`, `re_...`), `order`/`ticket`/`account` references, `#123` ids and money amounts before storing. The
memory collection has an `autoEmbed` index, so the trigger holds no embedding key.

Measured on 9.0.4:

- Insert into `failures_probe` to the memory document existing: 0.56 s to 0.81 s over three runs.
- Stored text for `Charge ch_204 for order 7781 refunded $1,250.00 twice; customer bob@example.com was paid twice.`
  is `Charge <id> for order <id> refunded <amount> twice; customer <email> was paid twice.`
- A second insert of the same `failure_id`: one memory document. Two forced re-deliveries of the same change
  event (`/debug/execute_function`, run as system): both returned `inserted: false`; count stayed 1.
- The sanitized memory was found by meaning ("customer received the same refund two times") by the
  `autoEmbed` index 5.0 s after the insert.

Not covered: the redelivery was forced through the function-execution endpoint, not by making the service
redeliver. The trigger's delivery guarantee is that of the platform; the function's idempotence is what was
proved. The sanitizer is a deterministic pattern filter; it removes the ids, emails and amounts listed above
and nothing else, so free-text names would pass through.

Reproduce: `uv run python -m ratchet.atlas_features.triggers` (idempotent; reuses the app).

## 4. Atlas Agent Engine (public preview): NOT RUN

Documentation captured with `curl` on 2026-10-11 and used for this section: the Policy Engine page, the memory
page, the API-keys page, the install page, the limitations page and the gateway's OpenAPI document
(`agentengine-openapi.json`, 133 paths, security scheme `BearerAuth`), listed in `SHA256SUMS`.

What the documentation states (the design this probe implements):

- `AUTHORIZED_TOOLS` is an allowlist policy. An organization policy applies to every project; a project policy
  "must be at least as restrictive" as the organization's of the same kind, the effective value is the
  intersection, and a project policy that is less restrictive is rejected on save. A tool outside the effective
  list halts the execution with `error_code: policy_denied`, message `Policy denied: <reason>`.
- Routes: `POST /api/v1/projects/{id}/policies` (body `{type, string_list, enabled}`), `PUT .../{policy_id}`,
  `GET .../policies/effective`, and the same under `/api/v1/organizations/{id}/policies` plus `.../preview`.
- Enforcement applies to executions on the platform's Orchestration Engine. Ratchet's worker runs outside it, so
  this is a policy integration demonstration: it shows the platform refusing a widening and mirrors an admitted
  allowlist; it does not enforce Ratchet's own worker.

Blocker, from commands that ran. The gateway answers `GET /health` with `200 {"status":"healthy"}`; every API
route answers without credentials with `401 {"success":false,"error":"Missing Authorization header","code":"MISSING_AUTH_HEADER"}`;
`POST /api/v1/oauth/token` without client credentials answers `401 {"error":"invalid_client","error_description":"client credentials required"}`;
the Atlas Admin API key sent with digest auth gets the same `MISSING_AUTH_HEADER`, and with HTTP Basic auth
`{"code":"INVALID_AUTH_FORMAT"}`. The gateway accepts only a bearer token: a service-account access token
(`ae_sa_id_...` / `ae_sa_sk_...`, created by a signed-in user with `PROJECT_OWNER`) or a user session from the
browser login (`agentengine auth login`, an OIDC flow). The keyring holds no `agentengine` entry and the CLI
binary is downloaded from the signed-in UI. Operator action, one time: sign in at
https://agentengine.mongodb.com with the Atlas account that owns this project, create a project service account
with role `PROJECT_OWNER`, and store its id and secret in the keyring as `service=agentengine
account=sa-client-id` and `account=sa-client-secret`. Setting the organization policy additionally needs an
`ORG_OWNER` session (a service account holds only `ORG_GROUP_CREATOR` or `ORG_READ_ONLY` at organization scope).

Shipped and tested against a stand-in gateway that implements the documented merge rule
(`src/ratchet/atlas_features/agent_engine.py`): the allowlist-to-policy mapping, the create-or-update install,
the widening attempt with the server's refusal captured verbatim, a revision test (whether a project revision
that is wider than its predecessor but still inside the organization list is accepted, which separates
"stricter than the organization" from "stricter than the previous revision"), and a procedural-memory
write-then-search round trip through the project's memory tool server. With a credential,
`uv run python -m ratchet.atlas_features.agent_engine` with `AGENTENGINE_PROJECT_ID` and
`AGENTENGINE_ORG_TOOLS` set runs it live. Until it does, the revision rule is untested and Ratchet's own
parent-to-candidate subset check is the only temporal guard.

Supply-chain check for the memory SDK (nothing was installed or imported). The documentation tells users to
install the memory SDK from a MongoDB-hosted package index with an extra index URL; that route needs the
credential above. Public PyPI also serves `agent-engine-sdk-memory` 0.11.8. It was downloaded with
`pip download --no-deps --require-hashes` from the default index only, against the wheel hash
`aad5d2d288486f7dd53b7f5348a618c1e96b88083ea049156c04282f72f1e745` (matched), and every file was read without
importing: pure Python, depends on `httpx` and `pydantic`, no subprocess, eval, dynamic import, `.pth` or entry
points, network calls only to the gateway URL or a caller-supplied base URL, license headers name MongoDB,
Inc. The PyPI project page lists no author, maintainer, project URLs or provenance attestation, so the publisher
is not verified from PyPI; the rule that the publisher must be verifiably MongoDB is therefore not met by the
public index, and the REST client in `agent_engine.py` calls the documented routes directly instead. The import
name the documentation uses for the older package, `agentic_platform_memory`, resolves on PyPI to a package that
describes itself as a benign security-research placeholder and does nothing; it must not be installed. The
PyPI package named `agentengine` is an unrelated framework.

## 5. MongoDB MCP server at runtime

Server: `mongodb-mcp-server` 3.0.5, installed with `npm install --prefix` into
`research/_raw/atlas-probes/mcp/` (maintainers on the registry include mongodb.com addresses; repository
`mongodb-js/mongodb-mcp-server`) and run as `node .../dist/esm/index.js` over stdio with
`MDB_MCP_READ_ONLY=true` and `MDB_MCP_TELEMETRY=disabled`; the connection string reaches the process only
through its environment. Identity: database user `ratchet-improver` (role `ratchetImprover`, inheriting
`readWrite` on database `ratchet` only).

- `tools/list` returned 20 tools. None of `insert-many`, `update-many`, `delete-many`, `drop-collection` or
  `drop-database` is present in read-only mode.
- One training read: `find` on `ratchet.traces` with `connectionId: "preconfigured"` returned the one sentinel
  document the probe wrote for the purpose and removed afterwards (`Query on collection "traces" resulted in 1
  documents`). The collection was empty before.
- One held-out read: `find` on `ratchet_heldout.cases` returned `isError: true`:
  `Error running find: not authorized on ratchet_heldout to execute command { find: "cases", ... }`.
- Wall time for initialize, listing and both calls: 11 s to 24 s, dominated by process start.

`connectionId` is required by this version's tools; `"preconfigured"` selects the connection given in the
environment. Reproduce: `uv run python -m ratchet.atlas_features.mcp_probe` (needs the private install).

## Operational notes

- The first connection to the sandbox measured 1.6 s, so the fail-fast limit is 5 s, not 2 s. A 2 s limit failed
  once with `ServerSelectionTimeoutError` while the cluster was changing state; the same connection succeeded
  with a longer limit seconds later.
- The sandbox access list holds two egress addresses and the workstation alternated between them within one
  session; both are on the list.
- Another lane's cleanup emptied the original probe database once; it was renamed to `ratchet_featureprobe`
  and every probe re-seeds its own data idempotently.
