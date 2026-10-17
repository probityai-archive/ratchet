# Built today and reused

The repository's first commit is from 2026-10-11, and everything under `src/ratchet`, `probity-ratchet-cases`, `tests`, `examples` and `docs` was written at the event. The rest is a pinned dependency or a service.

| Reused | Pin |
|---|---|
| Runtime | `jcs-admit==0.1.0`, `httpx>=0.27`, `cryptography>=44`, `rich>=15`; extras: `pymongo>=4.10`, `raindrop-ai==0.0.75` |
| Earlier work of the same authors | `probity-verify==0.1.1`, `agent-evidence-vectors==0.19.0` |
| Benchmark extra | `agentdojo==0.1.35` (tool outputs carry hidden prompt-injection attacks), `litellm==1.102.1`, `terminal-bench==0.2.18`, `tau-bench` at `59a200c` (simulated customer-service tasks graded on the final database state), `tau2` at `4ce7c03`, `tool-sandbox` at `c8571d7` |
| Instruction search extra | `gepa==0.1.4` |
| Services | MongoDB Atlas (`mongodb-atlas-local:8.3.9-20260928T091948Z` for development), Voyage embeddings, Raindrop |

The integration that writes and checks Ratchet's own admission records is new. The verifier it calls is not.

License: Apache-2.0.
