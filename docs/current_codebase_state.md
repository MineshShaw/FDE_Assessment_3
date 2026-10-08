# Current Codebase State

This document describes the supported implementation as of the current
repository state. The supported launcher is `./run.sh`; it starts the mock
vendor-risk API and `ui/app.py`.

## Runtime paths

- `src/data_loader.py` loads the synthetic CSV/JSON/Markdown snapshot into
  cached Pandas DataFrames.
- `src/tools.py` provides read-only budget, catalog, vendor-risk, and policy
  tools. Vendor results include strict-JSON normalization, expiry, conflict,
  source, and outage fields.
- `src/solution.py` contains `evaluate_request`, the deterministic policy
  floor. `handle_request(request_id, architecture)` uses that floor directly
  without a configured provider, or invokes the selected agent when a provider
  client is configured. Agent evidence is marked with source `agent`; floor
  evidence is preserved.
- `src/agents/single_agent.py` and `src/agents/two_agent.py` implement the
  Single and Staged paths. Each records LLM calls and tool names in
  `last_telemetry`, uses bounded provider retries, returns flagged escalation
  on failures, and treats request/tool content as untrusted data.
- `src/guardrails.py` reconciles model output with deterministic approvals,
  missing information, risk flags, and blocking recommendations.
- `ui/app.py` is the active Streamlit dashboard. It supports REQ-ID-backed
  structured facts, custom requests, evidence display, latency/tool telemetry,
  and explicit session-only human decisions.
- `evaluation/` contains the real-run evaluator and policy-derived cases.
  `evals/` contains the smaller public contract checks.

## Invariants

- Policy reference date: `2026-09-30`.
- Agent loops are capped at 10 iterations.
- No provider key means deterministic behavior; provider failures fall back to
  the deterministic decision and add `llm_unavailable`.
- No file under `data/` is modified by the application.
- `.env` and runtime caches are ignored and must not be committed.

## Legacy files

The root `app.py` and `run_local.py` are retained only as starter-pack
compatibility artifacts. They are not used by `run.sh` or current evaluations.
`templates/` and `STUDENT_CHECKLIST.md` are submission-reference artifacts,
not runtime inputs. The active UI and launcher are documented in `README.md`.
