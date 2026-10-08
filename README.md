# AI Procurement Request Copilot

This repository implements an evidence-grounded procurement copilot over synthetic business data. It checks budget, software overlap, vendor security, and deterministic policy requirements while keeping final approval with a human.

## Quickstart

Requirements: Python 3.11+ and network access for dependency installation and model calls.

```bash
git clone <repository-url>
cd FDE_Assessment_3_Starter_Pack
cp .env.example .env
# Set GROQ_API_KEY (or OPENAI_API_KEY) in .env and verify OPENAI_BASE_URL.
./run.sh
```

Before starting, confirm that `.env` contains the provider key, `MODEL_NAME`, and `OPENAI_BASE_URL`; `run.sh` installs `requirements.txt`, starts the local vendor-risk mock API on port 8001, and is the supported one-command startup path. It then starts Streamlit with browser auto-opening and usage-stat prompts disabled. Open the URL printed by Streamlit. The dashboard supports preloaded evaluation cases or a custom request, Single Agent or Staged Two-Agent execution, evidence inspection, and session-only Approve/Reject/Override review actions.

Product workflow: **request intake → Pandas/API evidence gathering → deterministic budget and policy checks → structured recommendation → human review**. See [`docs/architecture_workflow.md`](docs/architecture_workflow.md) for the architecture diagram and assumptions.

For validation without the UI:

```bash
python verify_setup.py
python -m pytest
PYTHONPATH=. .venv/bin/python evaluation/evaluator.py --offline-smoke
```

## Configuration

The standard OpenAI Python SDK is configured for Groq's OpenAI-compatible endpoint:

```dotenv
OPENAI_API_KEY=your-key
GROQ_API_KEY=your-groq-key
MODEL_NAME="llama-3.3-70b-versatile"
OPENAI_BASE_URL="https://api.groq.com/openai/v1"
```

`.env` is ignored by Git and loaded without overwriting environment variables already set by the operating system. Never commit credentials.

## Data and tools

`src/data_loader.py` loads the complete synthetic snapshot into memory as Pandas DataFrames: employees, budgets, software catalog, vendors, purchase history, requests, vendor risk, and policy text. No database is required. The deterministic tools in `src/tools.py` query those frames for budget availability, ranked catalog overlap, vendor security status, computed expiry/conflict, and approval/risk rules.

Both agent architectures use explicit pure-Python `while` loops. The loop sends a request through the OpenAI SDK, executes returned tool calls locally, appends tool results to the message history, and stops when validated Pydantic JSON is returned. Each loop is capped at 10 iterations. The staged design separates an Analyst evidence pack from a Reviewer policy decision.

## Evaluation

`evaluation/gold_cases.json` contains policy-derived cases from the request records plus threshold, unknown-requester, and prompt-injection variants. Gold expectations cite the policy and are scored independently from `evaluate_request`; disagreements should be reviewed rather than silently rewritten.

- normal and low-value requests;
- missing user counts;
- catalog duplication/overlap;
- expired or unavailable vendor compliance;
- extreme budget overruns;
- PII and other sensitive data;
- prompt injection;
- new-vendor legal review.

Run `PYTHONPATH=. .venv/bin/python evaluation/evaluator.py --runs 3 --sleep 2` with `GROQ_API_KEY` or `OPENAI_API_KEY` configured to execute both architectures against identical request data, include the policy-engine baseline, validate exact approvals/risk flags/evidence grounding, count LLM/tool calls, measure latency, print a Markdown summary, and write timestamped results under `evaluation/results/`. Normal mode fails without credentials; `--offline-smoke` only checks plumbing and writes no benchmark results. Live results should be regenerated before submission and are never substituted with fixture outputs.

There is intentionally no shipping comparison table here until a repeated live-provider run completes. Offline smoke is not a benchmark and must not be used as latency or reliability evidence. The decision memo contains the pending comparison schema.

The original public contract adapter remains available through:

```bash
python evals/run_public_evals.py --architecture single
python evals/run_public_evals.py --architecture staged
```

With no provider key, `handle_request` uses the deterministic policy engine.
When a key is configured, it invokes the selected agent, combines its evidence
with the deterministic policy floor, and records LLM/tool telemetry; failures
fall back to the floor with `llm_unavailable`.

## Repository layout

```text
data/                 Synthetic CSV, JSON, and policy snapshot
src/data_loader.py    In-memory Pandas loading
src/tools.py          Deterministic integration tools
src/agents/           Single and staged orchestration
src/schemas.py        Unified ProcurementOutput contract
evaluation/           Benchmark cases, runner, and results
ui/app.py             Streamlit dashboard
run.sh                One-command headless launcher
tests/                Data, tool, and orchestration tests
```

`app.py` and `run_local.py` are legacy starter-pack entry points and are not
used by `run.sh`, the active dashboard, or the current evaluation commands.
`templates/` contains legacy submission templates retained for reference.

## Assumptions and known limitations

All data is synthetic and small enough to load into memory. A real benchmark requires a configured provider and records per-run metadata and traces; smoke mode is not evidence. Both agent paths use deterministic policy floors, bounded provider retries, explicit tool errors, and human review. Vendor-risk service outages remain explicit uncertainty and require human review.

## Final ship decision

Use **Single Agent as the default**: it uses one fewer model call and has the smaller failure surface. Keep Staged Two-Agent as an experimental/audit-oriented option until a repeated live-provider benchmark demonstrates equal or better correctness and latency. No completed live comparison is currently available.
