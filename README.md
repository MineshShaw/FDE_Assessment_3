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

Before starting, confirm that `.env` contains the provider key, `MODEL_NAME`, and `OPENAI_BASE_URL`; `run.sh` installs `requirements.txt` and is the supported one-command startup path. It starts Streamlit with browser auto-opening and usage-stat prompts disabled. Open the URL printed by Streamlit. The dashboard supports preloaded evaluation cases or a custom request, Single Agent or Staged Two-Agent execution, evidence inspection, and session-only Approve/Reject/Override review actions.

Product workflow: **request intake → Pandas/API evidence gathering → deterministic budget and policy checks → structured recommendation → human review**. See [`docs/architecture_workflow.md`](docs/architecture_workflow.md) for the architecture diagram and assumptions.

For validation without the UI:

```bash
python verify_setup.py
python -m pytest
python evaluation/evaluator.py
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

`src/data_loader.py` loads the complete synthetic snapshot into memory as Pandas DataFrames: employees, budgets, software catalog, vendors, purchase history, requests, vendor risk, and policy text. No database is required. The deterministic tools in `src/tools.py` query those frames for budget availability, catalog overlap, vendor security status, and approval/risk rules.

Both agent architectures use explicit pure-Python `while` loops. The loop sends a request through the OpenAI SDK, executes returned tool calls locally, appends tool results to the message history, and stops when validated Pydantic JSON is returned. Each loop is capped at 10 iterations. The staged design separates an Analyst evidence pack from a Reviewer policy decision.

## Evaluation

`evaluation/test_cases.json` contains 10 cases covering:

- normal and low-value requests;
- missing user counts;
- catalog duplication/overlap;
- expired or unavailable vendor compliance;
- extreme budget overruns;
- PII and other sensitive data;
- prompt injection;
- new-vendor legal review.

Run `PYTHONPATH=. .venv/bin/python evaluation/evaluator.py` to execute both architectures against the exact same cases, compare each recommendation with its golden label, validate tool-grounded evidence, policy compliance, human-review correctness, and next-action presence, count LLM/tool calls, measure latency, print a Markdown summary, and write `evaluation/benchmark_results.json`. The committed artifact passed 10/10 cases for each architecture across all checks. Its rows are marked `offline_stub`; they are orchestration metrics, not live provider latency. With a configured key, the evaluator can run through the OpenAI-compatible provider. See [`docs/architecture_decision_memo.md`](docs/architecture_decision_memo.md) for the evidence-based ship decision and tradeoffs.

Initial offline comparison:

| Metric | Single Agent | Staged Two-Agent |
|---|---:|---:|
| Cases passing | 10/10 | 10/10 |
| Average LLM calls | 2.0 | 3.0 |
| Average tool calls | 1.0 | 1.0 |
| Average latency in committed artifact | 1.47 ms | 0.76 ms |
| Primary tradeoff | Lower cost and simpler flow | Stronger evidence/reviewer separation |

The original public contract adapter remains available through:

```bash
python evals/run_public_evals.py --architecture single
python evals/run_public_evals.py --architecture staged
```

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

## Assumptions and known limitations

All data is synthetic and small enough to load into memory. The committed benchmark uses an offline SDK-shaped stub (`execution_mode: offline_stub`), so its latency is orchestration overhead rather than production model latency. A live provider run requires a configured key and may require the local vendor-risk service for API-backed evidence. Real deployment should add provider retries, token/cost telemetry, broader hidden-case coverage, and operational authentication. Vendor-risk service outages remain explicit uncertainty and require human review.

## Final ship decision

Ship the **Staged Two-Agent** architecture: the Analyst separates evidence collection from the Reviewer’s deterministic policy decision, improving auditability for sensitive procurement cases. Keep Single Agent as the lower-cost fallback for simpler requests. Both passed 10/10 benchmark cases; staged used one additional LLM call per case and measured lower local orchestration latency in this offline run.
