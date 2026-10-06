# AI Procurement Request Copilot

This repository implements an evidence-grounded procurement copilot over synthetic business data. It checks budget, software overlap, vendor security, and deterministic policy requirements while keeping final approval with a human.

## Quickstart

Requirements: Python 3.11+ and network access for dependency installation and model calls.

```bash
git clone <repository-url>
cd FDE_Assessment_3_Starter_Pack
cp .env.example .env
# Set OPENAI_API_KEY in .env.
./run.sh
```

`run.sh` installs `requirements.txt` and starts Streamlit headlessly with browser auto-opening and usage-stat prompts disabled. Open the URL printed by Streamlit. The dashboard supports preloaded evaluation cases or a custom request, Single Agent or Staged Two-Agent execution, evidence inspection, and session-only Approve/Reject/Override review actions.

For validation without the UI:

```bash
python verify_setup.py
pytest
python evaluation/evaluator.py
```

## Configuration

The standard OpenAI Python SDK is configured for Gemini's OpenAI-compatible endpoint:

```dotenv
OPENAI_API_KEY=your-key
MODEL_NAME="gemini-2.0-flash"
OPENAI_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/"
```

`.env` is ignored by Git and loaded without overwriting environment variables already set by the operating system. Never commit credentials.

## Data and tools

`src/data_loader.py` loads the complete synthetic snapshot into memory as Pandas DataFrames: employees, budgets, software catalog, vendors, purchase history, requests, vendor risk, and policy text. No database is required. The deterministic tools in `src/tools.py` query those frames for budget availability, catalog overlap, vendor security status, and approval/risk rules.

Both agent architectures use explicit pure-Python `while` loops. The loop sends a request through the OpenAI SDK, executes returned tool calls locally, appends tool results to the message history, and stops when validated Pydantic JSON is returned. Each loop is capped at five iterations. The staged design separates an Analyst evidence pack from a Reviewer policy decision.

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

Run `python evaluation/evaluator.py` to execute both architectures, compare each recommendation with its golden label, count LLM/tool calls, measure latency, print a Markdown summary, and write `evaluation/benchmark_results.json`. The committed initial offline benchmark passed 10/10 cases for each architecture. See `docs/architecture_decision_memo.md` for the architecture choice and tradeoffs.

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

All data is synthetic. The copilot recommends actions only; it does not purchase software, approve spend, alter budgets, or accept legal terms.
