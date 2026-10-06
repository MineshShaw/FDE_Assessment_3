# Current Codebase State

This document describes the repository as it exists on disk at the time of the
audit. It is an inventory and behavioral description only; it does not describe
an intended future architecture.

## Repository structure and runtime entry points

The repository contains:

- `data/`: synthetic CSV, JSON, and Markdown procurement data.
- `src/`: data loading, deterministic tools, Pydantic schemas, LLM client
  helpers, agent orchestration, and the legacy public-contract implementation.
- `evaluation/`: the newer ten-case benchmark runner, cases, and generated
  results.
- `evals/`: the original public evaluation harness and generated CSV results.
- `ui/app.py`: the active Streamlit dashboard used by `run.sh`.
- `app.py`: a separate legacy starter UI that calls `src.solution.handle_request`.
- `run.sh`: installs requirements and launches `ui/app.py`.
- `verify_dashboard.sh`: starts `run.sh` in the background and checks the
  Streamlit health endpoint.
- `docs/`: architecture documentation and this physical-state audit.
- `tests/`: loader, tools, mock API, data-integrity, and agent tests.

The root also contains `.env.example`, `.gitignore`, `requirements.txt`, the
legacy `run_local.py`, and `verify_setup.py`. `.env` is ignored by Git. The
current requirements list is `openai`, `pydantic>=2.0`, `streamlit`, `pandas`,
`pytest`, and `python-dotenv`.

`run.sh` prefers `.venv/bin/python` and `.venv/bin/streamlit`, installs the
requirements on each launch, sets `PYTHONPATH` to include the repository root,
and runs Streamlit with browser auto-opening enabled and usage-stat collection
disabled. `.streamlit/credentials.toml` sets the Streamlit email to an empty
string. The legacy `run_local.py` instead starts the FastAPI mock vendor service
on port 8001 and the legacy root `app.py` on port 8501.

## Data management

The physical data snapshot is under `data/`:

- `employees.csv`
- `department_budgets.csv`
- `software_catalog.csv`
- `vendors.csv`
- `purchase_history.csv`
- `requests.json`
- `vendor_risk.json`
- `procurement_policy.md`

`src/data_loader.py` is the newer all-in-memory loader. CSV files are loaded
with `pandas.read_csv`. `requests.json` is normalized with
`pandas.json_normalize` when it contains a list. Dictionary-shaped JSON, as
used by `vendor_risk.json`, is converted with `DataFrame.from_dict(...,
orient="index")` and receives a `vendor_name` column. The policy Markdown is
represented as a one-row DataFrame with `policy_name` and `content`.

`load_all_data()` returns a dictionary of DataFrames with these keys:
`employees`, `budgets`, `software_catalog`, `vendors`, `purchase_history`,
`requests`, `vendor_risk`, and `policies`. The loader reads files each time it
is called; it does not maintain a process-level cache or database.

The older `src/data_access.py` remains in place and exposes separate loaders.
Its CSV functions return DataFrames, but `load_requests()` returns raw
dictionaries and `load_policy_text()` returns a raw string. `src/solution.py`
uses this older access layer, not `load_all_data()`.

## Deterministic tools in `src/tools.py`

The module calls `load_all_data()` through `_data()` and defines four public
tools. Each tool catches exceptions and returns a compact dictionary error
envelope rather than propagating tool failures.

### `check_budget(department_id: str, amount: float) -> dict`

The amount must be a non-negative finite real number and the department must
be a non-empty string. The tool filters the `budgets` DataFrame
case-insensitively. A match returns:

```json
{
  "department": "...",
  "available_funds": 0.0,
  "remaining_funds": 0.0,
  "within_budget": true
}
```

An unknown department returns `{"status": "not found", "department": ...}`.
Invalid inputs or loader failures return `{"error": "budget check failed: ..."}`.

### `search_software_catalog(need_description: str, category: str) -> list[dict] | dict`

Both inputs must be strings. The tool filters by exact case-insensitive
category and by terms longer than two characters found across selected catalog
text fields. It returns at most two compact records:

```json
[
  {"name": "Product name", "desc": "Catalog notes"}
]
```

An empty match is an empty list. Invalid inputs or failures return an error
dictionary.

### `get_vendor_security_status(vendor_name: str) -> dict`

The tool looks up the vendor in the local `vendors` and `vendor_risk` DataFrames,
then attempts the mock vendor-risk HTTP service through
`src.vendor_client.get_vendor_risk()` with a 0.5-second timeout. API data takes
precedence when available; local risk data is the fallback. The compact result
contains the vendor name and whichever registry, risk, review-date, and service
availability fields are available. An unknown vendor returns
`{"status": "not found", "vendor_name": ...}`. Request and other failures are
converted to an error dictionary.

### `evaluate_policy_rules(amount, vendor_risk, data_classification) -> dict`

This is deterministic. It validates the amount and string inputs, assigns
Manager, Department Head, Finance, CFO, and Procurement approvals based on
amount thresholds, and adds security/privacy risk flags for high or unknown
vendor risk and sensitive classifications such as PII, confidential data,
source code, or production. It returns the normalized amount, unique
`approvals_required`, unique `risk_flags`, and Boolean CFO/human-review
indicators. Failures return a policy error dictionary.

## Schemas and shared agent helpers

`src/schemas.py` defines:

- `ProcurementOutput`, whose recommendation is constrained to `APPROVE`,
  `REJECT`, `ESCALATE_TO_HUMAN`, or `REQUEST_INFO`. Evidence, approval,
  missing-information, and risk lists default to empty lists. `next_step`
  defaults to `Manual review required.`.
- `StructuredEvidencePack`, containing `budget_status`, `tool_overlap`, and
  `vendor_risk`. A dictionary supplied for `tool_overlap` is coerced into a
  one-item list.

`src/agents/_common.py` defines the four OpenAI-style tool schemas, dispatches
tool calls through `TOOL_FUNCTIONS`, extracts the first response choice, and
serializes tool results as JSON. `append_assistant_message()` appends either a
Pydantic message dump, a dictionary copy, or a manually normalized test-double
message containing assistant tool calls.

Its `parse_model()` requires content, calls
`extract_json_from_chatty_response()`, and then validates the resulting
dictionary with the requested Pydantic model. Parser failures or validation
failures therefore remain distinguishable as `ValueError` or Pydantic
`ValidationError` at the agent layer.

## LLM client and resilience layers

`src/llm_client.py` creates an OpenAI SDK client using `OPENAI_BASE_URL` and
prefers `GROQ_API_KEY`, falling back to `OPENAI_API_KEY`. The default base URL
is Groq's OpenAI-compatible endpoint and the default model is
`llama-3.3-70b-versatile`.

### Message sanitizer

`sanitize_messages(messages)` rebuilds dictionary messages and allows only
`role`, `content`, `name`, `tool_call_id`, and normalized `tool_calls`
structures. Tool-call structures retain only IDs, type, and function
name/arguments, removing provider-specific extras and hidden keys.

If the sanitized history ends in a `tool` or `system` message and contains no
user message anywhere, it appends:

```json
{
  "role": "user",
  "content": "Continue execution based on the tool results."
}
```

This is a narrow template compatibility guard. It does not add a user message
when a user message already exists earlier in the history.

### JSON extraction

`extract_json_from_chatty_response(raw_text)` searches with
`r"(\{.*\})"` using DOTALL, then parses the captured text with `json.loads`.
Non-text, invalid JSON, and non-object JSON produce structured error
dictionaries rather than raw JSON-decoding exceptions. The error dictionary
can subsequently fail Pydantic model validation when passed to `parse_model`,
which is handled by the agent retry logic in the final-output paths.

## Single-Agent orchestration

`src/agents/single_agent.py` initializes a history with exactly one `system`
message containing tool and JSON-output instructions and one `user` message
containing the request.

The pure-Python `while` loop has `MAX_ITERATIONS = 10`. On each iteration it:

1. Detects the penultimate iteration as a bailout point and appends a user
   system-alert requesting final JSON without more tools.
2. Builds an OpenAI-compatible request. Normal iterations include all tool
   schemas with `tool_choice="auto"`; the bailout request uses
   `tool_choice="none"`.
3. Adds JSON response mode on bailout or after a tool result exists.
4. Passes the history through `sanitize_messages()` immediately before the SDK
   call.
5. Appends the assistant message to the local history before appending one
   `role="tool"` message per returned tool call, preserving each
   `tool_call_id`.
6. Continues if tool calls were returned; otherwise parses and validates a
   `ProcurementOutput`.

The final-output path allows two correction retries. On a `ValidationError` or
`ValueError`, it appends a user message containing the exact exception and
calls the model again with `tool_choice="none"` and JSON response mode. If all
attempts fail, the current implementation raises the last parsing or
validation exception. If the ten-iteration loop is exhausted, it raises a
`RuntimeError`.

## Staged Two-Agent orchestration

`src/agents/two_agent.py` has two distinct stages.

### Analyst

`_run_analyst()` starts with a system prompt and user request message, then
uses a loop structurally similar to the Single Agent. It exposes only the
first three tool schemas: budget, software catalog, and vendor security.
It appends assistant tool-call messages before matching tool-result messages,
sanitizes history before each SDK call, enables JSON response mode on bailout
or after tool results, and stops when it can parse a
`StructuredEvidencePack`.

The Analyst loop is capped at ten iterations and uses the same penultimate
bailout alert. Unlike the Single Agent final-output path, Analyst parsing has
no separate two-attempt self-correction loop; a malformed evidence pack
propagates as a parsing or validation exception.

### Policy/Risk Reviewer

After the Analyst returns, `run_two_agent()` runs the deterministic
`evaluate_policy_rules()` tool using an amount extracted from
`evidence_pack.budget_status`, vendor risk from `evidence_pack.vendor_risk`,
and a classification inferred from the request text.

The Reviewer history is explicitly:

```text
system: reviewer instructions and ProcurementOutput JSON template
user:   original request, JSON evidence pack, deterministic policy result
```

The evidence pack is serialized with `json.dumps(evidence_pack.model_dump())`.
The initial Reviewer call uses JSON response mode. Reviewer validation allows
two correction retries, each with a user error message, sanitized history,
`tool_choice="none"`, and JSON response mode. If those retries are exhausted,
the current code returns a fallback `ProcurementOutput` with
`recommendation="ESCALATE_TO_HUMAN"`.

## Legacy deterministic public-contract path

`src/solution.py` implements `handle_request(request_id, architecture) ->
ProcurementDecision` for the original `evals/` contract. It uses
`src.data_access.py`, not the newer agent tools. It performs deterministic
employee, budget, catalog, vendor registry, vendor-risk API/fallback,
security, privacy, legal, prompt-injection, missing-information, and approval
checks. The `staged` argument records an additional telemetry tool name but
does not invoke the LLM agent modules.

The older `src/vendor_client.py` performs an HTTP GET against
`VENDOR_RISK_BASE_URL/vendor-risk/<quoted vendor>` and raises HTTP/request
exceptions to its caller. `src/solution.py` catches request failures and may
fall back to local registry data.

## Evaluation framework

`evaluation/test_cases.json` contains ten scenarios: happy path, missing users,
catalog duplication, expired compliance, budget overrun, PII, prompt injection,
vendor-tool timeout, low-value request, and new-vendor legal review.

`evaluation/evaluator.py` runs every case through both `"single"` and
`"staged"` architectures. When no `OPENAI_API_KEY` is set, it substitutes an
offline SDK-shaped client that produces deterministic tool calls and structured
responses. It instruments `_common.TOOL_FUNCTIONS` to count tool executions,
measures elapsed time with `time.perf_counter`, records available LLM call
counts, checks golden recommendations, evidence grounding, policy fields,
human-review correctness, and next-step presence, then writes
`evaluation/benchmark_results.json` and prints a Markdown table.

The current committed benchmark results are offline-stub results. The rows
show successful checks for the ten cases in both architectures, with the
single-agent stub generally using two LLM calls and the staged stub using
three. These measurements are local orchestration timings, not remote-provider
latency.

The older `evals/run_public_evals.py` remains separate and exercises the
deterministic `src.solution.handle_request()` contract against
`evals/public_cases.json`.

## Streamlit UI and state management

The active `ui/app.py` imports the two LLM-backed runners. It prepends the
repository root to `sys.path`, loads preconfigured cases from
`evaluation/test_cases.json` with `@st.cache_data`, and offers:

- a sidebar case selector or custom request mode;
- a request text area;
- a sidebar architecture radio choice between `Single Agent` and
  `Staged Two-Agent`;
- a primary execution button;
- recommendation and next-step display;
- approval, missing-information, and risk-flag lists;
- an evidence DataFrame panel;
- expandable raw structured JSON;
- session-only Approve, Reject, and Override buttons.

The result is stored in `st.session_state["procurement_result"]`; the review
action is stored in `st.session_state["review_action"]`. A Clear result button
removes both keys. Exceptions during execution are shown with `st.error`.
Human-review buttons only record session text; they do not purchase software or
change an approval backend.

The root `app.py` is a different legacy UI. It selects a request ID from
`data/requests.json`, chooses `"single"` or `"staged"`, and displays the
deterministic `handle_request()` result. It does not use the newer
`run_single_agent()` or `run_two_agent()` functions.

## Current discrepancies and operational notes

The physical files contain some historical documentation drift:

1. `README.md` and `docs/architecture_workflow.md` state that agent loops are
   capped at five iterations, but both current agent modules set
   `MAX_ITERATIONS = 10`.
2. `README.md` describes `.env` as loaded without overwriting environment
   variables, but the active `src/llm_client.py` reads environment variables
   directly and does not call `python-dotenv`.
3. `README.md` describes the newer dashboard, while the root `app.py` and
   `run_local.py` still describe and launch the legacy deterministic UI path.
4. The active `requirements.txt` no longer lists the legacy FastAPI/Uvicorn
   dependencies used by `mock_api`, `run_local.py`, and parts of
   `verify_setup.py`.
5. `verify_dashboard.sh` checks the active `ui/app.py` health endpoint, whereas
   `verify_setup.py` validates the older starter dependencies, contract, data
   integrity, and mock API.

These are observations of the current disk state, not changes made as part of
this audit.
