# Architecture Decision Memo

**Which architecture would I ship—and why?** Ship **Single Agent as the default**, with Staged Two-Agent retained as an experimental audit-oriented option.

**Evidence.** The repository now contains a policy-derived 16-case gold set and a real-run evaluator that records per-run latency, LLM/tool calls, traces, model, endpoint host, UTC time, and git SHA. `--offline-smoke` intentionally writes no results. A bounded live-provider attempt in this session did not complete, so there is no defensible live accuracy or latency number to cite and the prior offline artifact is not shipping evidence.

**Decision rationale.** Single Agent has one fewer model call and a smaller failure surface. Both paths now apply deterministic policy floors after model output, but Staged still adds an Analyst-to-Reviewer handoff whose correctness depends on the evidence pack and request-derived fallback context. Staged remains useful for audit experiments because the typed handoff is inspectable, but it should not be the shipping default until repeated live-provider runs demonstrate equal or better correctness and latency.

**Tradeoff and guardrails.** Single Agent remains the conservative default because it uses one fewer model call and has a smaller failure surface. Staged remains an audit-oriented option, not a proven winner. Before submission, run `evaluation/evaluator.py --runs 3 --sleep 2` with a configured provider and commit only the resulting timestamped real-run artifact; both paths use deterministic policy floors, retries, sanitized messages, Pydantic validation, and human approval.
