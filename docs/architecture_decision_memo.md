# Architecture Decision Memo

**Decision.** Use Single Agent as the default and retain Staged Two-Agent as an experimental audit option. The simpler path uses fewer model interactions and has a smaller provider failure surface. Both paths share deterministic policy floors and human approval.

**Evidence status.** No completed real-provider result file is currently present under `evaluation/results/`. The tracked `evaluation/benchmark_results.json` is not a real-run artifact and is not used as shipping evidence. Consequently, this memo deliberately reports no latency, call-count, accuracy, or reliability numbers. A reproducible live command is:

```bash
PYTHONPATH=. .venv/bin/python evaluation/evaluator.py --runs 3 --sleep 2
```

That command requires a provider key, runs both architectures on identical policy-derived cases, includes the deterministic baseline, and writes a timestamped result containing model, endpoint host, UTC time, git SHA, per-case outcomes, latency, LLM calls, tool calls, and failures.

| Metric | Single Agent | Staged Two-Agent | Policy engine only |
|---|---|---|---|
| Correct recommendation | pending real run | pending real run | pending real run |
| Approval precision / recall | pending real run | pending real run | pending real run |
| Grounded evidence | pending real run | pending real run | pending real run |
| Escalation correctness | pending real run | pending real run | pending real run |
| Mean latency | pending real run | pending real run | pending real run |
| Mean LLM calls | pending real run | pending real run | zero by design |
| Mean tool calls | pending real run | pending real run | pending real run |
| Failures | pending real run | pending real run | pending real run |

**Limits.** The case set is synthetic and finite, and model output is nondeterministic. The decision must be revisited after repeated live runs; smoke mode is plumbing validation only, not a benchmark.
