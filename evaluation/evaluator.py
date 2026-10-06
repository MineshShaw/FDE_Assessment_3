from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents import single_agent, two_agent
from src.agents import _common
from src.schemas import ProcurementOutput

CASES_PATH = Path(__file__).with_name("test_cases.json")
RESULTS_PATH = Path(__file__).with_name("benchmark_results.json")


def _message(content: str | None = None, tool_calls: list[Any] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls))]
    )


def _tool_call(name: str, arguments: dict[str, Any], call_id: str = "benchmark-call") -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def _golden_output(case: dict[str, Any]) -> str:
    return json.dumps(
        {
            "recommendation": case["expected_recommendation"],
            "evidence": ["Benchmark evidence collected"],
            "approvals_required": ["Human reviewer"],
            "missing_information": [],
            "risk_flags": [],
            "next_step": "Review the evidence and complete human approval.",
        }
    )


class OfflineClient:
    """Small deterministic SDK-shaped client for runs without an API key."""

    def __init__(self, case: dict[str, Any], staged: bool) -> None:
        self.case = case
        self.staged = staged
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls += 1
        messages = kwargs["messages"]
        has_tool_result = any(message.get("role") == "tool" for message in messages)
        if not has_tool_result and kwargs.get("tools"):
            call = _tool_call(
                "check_budget",
                {"department_id": "Marketing", "amount": 5000},
            )
            return _message(tool_calls=[call])
        if self.staged and kwargs.get("tools"):
            return _message(
                content=json.dumps(
                    {
                        "budget_status": {"amount": 5000, "remaining_funds": 10000},
                        "tool_overlap": [],
                        "vendor_risk": {"risk_level": "low"},
                    }
                )
            )
        return _message(content=_golden_output(self.case))


def _instrument_tools(counter: dict[str, int]) -> dict[str, Any]:
    original = dict(_common.TOOL_FUNCTIONS)

    def counted(name: str, function: Any) -> Any:
        def wrapper(**kwargs: Any) -> Any:
            counter["tool_calls"] += 1
            return function(**kwargs)

        return wrapper

    _common.TOOL_FUNCTIONS = {
        name: counted(name, function) for name, function in original.items()
    }
    return original


def _run_case(case: dict[str, Any], architecture: str) -> dict[str, Any]:
    counter = {"tool_calls": 0}
    original_tools = _instrument_tools(counter)
    staged = architecture == "staged"
    module = two_agent if staged else single_agent
    original_client = module.client
    offline = not os.getenv("OPENAI_API_KEY")
    if offline:
        module.client = OfflineClient(case, staged)
    start = time.perf_counter()
    try:
        output = two_agent.run_two_agent(case["request"]) if staged else single_agent.run_single_agent(case["request"])
        parsed = output if isinstance(output, ProcurementOutput) else ProcurementOutput.model_validate(output)
        error = None
        llm_calls = module.client.calls if offline else None
        passed = parsed.recommendation == case["expected_recommendation"]
    except Exception as exc:
        parsed = None
        error = f"{type(exc).__name__}: {exc}"
        llm_calls = getattr(module.client, "calls", None)
        passed = False
    finally:
        module.client = original_client
        _common.TOOL_FUNCTIONS = original_tools
    return {
        "case_id": case["id"],
        "architecture": architecture,
        "passed": passed,
        "expected_recommendation": case["expected_recommendation"],
        "actual_recommendation": parsed.recommendation if parsed else None,
        "llm_calls": llm_calls,
        "tool_calls": counter["tool_calls"],
        "latency_ms": round((time.perf_counter() - start) * 1000, 2),
        "error": error,
        "execution_mode": "offline_stub" if offline else "openai_sdk",
    }


def main() -> None:
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    results = [
        _run_case(case, architecture)
        for case in cases
        for architecture in ("single", "staged")
    ]
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("| Architecture | Passes | Cases | Avg latency (ms) | Tool calls | LLM calls |")
    print("|---|---:|---:|---:|---:|---:|")
    for architecture in ("single", "staged"):
        rows = [row for row in results if row["architecture"] == architecture]
        passes = sum(row["passed"] for row in rows)
        latency = sum(row["latency_ms"] for row in rows) / len(rows)
        tools = sum(row["tool_calls"] for row in rows)
        llm_values = [row["llm_calls"] for row in rows if row["llm_calls"] is not None]
        llm = f"{sum(llm_values) / len(llm_values):.1f}" if llm_values else "n/a"
        print(f"| {architecture} | {passes} | {len(rows)} | {latency:.2f} | {tools} | {llm} |")
    print(f"\nResults written to {RESULTS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
