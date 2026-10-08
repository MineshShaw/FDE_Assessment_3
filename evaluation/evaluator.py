from __future__ import annotations

import json
import os
import re
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


def _request_amount(request: str) -> float:
    match = re.search(r"\$([\d,]+(?:\.\d+)?)", request)
    return float(match.group(1).replace(",", "")) if match else 0.0


def _request_department(request: str) -> str:
    for department in ("Marketing", "Finance", "Engineering", "Customer Success", "Sales"):
        if department.casefold() in request.casefold():
            return department
    return "Marketing"


def _request_classification(request: str) -> str:
    lowered = request.casefold()
    for classification in ("customer_pii", "employee_pii", "source_code", "confidential_documents", "production"):
        if classification in lowered:
            return classification
    return "internal"


def _offline_output(request: str, tool_result: str) -> dict[str, Any]:
    """Model-independent fixture behavior derived from request and tool evidence."""
    missing = any(
        phrase in request.casefold()
        for phrase in ("not provided", "provides no", "no annual cost", "no user count", "no data classification")
    )
    amount = _request_amount(request)
    policy = _common.TOOL_FUNCTIONS["evaluate_policy_rules"](
        amount,
        "unknown" if "timeout" in request.casefold() or "expired" in request.casefold() else "low",
        _request_classification(request),
    )
    return {
        "recommendation": "REQUEST_INFO" if missing else "ESCALATE_TO_HUMAN",
        "evidence": [f"Tool result: {tool_result}" if tool_result else "Request evidence collected"],
        "approvals_required": policy.get("approvals_required", ["Human reviewer"]),
        "missing_information": ["material request details"] if missing else [],
        "risk_flags": policy.get("risk_flags", []),
        "next_step": "Provide missing request details." if missing else "Complete human review before procurement.",
    }


class OfflineClient:
    """Small deterministic SDK-shaped client for runs without an API key."""

    def __init__(self, case: dict[str, Any], staged: bool) -> None:
        self.request = case["request"]
        self.staged = staged
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        self.last_tool_result = ""

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls += 1
        messages = kwargs["messages"]
        has_tool_result = any(message.get("role") == "tool" for message in messages)
        for message in messages:
            if message.get("role") == "tool":
                self.last_tool_result = message.get("content", "")
        if not has_tool_result and kwargs.get("tools"):
            call = _tool_call(
                "check_budget",
                {"department_id": "Marketing", "amount": 5000},
            )
            return _message(tool_calls=[call])
        if self.staged and kwargs.get("tools"):
            try:
                budget_status = json.loads(self.last_tool_result)
            except json.JSONDecodeError:
                budget_status = {"status": "not found"}
            return _message(
                content=json.dumps(
                    {
                        "budget_status": budget_status,
                        "tool_overlap": [],
                        "vendor_risk": {"risk_level": "low"},
                    }
                )
            )
        if self.staged and not kwargs.get("tools"):
            return _message(content=json.dumps(_offline_output(self.request, self.last_tool_result)))
        return _message(content=json.dumps(_offline_output(self.request, self.last_tool_result)))


class CallCountingClient:
    def __init__(self, original: Any) -> None:
        self.original = original
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs: Any) -> Any:
        self.calls += 1
        return self.original.chat.completions.create(**kwargs)


def _instrument_tools(counter: dict[str, Any]) -> dict[str, Any]:
    original = dict(_common.TOOL_FUNCTIONS)

    def counted(name: str, function: Any) -> Any:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            counter["tool_calls"] += 1
            result = function(*args, **kwargs)
            counter["tool_results"].append(json.dumps(result, default=str))
            return result

        return wrapper

    _common.TOOL_FUNCTIONS = {
        name: counted(name, function) for name, function in original.items()
    }
    return original


def _run_case(case: dict[str, Any], architecture: str) -> dict[str, Any]:
    counter = {"tool_calls": 0, "tool_results": []}
    original_tools = _instrument_tools(counter)
    staged = architecture == "staged"
    module = two_agent if staged else single_agent
    original_client = module.client
    offline = not (os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY"))
    if offline:
        module.client = OfflineClient(case, staged)
    else:
        module.client = CallCountingClient(original_client)
    start = time.perf_counter()
    try:
        output = two_agent.run_two_agent(case["request"]) if staged else single_agent.run_single_agent(case["request"])
        parsed = output if isinstance(output, ProcurementOutput) else ProcurementOutput.model_validate(output)
        error = None
        llm_calls = module.client.calls
        evidence_text = " ".join(parsed.evidence).casefold()
        tool_text = " ".join(counter["tool_results"]).casefold()
        evidence_grounded = bool(parsed.evidence) and any(
            source in evidence_text for source in ("tool result:", "deterministic_policy:")
        ) and bool(tool_text)
        expected_policy = _common.TOOL_FUNCTIONS["evaluate_policy_rules"](
            _request_amount(case["request"]),
            "unknown" if any(word in case["request"].casefold() for word in ("timeout", "expired")) else "low",
            _request_classification(case["request"]),
        )
        policy_rules_followed = set(expected_policy.get("approvals_required", [])) <= set(parsed.approvals_required)
        policy_rules_followed = policy_rules_followed and set(expected_policy.get("risk_flags", [])) <= set(parsed.risk_flags)
        human_review_correct = parsed.recommendation == case["expected_recommendation"]
        next_action_present = bool(parsed.next_step.strip())
        passed = (
            parsed.recommendation == case["expected_recommendation"]
            and evidence_grounded
            and policy_rules_followed
            and human_review_correct
            and next_action_present
        )
    except Exception as exc:
        parsed = None
        error = f"{type(exc).__name__}: {exc}"
        llm_calls = getattr(module.client, "calls", None)
        passed = False
        evidence_grounded = False
        policy_rules_followed = False
        human_review_correct = False
        next_action_present = False
    finally:
        module.client = original_client
        _common.TOOL_FUNCTIONS = original_tools
    return {
        "case_id": case["id"],
        "architecture": architecture,
        "passed": passed,
        "expected_recommendation": case["expected_recommendation"],
        "actual_recommendation": parsed.recommendation if parsed else None,
        "evidence_grounded": evidence_grounded,
        "policy_rules_followed": policy_rules_followed,
        "human_review_correct": human_review_correct,
        "next_action_present": next_action_present,
        "llm_calls": llm_calls,
        "tool_calls": counter["tool_calls"],
        "latency_ms": round((time.perf_counter() - start) * 1000, 2),
        "error": error,
        "execution_mode": "offline_stub" if offline else "openai_sdk",
    }


def main() -> None:
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    results = []
    live = bool(os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY"))
    for case in cases:
        for architecture in ("single", "staged"):
            results.append(_run_case(case, architecture))
        if live:
            time.sleep(2)
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
