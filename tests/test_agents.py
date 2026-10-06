from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from src.agents import single_agent, two_agent
from src.schemas import ProcurementOutput
from src.schemas import StructuredEvidencePack


def _response(message: dict) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(**message))])


def _tool_call(name: str, arguments: dict, call_id: str = "call-1") -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def _mock_client(responses):
    def create(**kwargs):
        return next(responses)

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def test_standard_assistant_message_is_appended_for_tool_loop() -> None:
    from src.agents._common import append_assistant_message

    messages = []
    append_assistant_message(
        messages,
        SimpleNamespace(
            role="assistant",
            content=None,
            tool_calls=[_tool_call("check_budget", {})],
        ),
    )

    assert messages[0]["role"] == "assistant"
    assert messages[0]["tool_calls"][0]["function"]["name"] == "check_budget"


def test_parser_accepts_json_code_fence() -> None:
    from src.agents._common import parse_model

    result = parse_model(
        '```json\n{"recommendation":"REQUEST_INFO","next_step":"Ask for cost."}\n```',
        ProcurementOutput,
    )
    assert result.recommendation == "REQUEST_INFO"


def test_schemas_supply_defaults_and_coerce_single_overlap() -> None:
    assert ProcurementOutput(recommendation="APPROVE").next_step == "Pending manual review"
    pack = StructuredEvidencePack(tool_overlap={"name": "TaskFlow"})
    assert pack.tool_overlap == [{"name": "TaskFlow"}]


def test_single_agent_executes_tool_then_parses_output(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = iter(
        [
            _response({"content": None, "tool_calls": [_tool_call("check_budget", {"department_id": "Marketing", "amount": 5000})]}),
            _response(
                {
                    "content": json.dumps(
                        {
                            "recommendation": "APPROVE",
                            "evidence": ["Budget has sufficient remaining funds"],
                            "approvals_required": ["Manager"],
                            "missing_information": [],
                            "risk_flags": [],
                            "next_step": "Obtain manager approval.",
                        }
                    ),
                    "tool_calls": None,
                }
            ),
        ]
    )
    calls = []
    original = _mock_client(responses)
    original.chat.completions.create = lambda **kwargs: calls.append(deepcopy(kwargs)) or next(responses)
    monkeypatch.setattr(single_agent, "client", original)

    result = single_agent.run_single_agent("Marketing requests a $5,000 internal tool.")

    assert result.recommendation == "APPROVE"
    assert len(calls) == 2
    assert calls[1]["messages"][-1]["role"] == "tool"


def test_single_agent_has_a_hard_iteration_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    response = _response({"content": None, "tool_calls": [_tool_call("check_budget", {"department_id": "Marketing", "amount": 1})]})
    mock_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: response)
        )
    )
    monkeypatch.setattr(single_agent, "client", mock_client)

    with pytest.raises(RuntimeError, match="exceeded 10"):
        single_agent.run_single_agent("Keep gathering evidence.")


def test_single_agent_forces_tool_termination_before_final_iteration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    final_output = _response(
        {
            "content": json.dumps(
                {
                    "recommendation": "REQUEST_INFO",
                    "next_step": "Ask for missing details.",
                }
            ),
            "tool_calls": None,
        }
    )
    tool_output = _response(
        {
            "content": None,
            "tool_calls": [_tool_call("check_budget", {"department_id": "Marketing", "amount": 1})],
        }
    )

    def create(**kwargs):
        calls.append(kwargs)
        return final_output if kwargs["tool_choice"] == "none" else tool_output

    monkeypatch.setattr(
        single_agent,
        "client",
        SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
    )
    result = single_agent.run_single_agent("A request that needs evidence.")

    assert result.recommendation == "REQUEST_INFO"
    assert calls[-1]["tool_choice"] == "none"
    assert len(calls) == 9


def test_two_agent_analyst_loop_and_reviewer_parse(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = iter(
        [
            _response({"content": None, "tool_calls": [_tool_call("check_budget", {"department_id": "Marketing", "amount": 5000})]}),
            _response(
                {
                    "content": json.dumps(
                        {
                            "budget_status": {"amount": 5000, "remaining_funds": 10000},
                            "tool_overlap": [],
                            "vendor_risk": {"risk_level": "low"},
                        }
                    ),
                    "tool_calls": None,
                }
            ),
            _response(
                {
                    "content": json.dumps(
                        {
                            "recommendation": "ESCALATE_TO_HUMAN",
                            "evidence": ["Budget check completed"],
                            "approvals_required": ["Department Head", "Procurement"],
                            "missing_information": [],
                            "risk_flags": [],
                            "next_step": "Route to human approvers.",
                        }
                    ),
                    "tool_calls": None,
                }
            ),
        ]
    )
    calls = []
    original = _mock_client(responses)
    original.chat.completions.create = lambda **kwargs: calls.append(deepcopy(kwargs)) or next(responses)
    monkeypatch.setattr(two_agent, "client", original)

    result = two_agent.run_two_agent("Marketing requests a $5,000 internal tool.")

    assert result.recommendation == "ESCALATE_TO_HUMAN"
    assert len(calls) == 3
    assert "Deterministic policy result" in calls[-1]["messages"][0]["content"]
