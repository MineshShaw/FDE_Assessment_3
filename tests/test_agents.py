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


def test_parser_extracts_json_from_chatty_text() -> None:
    from src.agents._common import parse_model

    result = parse_model(
        'Here is the decision: {"recommendation":"REJECT","next_step":"Stop purchase."} Thanks.',
        ProcurementOutput,
    )
    assert result.recommendation == "REJECT"


def test_provider_neutral_message_sanitizer_strips_extras_and_adds_user() -> None:
    from src.llm_client import sanitize_messages

    result = sanitize_messages(
        [
            {"role": "system", "content": "Rules", "model_extra": {"trace": "secret"}},
            {
                "role": "tool",
                "content": '{"ok": true}',
                "tool_call_id": "call-1",
                "hidden": "removed",
            },
        ]
    )

    assert result[-1] == {
        "role": "user",
        "content": "Continue execution based on the tool results.",
    }
    assert "model_extra" not in result[0]
    assert "hidden" not in result[1]


def test_json_extractor_returns_structured_error() -> None:
    from src.llm_client import extract_json_from_chatty_response

    assert extract_json_from_chatty_response("not json")["error"]


def test_schemas_supply_defaults_and_coerce_single_overlap() -> None:
    assert ProcurementOutput(recommendation="APPROVE").next_step == "Manual review required."
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
    assert calls[1]["tool_choice"] == "auto"
    assert calls[1].get("response_format") is None
    assert calls[0].get("response_format") is None


def test_single_agent_retries_invalid_final_json(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = iter(
        [
            _response({"content": '{"recommendation":"NOT_VALID"}', "tool_calls": None}),
            _response({"content": '{"recommendation":"REQUEST_INFO"}', "tool_calls": None}),
        ]
    )
    calls = []

    def create(**kwargs):
        calls.append(deepcopy(kwargs))
        return next(responses)

    monkeypatch.setattr(
        single_agent,
        "client",
        SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
    )
    result = single_agent.run_single_agent("Need a procurement decision.")

    assert result.recommendation == "REQUEST_INFO"
    assert len(calls) == 2
    assert calls[1]["response_format"] == {"type": "json_object"}
    assert "failed validation" in calls[1]["messages"][-1]["content"]


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
    assert [message["role"] for message in calls[-1]["messages"]] == ["system", "user"]
    assert "Original Request:" in calls[-1]["messages"][1]["content"]
    assert "Evidence Pack from Analyst:" in calls[-1]["messages"][1]["content"]
    assert "Deterministic policy result" in calls[-1]["messages"][1]["content"]
    assert calls[-1]["tool_choice"] == "none"
    assert calls[-1]["response_format"] == {"type": "json_object"}


def test_two_agent_retries_invalid_analyst_evidence_pack(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = iter(
        [
            _response({"content": '{"budget_status": [], "tool_overlap": [], "vendor_risk": {}}', "tool_calls": None}),
            _response(
                {
                    "content": json.dumps(
                        {
                            "budget_status": {"amount": 5000},
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
                            "recommendation": "REQUEST_INFO",
                            "next_step": "Review the request.",
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

    result = two_agent.run_two_agent("Marketing requests a procurement decision.")

    assert result.recommendation == "REQUEST_INFO"
    assert calls[0].get("response_format") is None
    assert calls[1]["tool_choice"] == "none"
    assert calls[1]["response_format"] == {"type": "json_object"}
    assert "failed validation" in calls[1]["messages"][-1]["content"]
