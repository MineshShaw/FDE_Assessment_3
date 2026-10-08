from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from src.llm_client import extract_json_from_chatty_response
from src.tools import (
    check_budget,
    evaluate_policy_rules,
    get_vendor_security_status,
    search_software_catalog,
)


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "check_budget",
            "description": "Compare a requested amount with a department's available software budget.",
            "parameters": {
                "type": "object",
                "properties": {
                    "department_id": {"type": "string"},
                    "amount": {"type": "number"},
                },
                "required": ["department_id", "amount"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_software_catalog",
            "description": "Find existing software that overlaps with a stated need.",
            "parameters": {
                "type": "object",
                "properties": {
                    "need_description": {"type": "string"},
                    "category": {"type": "string"},
                },
                "required": ["need_description", "category"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_vendor_security_status",
            "description": "Get vendor registry and risk-service security information.",
            "parameters": {"type": "object", "properties": {"vendor_name": {"type": "string"}}, "required": ["vendor_name"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "evaluate_policy_rules",
            "description": "Evaluate deterministic approval and data-classification rules.",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {"type": "number"},
                    "vendor_risk": {"type": "string"},
                    "data_classification": {"type": "string"},
                },
                "required": ["amount", "vendor_risk", "data_classification"],
            },
        },
    },
]

TOOL_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "check_budget": check_budget,
    "search_software_catalog": search_software_catalog,
    "get_vendor_security_status": get_vendor_security_status,
    "evaluate_policy_rules": evaluate_policy_rules,
}


def value(response: Any, name: str, default: Any = None) -> Any:
    if isinstance(response, dict):
        return response.get(name, default)
    return getattr(response, name, default)


def message_from_response(response: Any) -> Any:
    choices = value(response, "choices", [])
    if not choices:
        raise ValueError("LLM response did not contain a choice")
    return value(choices[0], "message", {})


def parse_model(content: str | None, model_type: type[BaseModel]) -> BaseModel:
    if not content:
        raise ValueError("LLM response did not contain structured content")
    payload = extract_json_from_chatty_response(content)
    return model_type.model_validate(payload)


def policy_result_from_messages(request: str, messages: list[dict]) -> dict | None:
    """Build a deterministic policy result from tool evidence in the history."""
    amount: float | None = None
    vendor_risk = "unknown"
    classification = "internal"
    for message in messages:
        if message.get("role") != "tool":
            continue
        try:
            payload = json.loads(message.get("content", ""))
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        if "requested_amount" in payload:
            amount = float(payload["requested_amount"])
        if "risk_level" in payload:
            vendor_risk = str(payload["risk_level"])
    lowered = request.casefold()
    for candidate in ("customer_pii", "employee_pii", "source_code", "confidential_documents", "production"):
        if candidate in lowered:
            classification = candidate
            break
    if amount is None:
        return None
    return evaluate_policy_rules(amount, vendor_risk, classification)


def enforce_policy_floor(output: BaseModel, policy_result: dict | None) -> BaseModel:
    """Prevent model output from weakening deterministic approval requirements."""
    if not policy_result or "error" in policy_result:
        return output
    approvals = list(dict.fromkeys(output.approvals_required + policy_result.get("approvals_required", [])))
    risk_flags = list(dict.fromkeys(output.risk_flags + policy_result.get("risk_flags", [])))
    recommendation = output.recommendation
    if output.missing_information:
        recommendation = "REQUEST_INFO"
    elif approvals or risk_flags:
        recommendation = "ESCALATE_TO_HUMAN"
    return output.model_copy(
        update={
            "recommendation": recommendation,
            "approvals_required": approvals,
            "risk_flags": risk_flags,
        }
    )


def execute_tool_call(tool_call: Any) -> str:
    function = value(tool_call, "function", {})
    name = value(function, "name")
    arguments = value(function, "arguments", "{}")
    if name not in TOOL_FUNCTIONS:
        raise ValueError(f"Unsupported tool requested by model: {name}")
    try:
        kwargs = json.loads(arguments) if isinstance(arguments, str) else arguments
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid arguments for tool {name}") from exc
    result = TOOL_FUNCTIONS[name](**kwargs)
    return json.dumps(result, default=str)


def append_assistant_message(messages: list[dict], message: Any) -> None:
    if hasattr(message, "model_dump"):
        messages.append(message.model_dump(exclude_unset=True))
    elif isinstance(message, dict):
        messages.append(dict(message))
    else:
        messages.append(
            {
                "role": value(message, "role"),
                "content": value(message, "content"),
                "tool_calls": [
                    {
                        "id": value(call, "id"),
                        "type": value(call, "type", "function"),
                        "function": {
                            "name": value(value(call, "function", {}), "name"),
                            "arguments": value(value(call, "function", {}), "arguments", "{}"),
                        },
                    }
                    for call in (value(message, "tool_calls", None) or [])
                ],
            }
        )
