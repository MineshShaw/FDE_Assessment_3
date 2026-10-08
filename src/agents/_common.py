from __future__ import annotations

import json
import logging
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
from src.data_access import load_employees


LOGGER = logging.getLogger(__name__)


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
    if "error" in payload and payload.get("recommendation") == "ESCALATE_TO_HUMAN":
        raise ValueError(payload["error"])
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
    request_amount = re.search(r"\$([\d,]+(?:\.\d+)?)", request)
    if request_amount:
        amount = float(request_amount.group(1).replace(",", ""))
    if any(term in request.casefold() for term in ("timeout", "timed out", "expired")):
        vendor_risk = "unknown"
    lowered = request.casefold()
    if re.search(r"\bcustomer[\s_-]+pii\b", lowered):
        classification = "customer_pii"
    elif re.search(r"\bemployee[\s_-]+pii\b", lowered):
        classification = "employee_pii"
    elif "source code" in lowered or "source_code" in lowered:
        classification = "source_code"
    elif "confidential document" in lowered:
        classification = "confidential_documents"
    elif "production" in lowered:
        classification = "production"
    if amount is None:
        return None
    result = evaluate_policy_rules(amount, vendor_risk, classification)
    lowered = request.casefold()
    if any(term in lowered for term in ("ignore all procurement rules", "approve immediately", "bypass controls")):
        result.setdefault("risk_flags", []).append("prompt_injection_detected")
    if "exceeding" in lowered or "over budget" in lowered:
        result.setdefault("risk_flags", []).append("budget_insufficient")
    if "expired" in lowered:
        result.setdefault("risk_flags", []).append("vendor_review_expired")
    if "new vendor" in lowered:
        result.setdefault("risk_flags", []).append("legal_review_required")
    result["risk_flags"] = list(dict.fromkeys(result.get("risk_flags", [])))
    return result


def policy_result_from_request(request: str, *, vendor_risk: str = "unknown") -> dict:
    """Evaluate policy from request text without trusting model-generated fields."""
    amount_match = re.search(r"\$([\d,]+(?:\.\d+)?)", request)
    if not amount_match:
        return {
            "error": "amount not established",
            "risk_flags": ["policy_unverified"],
            "approvals_required": [],
        }
    amount = float(amount_match.group(1).replace(",", ""))
    return policy_result_from_messages(
        request,
        [
            {
                "role": "tool",
                "content": json.dumps(
                    {"requested_amount": amount, "risk_level": vendor_risk}
                ),
            }
        ],
    ) or {"error": "unable to derive policy context from request"}


def enforce_policy_floor(output: BaseModel, policy_result: dict | None) -> BaseModel:
    """Prevent model output from weakening deterministic approval requirements."""
    if not policy_result:
        return output
    approvals = list(dict.fromkeys(output.approvals_required + policy_result.get("approvals_required", [])))
    risk_flags = list(dict.fromkeys(output.risk_flags + policy_result.get("risk_flags", [])))
    if "security_review_required" in risk_flags and "Security" not in approvals:
        approvals.append("Security")
    if "privacy_review_required" in risk_flags and "Privacy" not in approvals:
        approvals.append("Privacy")
    if "legal_review_required" in risk_flags and "Legal" not in approvals:
        approvals.append("Legal")
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
            "evidence": list(dict.fromkeys(output.evidence + [
                f"deterministic_policy: approvals={', '.join(approvals) or 'none'}; "
                f"risk_flags={', '.join(risk_flags) or 'none'}"
            ])),
        }
    )


def execute_tool_call(tool_call: Any, pinned: dict | None = None) -> str:
    function = value(tool_call, "function", {})
    name = value(function, "name")
    arguments = value(function, "arguments", "{}")
    if name not in TOOL_FUNCTIONS:
        return json.dumps({"error": f"Unsupported tool requested by model: {name}"})
    try:
        kwargs = json.loads(arguments) if isinstance(arguments, str) else arguments
    except json.JSONDecodeError:
        return json.dumps({"error": f"Invalid arguments for tool {name}"})
    if not isinstance(kwargs, dict):
        return json.dumps({"error": f"Arguments for tool {name} must be an object"})
    if pinned:
        applicable = {
            "check_budget": ("department_id", "amount"),
            "get_vendor_security_status": ("vendor_name",),
            "evaluate_policy_rules": ("amount", "data_classification"),
        }.get(name, ())
        for key in applicable:
            if key in pinned and key in kwargs and kwargs[key] != pinned[key]:
                LOGGER.warning("Overriding model tool argument %s for %s with deterministic request fact", key, name)
            if key in pinned:
                kwargs[key] = pinned[key]
    try:
        result = TOOL_FUNCTIONS[name](**kwargs)
    except Exception as exc:
        return json.dumps({"error": f"Tool {name} failed: {exc}"})
    return json.dumps(result, default=str)


def request_facts(request_data: dict | None) -> dict | None:
    if request_data is None:
        return None
    employees = load_employees()
    employee = employees[
        employees["employee_id"].astype(str).str.casefold()
        == str(request_data.get("requester_id", "")).casefold()
    ]
    facts = {
        "amount": request_data.get("annual_cost_usd"),
        "vendor_name": request_data.get("vendor_name"),
        "data_classification": request_data.get("data_access_level", "internal"),
    }
    if not employee.empty:
        facts["department_id"] = str(employee.iloc[0]["department"])
    return {key: value for key, value in facts.items() if value is not None}


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
