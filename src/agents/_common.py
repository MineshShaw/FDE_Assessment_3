from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

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
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(line for line in lines[1:] if not line.strip().startswith("```")).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("LLM response was not valid JSON") from exc
    return model_type.model_validate(payload)


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
    tool_calls = value(message, "tool_calls", None)
    content = value(message, "content", None)
    assistant: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        assistant["tool_calls"] = [
            {
                "id": value(call, "id"),
                "type": "function",
                "function": {
                    "name": value(value(call, "function", {}), "name"),
                    "arguments": value(value(call, "function", {}), "arguments", "{}"),
                },
            }
            for call in tool_calls
        ]
    messages.append(assistant)
