from __future__ import annotations

import json
import os
import re
from typing import Any

from openai import OpenAI


def sanitize_messages(messages: list) -> list:
    """Return only provider-neutral OpenAI chat message fields."""
    sanitized: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        clean: dict[str, Any] = {}
        for key in ("role", "content", "name", "tool_call_id"):
            if key in message:
                clean[key] = message[key]
        if "tool_calls" in message:
            tool_calls = []
            for call in message["tool_calls"] or []:
                if not isinstance(call, dict):
                    continue
                clean_call: dict[str, Any] = {}
                for key in ("id", "type"):
                    if key in call:
                        clean_call[key] = call[key]
                function = call.get("function")
                if isinstance(function, dict):
                    clean_call["function"] = {
                        key: function[key]
                        for key in ("name", "arguments")
                        if key in function
                    }
                tool_calls.append(clean_call)
            clean["tool_calls"] = tool_calls
        if clean:
            sanitized.append(clean)

    if sanitized and (
        sanitized[-1].get("role") in {"tool", "system"}
        and not any(message.get("role") == "user" for message in sanitized)
    ):
        sanitized.append(
            {
                "role": "user",
                "content": "Continue execution based on the tool results.",
            }
        )
    return sanitized


def extract_json_from_chatty_response(raw_text: str) -> dict:
    """Extract a JSON object from model prose without leaking parser exceptions."""
    if not isinstance(raw_text, str):
        return {"error": "LLM response was not text"}
    match = re.search(r"(\{.*\})", raw_text, re.DOTALL)
    candidate = match.group(1) if match else raw_text.strip()
    try:
        payload = json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        return {"error": "LLM response did not contain valid JSON", "raw_response": raw_text[:500]}
    return payload if isinstance(payload, dict) else {"error": "LLM response JSON was not an object"}


def create_client() -> OpenAI:
    return OpenAI(
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1"),
        api_key=os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY"),
    )


client = create_client() if (os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")) else None
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
