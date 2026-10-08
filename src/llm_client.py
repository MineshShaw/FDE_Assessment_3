from __future__ import annotations

import json
import os
import time
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


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
    """Extract the first valid JSON object from model prose."""
    if not isinstance(raw_text, str):
        return {"recommendation": "ESCALATE_TO_HUMAN", "error": "Unparseable LLM output"}

    decoder = json.JSONDecoder()
    for index, character in enumerate(raw_text):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(raw_text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload

    return {"recommendation": "ESCALATE_TO_HUMAN", "error": "Unparseable LLM output"}


def create_client() -> OpenAI:
    return OpenAI(
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1"),
        api_key=os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY"),
    )


client = create_client() if (os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")) else None
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")


def chat_completion_with_retry(active_client: Any, **kwargs: Any) -> Any:
    """Call an OpenAI-compatible provider with bounded retry/backoff."""
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            return active_client.chat.completions.create(**kwargs)
        except Exception as exc:
            last_error = exc
            text = str(exc).casefold()
            retryable = any(
                marker in text
                for marker in ("429", "rate limit", "timeout", "timed out", "temporarily unavailable")
            )
            if not retryable or attempt == 2:
                raise
            time.sleep(2**attempt)
    raise RuntimeError(f"LLM request failed after retries: {last_error}")
