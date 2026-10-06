from __future__ import annotations

import os
from typing import Any

from openai import OpenAI


def create_client() -> OpenAI:
    return OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.getenv("OPENAI_BASE_URL"),
    )


client = create_client() if os.getenv("OPENAI_API_KEY") else None
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.0-flash")


def format_assistant_message(message: Any) -> dict:
    """Serialize an SDK message without dropping provider-specific metadata."""
    if hasattr(message, "model_dump"):
        msg_dict = message.model_dump(exclude_unset=True)
    elif isinstance(message, dict):
        msg_dict = dict(message)
    else:
        msg_dict = {
            key: value
            for key, value in vars(message).items()
            if not key.startswith("_") and key != "model_extra"
        }
        if getattr(message, "tool_calls", None):
            msg_dict["tool_calls"] = [
                {
                    key: value
                    for key, value in vars(tool_call).items()
                    if not key.startswith("_") and key != "model_extra" and key != "function"
                }
                | {
                    "function": {
                        key: value
                        for key, value in vars(tool_call.function).items()
                        if not key.startswith("_") and key != "model_extra"
                    }
                }
                for tool_call in message.tool_calls
            ]

    model_extra = getattr(message, "model_extra", None)
    if model_extra:
        msg_dict.update(model_extra)

    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        serialized_tool_calls = msg_dict.setdefault("tool_calls", [])
        for index, tool_call in enumerate(tool_calls):
            if index >= len(serialized_tool_calls):
                serialized_tool_calls.append({})
            tool_extra = getattr(tool_call, "model_extra", None)
            if tool_extra:
                serialized_tool_calls[index].update(tool_extra)
            function = getattr(tool_call, "function", None)
            function_extra = getattr(function, "model_extra", None)
            if function_extra:
                serialized_tool_calls[index].setdefault("function", {}).update(function_extra)

    return msg_dict
