from __future__ import annotations

from src.agents._common import (
    TOOL_SCHEMAS,
    append_assistant_message,
    execute_tool_call,
    message_from_response,
    parse_model,
    value,
)
from src.llm_client import MODEL_NAME, client
from src.schemas import ProcurementOutput


MAX_ITERATIONS = 5


def run_single_agent(request: str) -> ProcurementOutput:
    if client is None:
        raise RuntimeError("OPENAI_API_KEY is required to run the single-agent architecture")
    messages: list[dict] = [
        {
            "role": "system",
            "content": (
                "You are an analytical agent. Call the provided tools to gather evidence. "
                "Once you have sufficient evidence, DO NOT call any more tools. You must output "
                "your final decision as a raw JSON object matching the ProcurementOutput schema. "
                "Do not include markdown formatting, code blocks, or explanatory text outside the JSON."
            ),
        },
        {"role": "user", "content": request},
    ]

    iteration = 0
    while iteration < MAX_ITERATIONS:
        iteration += 1
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
        )
        message = message_from_response(response)
        tool_calls = value(message, "tool_calls", None)
        append_assistant_message(messages, message)
        if tool_calls:
            for tool_call in tool_calls:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": value(tool_call, "id"),
                        "content": execute_tool_call(tool_call),
                    }
                )
            continue
        return parse_model(value(message, "content"), ProcurementOutput)

    raise RuntimeError(f"Single-agent orchestration exceeded {MAX_ITERATIONS} iterations")
