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


MAX_ITERATIONS = 10


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
                "Do not include markdown formatting, code blocks, or explanatory text outside the JSON. "
                "You MUST output a valid JSON object matching this exact schema. Do NOT output tool "
                "variables like 'approved' or 'amount'. Map your final decision strictly to the "
                "'recommendation' key (choose from: APPROVE, REJECT, ESCALATE_TO_HUMAN, REQUEST_INFO).\n"
                "Call each relevant evidence tool at most once. Do not repeat tool calls. Once you "
                "receive tool responses, immediately synthesize the final JSON and stop calling tools.\n"
                '{"recommendation": "APPROVE | REJECT | ESCALATE_TO_HUMAN | REQUEST_INFO", '
                '"evidence": ["..."], "approvals_required": ["..."], "missing_information": [], '
                '"risk_flags": [], "next_step": "..."}'
            ),
        },
        {"role": "user", "content": request},
    ]

    iteration = 0
    while iteration < MAX_ITERATIONS:
        iteration += 1
        bailout = iteration == MAX_ITERATIONS - 1
        if bailout:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "SYSTEM ALERT: Maximum tool iterations reached. You must immediately output "
                        "your final JSON based on the context you have gathered so far. Do not call "
                        "any more tools."
                    ),
                }
            )
        request_kwargs = {
            "model": MODEL_NAME,
            "messages": messages,
        }
        if not bailout:
            request_kwargs.update({"tools": TOOL_SCHEMAS, "tool_choice": "auto"})
        else:
            request_kwargs["tool_choice"] = "none"
        response = client.chat.completions.create(**request_kwargs)
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
