from __future__ import annotations

import json

from pydantic import BaseModel, Field

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
from src.tools import evaluate_policy_rules


MAX_ITERATIONS = 5


class StructuredEvidencePack(BaseModel):
    budget_status: dict = Field(default_factory=dict)
    tool_overlap: list[dict] = Field(default_factory=list)
    vendor_risk: dict = Field(default_factory=dict)


def _run_analyst(request: str) -> StructuredEvidencePack:
    messages: list[dict] = [
        {
            "role": "system",
            "content": (
                "You are the Analyst, an analytical agent. Call the provided tools to gather evidence. "
                "Once you have sufficient evidence, DO NOT call any more tools. You must output a raw "
                "JSON object matching StructuredEvidencePack with budget_status, tool_overlap, and vendor_risk. "
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
            tools=TOOL_SCHEMAS[:3],
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
        return parse_model(value(message, "content"), StructuredEvidencePack)
    raise RuntimeError(f"Analyst orchestration exceeded {MAX_ITERATIONS} iterations")


def run_two_agent(request: str) -> ProcurementOutput:
    if client is None:
        raise RuntimeError("OPENAI_API_KEY is required to run the two-agent architecture")
    evidence_pack = _run_analyst(request)
    reviewer_prompt = (
        "You are the Reviewer, an analytical agent. Review the request and evidence pack. Apply the "
        "deterministic policy result supplied below. Once you have sufficient evidence, DO NOT call "
        "any more tools. Output a raw JSON object matching the ProcurementOutput schema. Do not include "
        "markdown formatting, code blocks, or explanatory text outside the JSON.\n\n"
        f"Request:\n{request}\n\nEvidence pack:\n{evidence_pack.model_dump_json()}\n\n"
        "Apply policy using the request's amount, vendor risk, and data classification."
    )

    # The reviewer applies this deterministic check before asking the LLM to phrase the final result.
    policy_result = evaluate_policy_rules(
        amount=_number_from_pack(evidence_pack, "amount"),
        vendor_risk=str(evidence_pack.vendor_risk.get("risk_level", "unknown")),
        data_classification=_classification_from_request(request),
    )
    reviewer_prompt += f"\n\nDeterministic policy result:\n{json.dumps(policy_result)}"
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": reviewer_prompt}],
    )
    return parse_model(value(message_from_response(response), "content"), ProcurementOutput)


def _number_from_pack(pack: StructuredEvidencePack, key: str) -> float:
    value = pack.budget_status.get(key, 0)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Evidence pack is missing numeric {key}") from exc


def _classification_from_request(request: str) -> str:
    lowered = request.casefold()
    for classification in ("customer_pii", "employee_pii", "source_code", "confidential_documents", "production"):
        if classification in lowered:
            return classification
    return "internal"
