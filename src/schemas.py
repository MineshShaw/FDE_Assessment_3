from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ProcurementOutput(BaseModel):
    recommendation: Literal["APPROVE", "REJECT", "ESCALATE_TO_HUMAN", "REQUEST_INFO"] = Field(
        default="ESCALATE_TO_HUMAN"
    )
    evidence: list[str] = Field(default_factory=list)
    approvals_required: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    next_step: str = Field(default="Manual review required.")


class StructuredEvidencePack(BaseModel):
    budget_status: dict = Field(default_factory=dict)
    tool_overlap: list[dict] = Field(default_factory=list)
    vendor_risk: dict = Field(default_factory=dict)

    @field_validator("tool_overlap", mode="before")
    @classmethod
    def coerce_to_list(cls, value: object) -> object:
        if isinstance(value, dict):
            return [value]
        return value if value is not None else []
