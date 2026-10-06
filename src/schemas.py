from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ProcurementOutput(BaseModel):
    recommendation: Literal["APPROVE", "REJECT", "ESCALATE_TO_HUMAN", "REQUEST_INFO"]
    evidence: list[str] = Field(default_factory=list)
    approvals_required: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    next_step: str
