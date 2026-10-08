from __future__ import annotations

import os

from src.contracts import ProcurementDecision
from src.schemas import ProcurementOutput


BLOCKING_FLAGS = {
    "security_review_required",
    "privacy_review_required",
    "legal_review_required",
    "budget_insufficient",
    "vendor_review_expired",
    "conflicting_vendor_evidence",
    "vendor_risk_unavailable",
    "prompt_injection_detected",
    "missing_information",
    "policy_unverified",
}


def apply_policy_floor(
    output: ProcurementOutput,
    floor: ProcurementDecision,
) -> ProcurementOutput:
    """Merge deterministic policy facts and prevent unsafe model recommendations."""
    approvals = list(dict.fromkeys(floor.required_approvals + output.approvals_required))
    risk_flags = list(dict.fromkeys(floor.risk_flags + output.risk_flags))
    missing = list(dict.fromkeys(floor.missing_information + output.missing_information))
    recommendation = output.recommendation
    evidence = list(output.evidence)

    if missing:
        forced = "REQUEST_INFO"
    elif recommendation == "APPROVE" and BLOCKING_FLAGS.intersection(risk_flags):
        forced = "ESCALATE_TO_HUMAN"
    else:
        forced = recommendation

    if forced != recommendation:
        reasons = ", ".join(sorted(BLOCKING_FLAGS.intersection(risk_flags)))
        evidence.append(
            f"Guardrail: recommendation changed from {recommendation} to {forced} because {reasons}."
        )
    evidence.extend(
        f"[{item.source}] {item.finding}"
        for item in floor.evidence
    )
    return output.model_copy(
        update={
            "recommendation": forced,
            "approvals_required": approvals,
            "risk_flags": risk_flags,
            "missing_information": missing,
            "evidence": list(dict.fromkeys(evidence)),
        }
    )


def guardrails_enabled() -> bool:
    return os.getenv("DISABLE_GUARDRAIL", "").strip() != "1"
