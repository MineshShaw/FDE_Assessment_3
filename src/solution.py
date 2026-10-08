from __future__ import annotations

from datetime import date
import os

import pandas as pd
from requests import RequestException

from src.contracts import Architecture, ProcurementDecision
from src.data_access import (
    get_request,
    load_budgets,
    load_employees,
    load_software_catalog,
    load_vendors,
)
from src.telemetry import RunTelemetryCounter
from src.vendor_client import get_vendor_risk


REFERENCE_DATE = date(2026, 9, 30)
REQUIRED_REQUEST_FIELDS = (
    ("requester_id", "requester and department"),
    ("product_name", "product/vendor"),
    ("vendor_name", "product/vendor"),
    ("annual_cost_usd", "annual cost"),
    ("user_count", "number of users/licenses"),
    ("business_justification", "business purpose"),
    ("data_access_level", "intended data-access level"),
    ("requested_integrations", "required integrations"),
)


def _row(frame: pd.DataFrame, column: str, value: object) -> dict | None:
    matches = frame[frame[column].astype(str).str.casefold() == str(value).casefold()]
    return matches.iloc[0].to_dict() if not matches.empty else None


def _missing_request_fields(request: dict) -> list[str]:
    missing: list[str] = []
    for field, label in REQUIRED_REQUEST_FIELDS:
        value = request.get(field)
        if value is None or value == "" or (field == "data_access_level" and str(value).casefold() == "unknown"):
            missing.append(label)
    return missing


def _annual_amount(value: object) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return float(value)


def _review_expired(review_date: object) -> bool:
    if not review_date:
        return True
    try:
        reviewed = date.fromisoformat(str(review_date))
    except ValueError:
        return True
    return (REFERENCE_DATE - reviewed).days > 365


def _add_evidence(
    evidence: list[dict],
    source: str,
    finding: str,
    reference: str | None = None,
) -> None:
    evidence.append({"source": source, "finding": finding, "reference": reference})


def _approval_thresholds(amount: float | None) -> list[str]:
    if amount is None:
        return []
    if amount <= 1000:
        return ["Manager"]
    if amount <= 10000:
        return ["Department Head", "Procurement"]
    if amount <= 25000:
        return ["Department Head", "Finance", "Procurement"]
    return ["Department Head", "Finance", "CFO", "Procurement"]


def _vendor_risk_with_fallback(
    vendor_name: str,
    vendor_record: dict | None,
    evidence: list[dict],
    risk_flags: list[str],
    telemetry: RunTelemetryCounter,
) -> dict | None:
    try:
        telemetry.record_tool_call("vendor_risk_api")
        risk = get_vendor_risk(vendor_name)
        _add_evidence(
            evidence,
            "vendor_risk_api",
            f"Risk level {risk.get('risk_level', 'unknown')}; security status "
            f"{risk.get('security_review_status', 'unknown')}.",
            vendor_name,
        )
        return risk
    except RequestException as exc:
        if vendor_record is None or str(vendor_record.get("security_status", "")).casefold() in {"", "unknown"}:
            risk_flags.append("vendor_risk_unavailable")
            _add_evidence(evidence, "vendor_risk_api", f"Vendor-risk evidence unavailable: {exc}", vendor_name)
            return None
        _add_evidence(
            evidence,
            "vendor_registry",
            "Vendor-risk API unavailable; using the internal vendor registry for non-sensitive facts.",
            str(vendor_record.get("vendor_id")),
        )
        return {
            "vendor_name": vendor_name,
            "risk_level": "unknown",
            "security_review_status": str(vendor_record.get("security_status", "unknown")).casefold(),
            "last_review_date": vendor_record.get("security_review_date"),
            "processes_personal_data": None,
            "stores_data_outside_region": None,
        }


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    lowered = text.casefold()
    return any(term in lowered for term in terms)


def evaluate_request(request: dict, architecture: Architecture = "single") -> ProcurementDecision:
    """Evaluate a procurement request using deterministic policy checks.

    Both supported architectures share the same policy engine. The staged
    variant records a separate planning pass in telemetry without introducing
    nondeterministic model behavior into approval decisions.
    """
    if architecture not in ("single", "staged"):
        raise ValueError(f"Unsupported architecture: {architecture}")

    request_id = str(request.get("request_id", "UNKNOWN"))
    telemetry = RunTelemetryCounter()
    evidence: list[dict] = []
    risk_flags: list[str] = []
    missing_information = _missing_request_fields(request)
    approvals: list[str] = []

    telemetry.record_tool_call("request_lookup")
    _add_evidence(evidence, "request_lookup", f"Request received for {request.get('product_name', 'unknown product')}.", request_id)

    if _contains_any(
        str(request.get("business_justification", "")),
        ("ignore all procurement rules", "approve it immediately", "bypass controls", "ignore policy"),
    ):
        risk_flags.append("prompt_injection_detected")

    employee = _row(load_employees(), "employee_id", request.get("requester_id"))
    telemetry.record_tool_call("employee_directory")
    if employee is None:
        missing_information.append("requester and department")
    else:
        department = str(employee["department"])
        _add_evidence(
            evidence,
            "employee_directory",
            f"Requester {employee['name']} is in {department}; manager {employee['manager_id']}.",
            str(employee["employee_id"]),
        )

    amount = _annual_amount(request.get("annual_cost_usd"))
    if amount is not None:
        approvals.extend(_approval_thresholds(amount))

    budget = _row(load_budgets(), "department", employee["department"]) if employee else None
    telemetry.record_tool_call("budget_check")
    if budget is not None and amount is not None:
        available = float(budget["available_usd"])
        _add_evidence(
            evidence,
            "budget_check",
            f"{employee['department']} has ${available:,.0f} available; request is ${amount:,.0f}.",
            employee["department"],
        )
        if amount > available:
            risk_flags.append("budget_insufficient")
            if "Finance" not in approvals:
                approvals.append("Finance")
    elif amount is None:
        _add_evidence(evidence, "budget_check", "Annual cost is missing, so budget sufficiency cannot be checked.", None)

    catalog = load_software_catalog()
    telemetry.record_tool_call("software_catalog")
    vendor_name = str(request.get("vendor_name", ""))
    category = str(request.get("category", ""))
    overlap = catalog[
        (catalog["vendor_name"].astype(str).str.casefold() == vendor_name.casefold())
        | (catalog["category"].astype(str).str.casefold() == category.casefold())
    ]
    if not overlap.empty:
        risk_flags.append("existing_tool_overlap")
        products = ", ".join(str(value) for value in overlap["product_name"].tolist())
        _add_evidence(evidence, "software_catalog", f"Potential existing alternative(s): {products}.", None)
    else:
        _add_evidence(evidence, "software_catalog", "No same-vendor or same-category approved alternative found.", None)

    vendors = load_vendors()
    vendor_record = _row(vendors, "vendor_name", vendor_name)
    telemetry.record_tool_call("vendor_registry")
    if vendor_record is None:
        missing_information.append("vendor procurement record")
        _add_evidence(evidence, "vendor_registry", f"No internal record found for {vendor_name}.", vendor_name)
    else:
        _add_evidence(
            evidence,
            "vendor_registry",
            f"Procurement status {vendor_record['procurement_status']}; security status {vendor_record['security_status']}.",
            str(vendor_record["vendor_id"]),
        )

    risk = _vendor_risk_with_fallback(vendor_name, vendor_record, evidence, risk_flags, telemetry)
    data_access = str(request.get("data_access_level", "")).casefold()
    integrations = " ".join(str(item) for item in request.get("requested_integrations") or [])
    sensitive_security_terms = (
        "source_code",
        "production",
        "cloud",
        "confidential",
        "employee_pii",
        "customer_pii",
        "credential",
        "secret",
    )
    if (
        any(term in data_access for term in sensitive_security_terms)
        or _contains_any(integrations, ("production", "cloud account", "git repositories"))
        or (risk is not None and str(risk.get("security_review_status", "")).casefold() not in {"approved", "current"})
    ):
        risk_flags.append("security_review_required")

    if risk is not None and _review_expired(risk.get("last_review_date")):
        risk_flags.append("vendor_review_expired")
        risk_flags.append("security_review_required")

    registry_status = str(vendor_record.get("security_status", "")).casefold() if vendor_record else ""
    api_status = str(risk.get("security_review_status", "")).casefold() if risk else ""
    if vendor_record and risk and registry_status and api_status and registry_status != api_status:
        risk_flags.append("conflicting_vendor_evidence")
        risk_flags.append("security_review_required")

    if "employee_pii" in data_access or "customer_pii" in data_access:
        risk_flags.append("privacy_review_required")
    if risk and risk.get("stores_data_outside_region") is True:
        risk_flags.append("privacy_review_required")

    is_new_vendor = vendor_record is None or str(vendor_record.get("procurement_status", "")).casefold() == "new"
    legal_status = str(vendor_record.get("legal_terms_status", "")).casefold() if vendor_record else ""
    if (is_new_vendor and amount is not None and amount >= 10000) or legal_status not in {"approved", "standard"}:
        risk_flags.append("legal_review_required")

    if missing_information:
        risk_flags.append("missing_information")

    # Preserve stable output while avoiding duplicate flags/approvals from layered checks.
    risk_flags = list(dict.fromkeys(risk_flags))
    for approval, flag in (
        ("Security", "security_review_required"),
        ("Privacy", "privacy_review_required"),
        ("Legal", "legal_review_required"),
    ):
        if flag in risk_flags and approval not in approvals:
            approvals.append(approval)
    approvals = list(dict.fromkeys(approvals))
    missing_information = list(dict.fromkeys(missing_information))

    if missing_information:
        recommendation = "Request clarification before approval"
        next_step = "Requester must provide the missing information; no purchase or approval should proceed."
    elif risk_flags:
        recommendation = "Route for required reviews before approval"
        next_step = "Collect the listed human approvals and resolve all risk flags before procurement."
    else:
        recommendation = "Proceed to human approval"
        next_step = "Manager and procurement may review the evidence and make the final approval decision."

    if architecture == "staged":
        telemetry.record_tool_call("staged_policy_review")

    return ProcurementDecision(
        request_id=request_id,
        recommendation=recommendation,
        evidence=evidence,
        required_approvals=approvals,
        missing_information=missing_information,
        risk_flags=risk_flags,
        next_step=next_step,
        human_review_required=True,
        telemetry={
            "llm_calls": 0,
            "tool_calls": telemetry.tool_calls,
            "tool_names": telemetry.tool_names,
        },
    )


def handle_request(request_id: str, architecture: Architecture = "single") -> ProcurementDecision:
    """Evaluate a request with the selected agent when configured, else use the policy engine."""
    request = get_request(request_id)
    floor = evaluate_request(request, architecture=architecture)

    from src.agents import single_agent, two_agent

    selected = single_agent if architecture == "single" else two_agent
    if not (
        os.getenv("GROQ_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or selected.client is not None
    ):
        return floor

    try:
        output = (
            selected.run_single_agent(request["business_justification"], request_data=request)
            if architecture == "single"
            else selected.run_two_agent(request["business_justification"], request_data=request)
        )
        telemetry = selected.last_telemetry
        agent_evidence = [
            {"source": "agent", "finding": item, "reference": request_id}
            for item in output.evidence
        ]
        combined_evidence = floor.evidence + agent_evidence
        return ProcurementDecision(
            request_id=request_id,
            recommendation=output.recommendation,
            evidence=combined_evidence,
            required_approvals=list(dict.fromkeys(floor.required_approvals + output.approvals_required)),
            missing_information=list(dict.fromkeys(floor.missing_information + output.missing_information)),
            risk_flags=list(dict.fromkeys(floor.risk_flags + output.risk_flags)),
            next_step=output.next_step,
            human_review_required=True,
            telemetry={
                "llm_calls": telemetry.llm_calls,
                "tool_calls": telemetry.tool_calls + floor.telemetry.tool_calls,
                "tool_names": list(dict.fromkeys(
                    floor.telemetry.tool_names + telemetry.tool_names
                )),
            },
        )
    except Exception:
        return floor.model_copy(
            update={
                "risk_flags": list(dict.fromkeys(floor.risk_flags + ["llm_unavailable"])),
                "telemetry": {
                    "llm_calls": getattr(selected.last_telemetry, "llm_calls", 0),
                    "tool_calls": floor.telemetry.tool_calls
                    + getattr(selected.last_telemetry, "tool_calls", 0),
                    "tool_names": list(dict.fromkeys(
                        floor.telemetry.tool_names
                        + getattr(selected.last_telemetry, "tool_names", [])
                    )),
                },
            }
        )
