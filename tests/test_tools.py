from __future__ import annotations

import pytest
import requests

from src.schemas import ProcurementOutput
import src.tools as tools
from src.tools import (
    check_budget,
    evaluate_policy_rules,
    get_vendor_security_status,
    search_software_catalog,
)


def test_check_budget_reports_remaining_funds_and_unknown_department() -> None:
    result = check_budget("Marketing", 5000)
    assert result["remaining_funds"] == 10000
    assert result["within_budget"] is True
    assert check_budget("Unknown", 1) == {"status": "not found", "department": "Unknown"}


def test_catalog_search_matches_category_and_need() -> None:
    results = search_software_catalog("campaign task tracker", "Project Management")
    assert len(results) <= 5
    assert any(row["name"] == "TaskFlow" for row in results)
    assert {"name", "desc", "category", "status", "vendor", "scope", "licensed_seats", "annual_cost_usd"} <= set(results[0])


def test_catalog_search_prioritizes_exact_category_and_relevant_terms() -> None:
    results = search_software_catalog("AI support assistant", "General AI")
    assert results[0]["name"] == "NeuralDesk Business"


def test_catalog_search_uses_category_for_unknown_product_names() -> None:
    results = search_software_catalog("Asana project management", "Project Management")
    assert results[0]["category"] == "Project Management"
    assert results[0]["name"] == "TaskFlow"


def test_vendor_status_reports_expiry_and_conflict(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        tools,
        "get_vendor_risk",
        lambda vendor_name, timeout_seconds: {
            "security_review_status": "expired",
            "last_review_date": "2025-01-01",
            "risk_level": "medium",
        },
    )
    result = get_vendor_security_status("CodeMate")
    assert result["review_expired"] is True
    assert result["registry_api_conflict"] is True
    assert result["evidence_conflict"] is True


def test_vendor_status_json_is_strict_and_computed_fields_are_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tools, "get_vendor_risk", lambda *args, **kwargs: (_ for _ in ()).throw(requests.Timeout("down")))
    result = get_vendor_security_status("NimbusAI")
    import json
    json.dumps(result, allow_nan=False)
    assert result["data_source"] == "local_snapshot_fallback"
    assert result["risk_service_available"] is False
    assert result["review_expired"] is True


def test_signalwatch_and_codemate_conflict_and_expiry(monkeypatch: pytest.MonkeyPatch) -> None:
    def risk(name: str, timeout_seconds: float) -> dict:
        if name == "SignalWatch":
            return {"security_review_status": "expired", "last_review_date": "2025-07-01", "risk_level": "medium"}
        return {"security_review_status": "approved", "last_review_date": "2026-08-20", "risk_level": "medium"}
    monkeypatch.setattr(tools, "get_vendor_risk", risk)
    signal = get_vendor_security_status("SignalWatch")
    code = get_vendor_security_status("CodeMate")
    assert signal["review_expired"] is True and signal["evidence_conflict"] is True
    assert code["review_expired"] is False and code["evidence_conflict"] is False


def test_vendor_security_status_uses_mock_service_and_handles_unknown_vendors(monkeypatch: pytest.MonkeyPatch) -> None:
    def mock_vendor_risk(vendor_name: str, timeout_seconds: float) -> dict:
        if vendor_name == "No Such Vendor":
            raise requests.HTTPError("not found")
        return {
            "vendor_name": vendor_name,
            "security_review_status": "approved",
            "last_review_date": "2026-08-20",
        }

    monkeypatch.setattr(tools, "get_vendor_risk", mock_vendor_risk)
    result = get_vendor_security_status("CodeMate")
    assert result["security_status"] == "Approved"
    assert result["security_review_status"] == "approved"
    assert result["risk_service_available"] is True
    assert get_vendor_security_status("No Such Vendor")["status"] == "not found"


def test_tools_return_error_envelopes_for_invalid_inputs() -> None:
    assert "error" in check_budget("", 1)
    assert "error" in search_software_catalog(None, "Project Management")
    assert "error" in get_vendor_security_status("")
    assert "error" in evaluate_policy_rules(-1, "low", "internal")


def test_policy_rules_apply_cfo_and_sensitive_data_requirements() -> None:
    result = evaluate_policy_rules(30000, "high", "customer_pii")
    assert result["requires_cfo_approval"] is True
    assert {"CFO", "Security", "Privacy"} <= set(result["approvals_required"])
    assert "high_vendor_risk" in result["risk_flags"]


@pytest.mark.parametrize("amount", [-1, float("nan")])
def test_tools_reject_invalid_amounts(amount: float) -> None:
    assert "error" in check_budget("Marketing", amount)


def test_procurement_output_uses_unified_literal_contract() -> None:
    output = ProcurementOutput(
        recommendation="REQUEST_INFO",
        evidence=["Annual cost is missing"],
        missing_information=["annual cost"],
        next_step="Ask the requester for the annual cost.",
    )
    assert output.recommendation == "REQUEST_INFO"
