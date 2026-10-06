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
    assert check_budget("Unknown", 1) == {"error": "Not found", "department": "Unknown"}


def test_catalog_search_matches_category_and_need() -> None:
    results = search_software_catalog("campaign task tracker", "Project Management")
    assert len(results) <= 3
    assert any(row["name"] == "TaskFlow" for row in results)
    assert set(results[0]) == {"name", "category"}


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
    assert get_vendor_security_status("No Such Vendor")["error"] == "Not found"


def test_policy_rules_apply_cfo_and_sensitive_data_requirements() -> None:
    result = evaluate_policy_rules(30000, "high", "customer_pii")
    assert result["requires_cfo_approval"] is True
    assert {"CFO", "Security", "Privacy"} <= set(result["approvals_required"])
    assert "high_vendor_risk" in result["risk_flags"]


@pytest.mark.parametrize("amount", [-1, float("nan")])
def test_tools_reject_invalid_amounts(amount: float) -> None:
    with pytest.raises(ValueError):
        check_budget("Marketing", amount)


def test_procurement_output_uses_unified_literal_contract() -> None:
    output = ProcurementOutput(
        recommendation="REQUEST_INFO",
        evidence=["Annual cost is missing"],
        missing_information=["annual cost"],
        next_step="Ask the requester for the annual cost.",
    )
    assert output.recommendation == "REQUEST_INFO"
