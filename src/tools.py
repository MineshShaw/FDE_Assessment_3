from __future__ import annotations

import logging
from numbers import Real

import pandas as pd
import requests

from src.data_loader import load_all_data
from src.vendor_client import get_vendor_risk


LOGGER = logging.getLogger(__name__)


def _data() -> dict[str, pd.DataFrame]:
    return load_all_data()


def _validate_amount(amount: float) -> float:
    if isinstance(amount, bool) or not isinstance(amount, Real) or pd.isna(amount):
        raise ValueError("amount must be a finite number")
    amount = float(amount)
    if amount < 0:
        raise ValueError("amount must be non-negative")
    return amount


def check_budget(department_id: str, amount: float) -> dict:
    """Return a compact budget result or a safe error envelope."""
    try:
        if not department_id or not isinstance(department_id, str):
            raise ValueError("department_id must be a non-empty string")
        amount = _validate_amount(amount)
        budgets = _data()["budgets"]
        match = budgets[budgets["department"].astype(str).str.casefold() == department_id.casefold()]
        if match.empty:
            return {"status": "not found", "department": department_id}
        row = match.iloc[0]
        remaining = float(row["available_usd"]) - amount
        return {
            "department": str(row["department"]),
            "available_funds": float(row["available_usd"]),
            "remaining_funds": remaining,
            "within_budget": remaining >= 0,
        }
    except Exception as exc:
        return {"error": f"budget check failed: {exc}"}


def search_software_catalog(need_description: str, category: str) -> list[dict] | dict:
    """Return at most two concise catalog matches or a safe error envelope."""
    try:
        if not isinstance(need_description, str) or not isinstance(category, str):
            raise ValueError("need_description and category must be strings")
        catalog = _data()["software_catalog"]
        category_query = category.strip().casefold()
        need_terms = [term for term in need_description.casefold().split() if len(term) > 2]
        category_match = (
            catalog["category"].astype(str).str.casefold().eq(category_query)
            if category_query
            else False
        )
        searchable = catalog[["product_name", "category", "vendor_name", "status", "notes"]].fillna("").astype(str)
        text_match = (
            searchable.apply(
                lambda column: column.str.casefold().apply(
                    lambda value: any(term in value for term in need_terms)
                )
            ).any(axis=1)
            if need_terms
            else False
        )
        matches = catalog[category_match | text_match].head(2)
        return [{"name": str(row["product_name"]), "desc": str(row["notes"])} for _, row in matches.iterrows()]
    except Exception as exc:
        return {"error": f"catalog search failed: {exc}"}


def get_vendor_security_status(vendor_name: str) -> dict:
    """Return compact vendor security data or a safe error envelope."""
    try:
        if not isinstance(vendor_name, str) or not vendor_name.strip():
            raise ValueError("vendor_name must be a non-empty string")
        name = vendor_name.strip()
        data = _data()
        vendors = data["vendors"]
        registry = vendors[vendors["vendor_name"].astype(str).str.casefold() == name.casefold()]
        risk = data["vendor_risk"]
        risk_match = risk[risk["vendor_name"].astype(str).str.casefold() == name.casefold()]
        api_risk: dict | None = None
        try:
            api_risk = get_vendor_risk(name, timeout_seconds=0.5)
        except requests.RequestException as exc:
            LOGGER.warning("Vendor risk API unavailable for %s, using local snapshot fallback: %s", name, exc)

        if registry.empty and risk_match.empty and api_risk is None:
            return {"status": "not found", "vendor_name": name}

        result: dict = {"vendor_name": name}
        if not registry.empty:
            row = registry.iloc[0]
            result.update(
                {
                    "security_status": str(row["security_status"]),
                    "security_review_date": (
                        None if pd.isna(row["security_review_date"]) else str(row["security_review_date"])
                    ),
                }
            )
        if api_risk is not None:
            result.update(
                {
                    "risk_level": api_risk.get("risk_level", "unknown"),
                    "security_review_status": api_risk.get("security_review_status", "unknown"),
                    "last_review_date": api_risk.get("last_review_date"),
                    "risk_service_available": True,
                }
            )
        elif not risk_match.empty:
            row = risk_match.iloc[0]
            result.update(
                {
                    "risk_level": row.get("risk_level", "unknown"),
                    "security_review_status": row.get("security_review_status", "unknown"),
                    "last_review_date": row.get("last_review_date"),
                    "risk_service_available": not bool(row.get("force_error", False)),
                }
            )
        else:
            result["risk_service_available"] = False
        return result
    except Exception as exc:
        return {"error": f"vendor security lookup failed: {exc}"}


def evaluate_policy_rules(amount: float, vendor_risk: str, data_classification: str) -> dict:
    """Return deterministic approvals and risk flags or a safe error envelope."""
    try:
        amount = _validate_amount(amount)
        if not isinstance(vendor_risk, str) or not vendor_risk.strip():
            raise ValueError("vendor_risk must be a non-empty string")
        if not isinstance(data_classification, str) or not data_classification.strip():
            raise ValueError("data_classification must be a non-empty string")

        approvals: list[str] = []
        risk_flags: list[str] = []
        if amount > 25000:
            approvals.extend(["Department Head", "Finance", "CFO", "Procurement"])
        elif amount > 10000:
            approvals.extend(["Department Head", "Finance", "Procurement"])
        elif amount > 1000:
            approvals.extend(["Department Head", "Procurement"])
        else:
            approvals.append("Manager")

        risk = vendor_risk.casefold()
        classification = data_classification.casefold()
        if risk in {"high", "critical"}:
            risk_flags.append("high_vendor_risk")
            approvals.append("Security")
        if risk in {"unknown", "unavailable", "not_completed"}:
            risk_flags.append("vendor_risk_unavailable")
            approvals.append("Security")
        if any(term in classification for term in ("pii", "personal", "confidential", "source_code", "production")):
            risk_flags.append("sensitive_data")
            approvals.append("Security")
        if "pii" in classification or "personal" in classification:
            approvals.append("Privacy")

        approvals = list(dict.fromkeys(approvals))
        risk_flags = list(dict.fromkeys(risk_flags))
        return {
            "amount": amount,
            "approvals_required": approvals,
            "risk_flags": risk_flags,
            "requires_cfo_approval": "CFO" in approvals,
            "requires_human_review": bool(risk_flags or approvals),
        }
    except Exception as exc:
        return {"error": f"policy evaluation failed: {exc}"}
