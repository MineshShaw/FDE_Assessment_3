from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import pandas as pd
import streamlit as st

from src.agents.single_agent import run_single_agent
from src.agents.two_agent import run_two_agent
from src.data_access import get_request, load_budgets, load_employees, load_requests
from src.schemas import ProcurementOutput


ROOT = Path(__file__).resolve().parents[1]
@st.cache_data
def load_cases() -> list[dict]:
    return load_requests()


def render_list(title: str, values: list[str], empty_label: str = "None reported") -> None:
    st.subheader(title)
    if values:
        for value in values:
            st.markdown(f"- {value}")
    else:
        st.caption(empty_label)


def run_architecture(
    request: str, architecture: str, request_data: dict | None = None
) -> ProcurementOutput:
    if architecture == "Single Agent":
        return run_single_agent(request, request_data=request_data)
    return run_two_agent(request, request_data=request_data)


st.set_page_config(page_title="Procurement Copilot", page_icon="🧾", layout="wide")
st.title("Procurement Copilot")
st.caption("Evidence-based recommendations with human approval preserved.")

cases = load_cases()
case_labels = ["Custom request"] + [case["request_id"] for case in cases]
selected_label = st.sidebar.selectbox("Request source", case_labels)

if selected_label == "Custom request":
    selected_case = None
    default_request = ""
else:
    selected_case = get_request(selected_label)
    default_request = selected_case["business_justification"]

request = st.text_area(
    "Purchase request",
    value=default_request,
    height=140,
    placeholder="Describe the product, vendor, department, cost, users, and data access.",
)
with st.expander("Request details", expanded=True):
    if selected_case is not None:
        employee = load_employees()
        employee_row = employee[employee["employee_id"] == selected_case["requester_id"]]
        department = employee_row.iloc[0]["department"] if not employee_row.empty else "Unknown"
        manager_id = employee_row.iloc[0]["manager_id"] if not employee_row.empty else None
        manager_row = employee[employee["employee_id"] == manager_id]
        manager = manager_row.iloc[0]["name"] if not manager_row.empty else (manager_id or "Unknown")
        budget_rows = load_budgets()
        budget_row = budget_rows[budget_rows["department"] == department]
        available_budget = (
            float(budget_row.iloc[0]["available_usd"]) if not budget_row.empty else None
        )
        details = {
            "Requester": selected_case["requester_id"],
            "Department": department,
            "Manager": manager,
            "Available budget": available_budget,
            "Vendor": selected_case["vendor_name"],
            "Amount": selected_case["annual_cost_usd"],
            "Users": selected_case["user_count"],
            "Data access": selected_case["data_access_level"],
            "Integrations": ", ".join(selected_case["requested_integrations"]) or "None",
        }
        detail_cols = st.columns(3)
        for index, (label, value) in enumerate(details.items()):
            detail_cols[index % 3].metric(label, "Missing" if value is None else str(value))
        request_data = selected_case
    else:
        st.caption("Enter a custom request below, or choose a REQ-ID for structured policy facts.")
        form_cols = st.columns(4)
        requester = form_cols[0].text_input("Requester ID")
        department = form_cols[1].text_input("Department")
        amount = form_cols[2].number_input("Annual cost", min_value=0.0, value=0.0)
        vendor = form_cols[3].text_input("Vendor / product")
        request_data = None
        if requester and amount:
            request_data = {
                "requester_id": requester,
                "vendor_name": vendor,
                "annual_cost_usd": amount,
                "data_access_level": "internal",
                "business_justification": request,
            }
architecture = st.sidebar.radio("Architecture", ["Single Agent", "Staged Two-Agent"])

if st.sidebar.button("Clear result"):
    st.session_state.pop("procurement_result", None)
    st.session_state.pop("review_action", None)

if st.button("Run procurement review", type="primary", disabled=not request.strip()):
    started = __import__("time").perf_counter()
    with st.spinner(f"Running {architecture.lower()} review..."):
        try:
            st.session_state["procurement_result"] = run_architecture(
                request.strip(), architecture, request_data=request_data
            )
            st.session_state["review_metrics"] = {
                "latency_ms": round((__import__("time").perf_counter() - started) * 1000, 1),
                "tools": ["check_budget", "search_software_catalog", "get_vendor_security_status"]
                if request_data is not None else [],
            }
            st.session_state.pop("review_action", None)
            st.session_state.setdefault("decision_log", [])
        except Exception as exc:
            st.error(f"Review failed: {type(exc).__name__}: {exc}")

result = st.session_state.get("procurement_result")
if result is None:
    st.info("Choose a preloaded case or enter a request, then run the review.")
else:
    if isinstance(result, dict):
        result = ProcurementOutput.model_validate(result)

    st.divider()
    st.subheader("Recommendation")
    recommendation_colors = {
        "APPROVE": "success",
        "REJECT": "error",
        "ESCALATE_TO_HUMAN": "warning",
        "REQUEST_INFO": "info",
    }
    getattr(st, recommendation_colors[result.recommendation])(result.recommendation)
    st.write(result.next_step)

    left, right = st.columns(2)
    with left:
        render_list("Approvals required", result.approvals_required)
        render_list("Missing information", result.missing_information)
    with right:
        render_list("Risk flags", result.risk_flags)
        st.subheader("Evidence")
        if result.evidence:
            evidence_rows = [
                {
                    "Source": item.split(":", 1)[0] if ":" in item else "agent",
                    "Finding": item.split(":", 1)[1].strip() if ":" in item else item,
                }
                for item in result.evidence
            ]
            st.dataframe(pd.DataFrame(evidence_rows), hide_index=True, use_container_width=True)
        else:
            st.caption("No evidence returned.")
        metrics = st.session_state.get("review_metrics")
        if metrics:
            st.caption(f"Latency: {metrics['latency_ms']} ms")
            st.caption("Tools used: " + (", ".join(metrics["tools"]) or "none"))

    with st.expander("Raw structured output"):
        st.json(result.model_dump())

    st.divider()
    st.subheader("Human review")
    st.caption("These controls record a reviewer decision in this session; they do not purchase or approve spend.")
    approve, reject, override = st.columns(3)
    with approve:
        approve_disabled = result.recommendation == "REQUEST_INFO"
        if st.button("Approve", disabled=approve_disabled, use_container_width=True):
            action = "Approved by human reviewer"
            st.session_state["review_action"] = action
            st.session_state.setdefault("decision_log", []).append(action)
    with reject:
        reject_reason = st.text_input("Reject reason", key="reject_reason")
        if st.button("Reject", disabled=not reject_reason.strip(), use_container_width=True):
            action = f"Rejected by human reviewer: {reject_reason.strip()}"
            st.session_state["review_action"] = action
            st.session_state.setdefault("decision_log", []).append(action)
    with override:
        override_reason = st.text_input("Override reason", key="override_reason")
        if st.button("Override", disabled=not override_reason.strip(), use_container_width=True):
            action = f"Overridden by human reviewer: {override_reason.strip()}"
            st.session_state["review_action"] = action
            st.session_state.setdefault("decision_log", []).append(action)

    review_action = st.session_state.get("review_action")
    if review_action:
        st.warning(review_action)
    if st.session_state.get("decision_log"):
        st.subheader("Session decision log")
        for action in st.session_state["decision_log"]:
            st.write(action)
