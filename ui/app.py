from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import pandas as pd
import streamlit as st

from src.agents.single_agent import run_single_agent
from src.agents.two_agent import run_two_agent
from src.schemas import ProcurementOutput


ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "evaluation" / "test_cases.json"


@st.cache_data
def load_cases() -> list[dict]:
    return json.loads(CASES_PATH.read_text(encoding="utf-8"))


def render_list(title: str, values: list[str], empty_label: str = "None reported") -> None:
    st.subheader(title)
    if values:
        for value in values:
            st.markdown(f"- {value}")
    else:
        st.caption(empty_label)


def run_architecture(request: str, architecture: str) -> ProcurementOutput:
    if architecture == "Single Agent":
        return run_single_agent(request)
    return run_two_agent(request)


st.set_page_config(page_title="Procurement Copilot", page_icon="🧾", layout="wide")
st.title("Procurement Copilot")
st.caption("Evidence-based recommendations with human approval preserved.")

cases = load_cases()
case_labels = ["Custom request"] + [f"{case['id']} — {case['request'][:80]}" for case in cases]
selected_label = st.sidebar.selectbox("Request source", case_labels)

if selected_label == "Custom request":
    default_request = ""
else:
    selected_case = cases[case_labels.index(selected_label) - 1]
    default_request = selected_case["request"]

request = st.text_area(
    "Purchase request",
    value=default_request,
    height=140,
    placeholder="Describe the product, vendor, department, cost, users, and data access.",
)
with st.expander("Request details", expanded=True):
    detail_cols = st.columns(4)
    with detail_cols[0]:
        requester = st.text_input("Requester", placeholder="Employee name or ID")
    with detail_cols[1]:
        department = st.text_input("Department", placeholder="Marketing")
    with detail_cols[2]:
        budget = st.text_input("Annual budget / cost", placeholder="$5,000")
    with detail_cols[3]:
        vendor = st.text_input("Vendor / product", placeholder="Vendor or product name")
architecture = st.sidebar.radio("Architecture", ["Single Agent", "Staged Two-Agent"])

if st.sidebar.button("Clear result"):
    st.session_state.pop("procurement_result", None)
    st.session_state.pop("review_action", None)

if st.button("Run procurement review", type="primary", disabled=not request.strip()):
    with st.spinner(f"Running {architecture.lower()} review..."):
        try:
            st.session_state["procurement_result"] = run_architecture(request.strip(), architecture)
            st.session_state.pop("review_action", None)
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

    with st.expander("Raw structured output"):
        st.json(result.model_dump())

    st.divider()
    st.subheader("Human review")
    st.caption("These controls record a reviewer decision in this session; they do not purchase or approve spend.")
    approve, reject, override = st.columns(3)
    with approve:
        approve_disabled = result.recommendation == "REQUEST_INFO"
        if st.button("Approve", disabled=approve_disabled, use_container_width=True):
            st.session_state["review_action"] = "Approved by human reviewer"
    with reject:
        if st.button("Reject", use_container_width=True):
            st.session_state["review_action"] = "Rejected by human reviewer"
    with override:
        override_reason = st.text_input("Override reason", key="override_reason")
        if st.button("Override", disabled=not override_reason.strip(), use_container_width=True):
            st.session_state["review_action"] = f"Overridden by human reviewer: {override_reason.strip()}"

    review_action = st.session_state.get("review_action")
    if review_action:
        st.warning(review_action)
