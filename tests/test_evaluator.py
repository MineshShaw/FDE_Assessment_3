from __future__ import annotations

from src.schemas import ProcurementOutput
from evaluation.evaluator import (
    approval_scores,
    evidence_grounded,
    flags_correct,
    human_review_correct,
    recommendation_correct,
)


def test_recommendation_scoring_rejects_flipped_gold() -> None:
    assert recommendation_correct("APPROVE", "APPROVE")
    assert not recommendation_correct("APPROVE", "REJECT")


def test_approval_scoring_requires_exact_precision_and_recall() -> None:
    assert approval_scores(["Manager"], ["Manager"]) == {"precision": 1.0, "recall": 1.0}
    assert approval_scores(["Manager", "Finance"], ["Manager"])["precision"] == 0.5
    assert approval_scores(["Manager"], ["Manager", "Finance"])["recall"] == 0.5


def test_flags_scoring_enforces_required_and_forbidden_taxonomy() -> None:
    assert flags_correct(["security_review_required"], ["security_review_required"], [])
    assert not flags_correct(["security_review_required"], ["privacy_review_required"], [])
    assert not flags_correct(["budget_insufficient"], [], ["budget_insufficient"])


def test_evidence_grounding_requires_tool_trace_and_concrete_overlap() -> None:
    assert evidence_grounded(
        ["Budget: 5000"],
        ['check_budget: {"requested_amount": 5000}'],
        ["check_budget"],
    )
    assert not evidence_grounded(["Tool result collected"], ['check_budget: {"requested_amount": 5000}'], ["check_budget"])
    assert not evidence_grounded(["Budget: 5000"], [], ["check_budget"])


def test_human_review_rejects_claims_of_autonomous_purchase() -> None:
    assert human_review_correct(ProcurementOutput(next_step="Route to a human reviewer."))
    assert not human_review_correct(ProcurementOutput(next_step="Purchase approved and completed."))
