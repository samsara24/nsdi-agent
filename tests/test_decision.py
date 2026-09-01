from dataclasses import replace

import pytest

from rca_framework.branches.base import BranchOutcome
from rca_framework.decision import (
    DEFAULT_DECISION_POLICY,
    DecisionPolicy,
    LLMCalibration,
    apply_llm_calibration,
    decide,
    fit_decision_policy,
)
from rca_framework.llm import DiagnosisResponse, ReasoningTrace


def outcome(**overrides):
    base = BranchOutcome(
        case_id="case_test",
        branch="N5a",
        verdict="L1",
        confidence=0.8,
        confidence_lower_bound=0.55,
        calibration_group="N5a_pure",
        calibration_support=12,
    )
    return replace(base, **overrides)


def trace(case_id, verdict, confidence):
    return ReasoningTrace(
        case_id=case_id,
        accepted=DiagnosisResponse(verdict=verdict, confidence=confidence),
        backend_name="scripted",
    )


def test_final_decision_requires_wilson_lower_bound_and_support():
    policy = DecisionPolicy(final_lower_bound=0.5, minimum_support=10)
    accepted = decide(outcome(), policy)
    assert accepted.action == "final"
    assert accepted.verdict == "L1"

    low_support = decide(outcome(calibration_support=2), policy)
    assert low_support.action == "human_review"
    assert low_support.verdict is None
    assert low_support.proposed_verdict == "L1"


def test_formal_default_candidate_order_is_branch_only():
    assert DEFAULT_DECISION_POLICY.candidate_order == ("branch",)
    rows = [([ ], "L1")]
    policy, _ = fit_decision_policy(rows, target_selective_risk=0.3, minimum_support=1)
    assert policy.candidate_order == ("branch",)


def test_low_confidence_with_missing_evidence_requests_collection():
    decision = decide(
        outcome(
            branch="N5b",
            confidence_lower_bound=0.2,
            missing_evidence=("status:L1:RxLOS",),
        )
    )
    assert decision.action == "request_evidence"
    assert decision.requested_evidence == ("status:L1:RxLOS",)


def test_uncalibrated_llm_confidence_is_not_treated_as_reliability():
    raw = outcome(
        branch="N5c",
        confidence=0.99,
        confidence_lower_bound=0.0,
        calibration_group="llm_raw:N5c",
        calibration_support=0,
    )
    calibrated = apply_llm_calibration(raw, trace(raw.case_id, "L1", 0.99), None)
    assert calibrated.confidence == pytest.approx(0.99)
    assert calibrated.confidence_lower_bound == 0.0
    assert calibrated.calibration_support == 0
    assert decide(calibrated).action == "final"


def test_fatal_llm_fallback_cannot_regain_final_status_from_scalar_confidence():
    invalid = outcome(
        branch="N5b",
        confidence=0.99,
        confidence_lower_bound=0.6,
        calibration_group="llm:N5b:[0.9,1.0]",
        calibration_support=20,
        confidence_breakdown={"physical_compliance": 0.0, "evidence_completeness": 0.8},
        fallback_source="last_parsed_after_fatal",
        compliance_penalties=({"kind": "fabricated_evidence", "physical_compliance_cap": 0.0},),
    )
    decision = decide(invalid)
    assert decision.action == "human_review"
    assert decision.verdict is None
    assert decision.proposed_verdict == "L1"


def test_llm_verdict_step_mismatch_requires_review_even_after_reconciliation():
    conflicted = outcome(
        branch="N5c",
        confidence=0.9,
        confidence_lower_bound=0.6,
        calibration_group="llm:N5c:[0.9,1.0]",
        calibration_support=20,
        confidence_breakdown={"physical_compliance": 0.9},
        compliance_penalties=({"kind": "verdict_step_mismatch", "physical_compliance_cap": 1.0},),
    )
    assert decide(conflicted).action == "human_review"


def test_llm_calibration_uses_independent_correctness_frequency():
    outcomes = [
        outcome(case_id=f"case_{index}", branch="N5c", verdict="L1")
        for index in range(20)
    ]
    traces = [trace(item.case_id, "L1", 0.8) for item in outcomes]
    labels = ["L1"] * 18 + ["L2"] * 2
    calibration = LLMCalibration.fit(outcomes, traces, labels, source="train-loo:test")

    calibrated = apply_llm_calibration(outcomes[0], traces[0], calibration)
    assert calibrated.confidence == pytest.approx(0.8)
    assert calibrated.confidence_lower_bound > 0.5
    assert calibrated.calibration_support == 20
    assert calibrated.calibration_group.startswith("llm:N5c:")
    assert decide(calibrated).action == "final"
