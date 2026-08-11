from dataclasses import replace

import pytest

from rca_framework.branches.base import BranchOutcome
from rca_framework.decision import (
    DecisionPolicy,
    LLMCalibration,
    apply_llm_calibration,
    decide,
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
    assert decide(calibrated).action == "human_review"


def test_llm_calibration_uses_independent_correctness_frequency():
    outcomes = [
        outcome(case_id=f"case_{index}", branch="N5c", verdict="L1")
        for index in range(20)
    ]
    traces = [trace(item.case_id, "L1", 0.8) for item in outcomes]
    labels = ["L1"] * 18 + ["L2"] * 2
    calibration = LLMCalibration.fit(outcomes, traces, labels, source="train-loo:test")

    calibrated = apply_llm_calibration(outcomes[0], traces[0], calibration)
    assert calibrated.confidence == pytest.approx(0.9)
    assert calibrated.confidence_lower_bound > 0.5
    assert calibrated.calibration_support == 20
    assert calibrated.calibration_group.startswith("llm:N5c:")
    assert decide(calibrated).action == "final"

