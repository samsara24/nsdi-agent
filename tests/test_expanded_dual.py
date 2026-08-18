import json
from pathlib import Path
from types import SimpleNamespace

from rca_framework.expanded_dual import (
    SOP_STEP_IDS,
    DualCandidate,
    DualMatchResult,
    EvidenceView,
    execute_sop,
    route_dual,
    validate_expanded_llm_response,
    build_views,
)
from rca_framework.llm.protocol import DiagnosisResponse, ReasoningStep
from rca_framework.llm.backend import validate_context_window


def _view(*, quality="valid", tokens=("a",), edges=("e",)):
    return EvidenceView("query", tuple(tokens), tuple(edges), (), quality, ())


def _result(candidates, *, ft=0.7, gt=0.7, quality="valid"):
    return DualMatchResult("query", _view(quality=quality), tuple(candidates), ft, gt)


def _candidate(case_id, label, sf, sg, *, compatible=True, conflicts=()):
    return DualCandidate(case_id, label, sf, sg, compatible, tuple(conflicts))


def test_dual_high_pure_bucket_reuses_only_with_three_compatible_cases():
    result = _result(tuple(_candidate(str(i), "L1", 0.9, 0.9) for i in range(3)))
    assert result.strict_reuse
    assert route_dual(result).branch == "N5a"


def test_mixed_single_high_low_and_conflict_never_reuse():
    mixed = _result((_candidate("a", "L1", .9, .9), _candidate("b", "L2", .9, .9),
                     _candidate("c", "L1", .9, .9)))
    assert route_dual(mixed).branch == "N5b"
    single = _result((_candidate("a", "L1", .9, .2),))
    assert route_dual(single).branch == "N5b"
    low = _result((_candidate("a", "L1", .2, .2),))
    assert route_dual(low).branch == "N5c"
    conflict = _result(tuple(_candidate(str(i), "L1", .9, .9, conflicts=(("x", "y"),)) for i in range(3)))
    assert route_dual(conflict).branch == "N5b"


def test_blackout_preempts_history_and_sop():
    candidates = tuple(_candidate(str(i), "L1", 1.0, 1.0) for i in range(3))
    assert route_dual(_result(candidates, quality="optical_blackout")).branch == "N6"
    case = {
        "case_id": "blackout", "label": "L1",
        "txpower": {"L1": [-40.0] * 4, "L2": [-40.0] * 4},
        "rxpower": {"L1": [-40.0] * 4, "L2": [-40.0] * 4},
        "media_snr": {"L1": [0.0] * 4, "L2": [0.0] * 4},
        "host_snr": {"L1": [0.0] * 4, "L2": [0.0] * 4},
        "serdes_snr": {"L1": [1.0] * 4, "L2": [1.0] * 4},
    }
    sop = execute_sop(case, _view(quality="optical_blackout"))
    assert sop.decision_action == "human_review"
    assert sop.deterministic_verdict is None


def test_expanded_llm_rejects_invented_threshold_wrong_order_and_fiber():
    response = DiagnosisResponse(
        steps=(
            ReasoningStep("高于 99.0", sop_step_id=SOP_STEP_IDS[2]),
            ReasoningStep("回到质量检查", sop_step_id=SOP_STEP_IDS[0]),
        ),
        verdict="fiber",
    )
    request = SimpleNamespace(sop_candidates=("L1", "fiber"))
    violations = validate_expanded_llm_response(response, request)
    assert any("未声明阈值" in item for item in violations)
    assert any("顺序" in item for item in violations)
    assert any("fiber" in item for item in violations)


def test_expanded_llm_rejects_illegal_step_and_out_of_candidate_verdict():
    response = DiagnosisResponse(
        steps=(ReasoningStep("检查", sop_step_id="made_up"),), verdict="L2",
    )
    violations = validate_expanded_llm_response(response, SimpleNamespace(sop_candidates=("L1",)))
    assert any("sop_step_id" in item for item in violations)
    assert any("不在 SOP 候选" in item for item in violations)


def test_expanded_llm_accepts_declared_predicate_and_existing_evidence_reference():
    response = DiagnosisResponse(
        steps=(ReasoningStep(
            "SNR <= 0.0",
            sop_step_id=SOP_STEP_IDS[1],
            cited_predicates=("P_media_or_host_snr_floor",),
            cited_evidence=("physical:L1:media_snr:value_le_0",),
        ),),
        verdict="L1",
    )
    request = SimpleNamespace(
        sop_candidates=("L1",),
        evidence_tokens=("physical:L1:media_snr:value_le_0",),
        declared_predicates=({"predicate_id": "P_media_or_host_snr_floor"},),
    )
    assert validate_expanded_llm_response(response, request) == ()


def test_labels_do_not_enter_feature_or_graph_views():
    path = Path(__file__).resolve().parents[1] / "experiments/20260816_expanded-pattern-conflict/clean_train.jsonl"
    case = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    changed = dict(case, case_id="same-observations-different-label", label="fiber" if case["label"] != "fiber" else "L1")
    views, _, _, _ = build_views((case, changed))
    assert views[0].feature_tokens == views[1].feature_tokens
    assert views[0].graph_edges == views[1].graph_edges


def test_context_window_preflight_reserves_output_and_fails_before_model_load():
    class CharacterTokenizer:
        @staticmethod
        def encode(text, add_special_tokens=False):
            return list(text)

    assert validate_context_window(
        ("x" * 100,), CharacterTokenizer(), max_model_len=200, max_new_tokens=50,
    ) == [100]
    try:
        validate_context_window(
            ("x" * 100,), CharacterTokenizer(), max_model_len=181, max_new_tokens=50,
        )
    except ValueError as error:
        assert "required_max_model_len>=182" in str(error)
    else:
        raise AssertionError("context preflight should reject an undersized window")
