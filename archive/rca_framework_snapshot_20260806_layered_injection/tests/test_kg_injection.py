import json

from rca_framework.llm import (
    PathLLMReasoner,
    build_layered_prompt,
    build_path_prompt,
    classify_kg_coverage,
    parse_llm_json,
)
from rca_framework.types import Anomaly, CaseEvidence


def anomaly(anomaly_id: str) -> Anomaly:
    return Anomaly(
        anomaly_id=anomaly_id, node_type="Metric", noun=f"{anomaly_id} 异常",
        relation="INDICATES", side="L1", metric="rxpower", severity=1.0,
    )


def make_case(*anomaly_ids: str) -> CaseEvidence:
    return CaseEvidence(
        case_id="case-x", label="", anomalies=[anomaly(item) for item in anomaly_ids],
        observed_fields=8, expected_fields=10, missing_fields=["bias"],
    )


def graph_result(*, paths: int, rules: int, similarity: float) -> dict:
    return {
        "prediction": "L2",
        "confidence": 0.5,
        "scores": {"L1": 0.30, "L2": 0.65, "fiber": 0.05},
        "paths": [
            {"root_cause": "L2", "anomaly_id": f"a{i}", "score": 0.1, "edge_statistics": {"precision": 0.7}}
            for i in range(paths)
        ],
        "path_count": paths,
        "retrieved_cases": [{"case_id": "t1", "root_cause": "L2", "similarity": similarity, "overlap_anomalies": []}],
        "feature_profile_scores": {"L1": 0.0, "L2": 1.0, "fiber": 0.0},
        "matched_feature_rules": {"L1": [], "L2": [{"rule_id": "KG_RULE_L2_0001"}] * rules, "fiber": []},
        "evidence_coverage": 0.8,
    }


def test_coverage_regime_is_decided_by_kg_structure_only() -> None:
    covered = classify_kg_coverage(make_case("a0"), graph_result(paths=3, rules=1, similarity=0.02))
    partial = classify_kg_coverage(make_case("a0"), graph_result(paths=3, rules=0, similarity=0.90))
    uncovered = classify_kg_coverage(make_case(), graph_result(paths=0, rules=0, similarity=0.90))
    assert covered["regime"] == "covered"
    assert partial["regime"] == "partial"
    assert uncovered["regime"] == "uncovered"


def test_layered_prompt_withholds_kg_score_when_case_is_not_covered() -> None:
    case = make_case("a0")
    uncovered = json.loads(build_layered_prompt(make_case(), graph_result(paths=0, rules=0, similarity=0.0)).split("\n", 1)[1])
    partial = json.loads(build_layered_prompt(case, graph_result(paths=3, rules=0, similarity=0.1)).split("\n", 1)[1])
    covered = json.loads(build_layered_prompt(case, graph_result(paths=3, rules=1, similarity=0.1)).split("\n", 1)[1])

    assert "candidate_path_scores" not in uncovered
    assert "root_cause_paths" not in uncovered
    assert "candidate_path_scores" not in partial
    assert partial["root_cause_paths"], "atom-level path statistics stay available under partial coverage"
    assert covered["candidate_path_scores"] == {"L1": 0.30, "L2": 0.65, "fiber": 0.05}
    assert covered["matched_kg_feature_rules"]["L2"]


def test_layered_prompt_never_leaks_the_kg_prediction() -> None:
    for paths, rules in ((0, 0), (3, 0), (3, 1)):
        payload = build_layered_prompt(make_case("a0"), graph_result(paths=paths, rules=rules, similarity=0.1))
        assert '"prediction":"L2"' not in payload.replace(" ", "")


def test_full_injection_prompt_is_unchanged() -> None:
    payload = json.loads(build_path_prompt(make_case("a0"), graph_result(paths=3, rules=1, similarity=0.1)).split("\n", 1)[1])
    assert payload["candidate_path_scores"] == {"L1": 0.30, "L2": 0.65, "fiber": 0.05}
    assert "kg_coverage" not in payload
    assert payload["constraints"][-1] == "必须在 L1、L2、fiber 中选择一个结果。"


def test_llm_only_scores_are_independent_of_the_kg_score() -> None:
    result = graph_result(paths=3, rules=1, similarity=0.1)
    # A hedged minority-class answer is exactly where the blended KG prior
    # overtakes the LLM's own choice.
    parsed = {"prediction": "fiber", "confidence": 0.30}
    legacy = PathLLMReasoner(score_mode="legacy")._score_llm_result(result, parsed)
    independent = PathLLMReasoner(score_mode="llm_only")._score_llm_result(result, parsed)
    assert legacy["L2"] > legacy["fiber"], "the legacy blend lets the KG majority class outrank the LLM answer"
    assert independent["fiber"] > independent["L2"], "the independent route keeps the LLM's own choice on top"
    assert independent["fiber"] == 0.30 + (1 - 0.30) / 3
    assert independent["L1"] == independent["L2"] == (1 - 0.30) / 3


def test_predicted_class_is_always_the_argmax_of_the_llm_score() -> None:
    reasoner = PathLLMReasoner(score_mode="llm_only")
    result = graph_result(paths=3, rules=1, similarity=0.1)
    for confidence in (0.0, 0.05, 0.2, 0.34, 0.5, 0.75, 1.0):
        scores = reasoner._score_llm_result(result, {"prediction": "fiber", "confidence": confidence})
        best = max(scores, key=lambda label: scores[label])
        assert best == "fiber" or confidence == 0.0, f"argmax flipped away from the prediction at {confidence}"
        assert abs(sum(scores.values()) - 1.0) < 1e-9


def test_zero_confidence_falls_back_to_a_uniform_distribution() -> None:
    scores = PathLLMReasoner(score_mode="llm_only")._score_llm_result(
        graph_result(paths=0, rules=0, similarity=0.0), {"prediction": "fiber", "confidence": 0.0},
    )
    assert scores == {"L1": 1 / 3, "L2": 1 / 3, "fiber": 1 / 3}


def test_sufficiency_is_parsed_and_defaults_to_unreported() -> None:
    body = '{"prediction":"fiber","confidence":0.6,"path_ids":[],"reasoning":"r","missing_information":[]}'
    with_field = json.loads(body)
    with_field["evidence_sufficiency"] = "insufficient"
    assert parse_llm_json(json.dumps(with_field))["evidence_sufficiency"] == "insufficient"
    assert parse_llm_json(body)["evidence_sufficiency"] == "unreported"


def test_reasoner_rejects_reconfiguring_model_loading_settings() -> None:
    reasoner = PathLLMReasoner()
    reasoner.configure(injection_mode="full", score_mode="legacy")
    assert reasoner.injection_mode == "full"
    try:
        reasoner.configure(model_path="/other/model")
    except ValueError:
        return
    raise AssertionError("changing model_path through configure must fail")
