from __future__ import annotations

from copy import deepcopy

from rca_framework.expert_diagnosis import (
    annotation_pattern_audit,
    build_expert_diagnosis_graph,
    classify_reading,
    review_training_case,
)


def _case() -> dict:
    lanes = {str(index): 7.4 for index in range(4)}
    optical = {str(index): 0.0 for index in range(4)}
    media = {str(index): 25.0 for index in range(4)}
    serdes = {str(index): 700000.0 for index in range(4)}
    return {
        "case_id": "synthetic", "label": "L2",
        "bias": {"L1": dict(lanes), "L2": dict(lanes)},
        "txpower": {"L1": dict(optical), "L2": dict(optical)},
        "rxpower": {"L1": dict(optical), "L2": dict(optical)},
        "media_snr": {"L1": dict(media), "L2": dict(media)},
        "host_snr": {"L1": dict(media), "L2": dict(media)},
        "serdes_snr": {"L1": dict(serdes), "L2": dict(serdes)},
    }


def test_zero_semantics_are_metric_specific():
    assert classify_reading("txpower", 0.0) == "light_present"
    assert classify_reading("rxpower", 0.0) == "light_present"
    assert classify_reading("media_snr", 0.0) == "invalid_or_floor"
    assert classify_reading("host_snr", 0.0) == "invalid_or_floor"
    assert classify_reading("serdes_snr", 0.0) == "invalid_state"
    assert classify_reading("serdes_snr", 1.0) == "invalid_state"
    assert classify_reading("bias", 0.0) == "laser_not_driven"


def test_minus_40_is_distinct_from_engineering_drop_boundary():
    assert classify_reading("rxpower", -40.0) == "exact_minus_40_sentinel"
    assert classify_reading("rxpower", -39.5) == "no_light"
    assert classify_reading("rxpower", -38.9) == "light_present"


def test_hard_receive_pattern_keeps_endpoint_and_fiber_competing():
    case = _case()
    case["rxpower"]["L1"]["1"] = -40.0
    case["media_snr"]["L1"]["1"] = 0.0
    case["serdes_snr"]["L1"]["1"] = 1.0
    graph = build_expert_diagnosis_graph(case)
    assert "EP_RX_HARD_DOWN" in graph["matched_patterns"]
    edges = {(item["src"], item["type"], item["dst"]) for item in graph["edges"]}
    assert ("pattern:EP_RX_HARD_DOWN:L1", "SUPPORTS", "candidate:L1") in edges
    assert ("pattern:EP_RX_HARD_DOWN:L1", "COMPETES_WITH", "candidate:fiber") in edges
    assert ("pattern:EP_RX_HARD_DOWN:L1", "COMPETES_WITH", "candidate:L2") in edges


def test_case_label_never_changes_diagnosis_signature():
    case = _case()
    left = build_expert_diagnosis_graph(case)
    changed = deepcopy(case)
    changed["label"] = "fiber"
    right = build_expert_diagnosis_graph(changed)
    assert left["diagnostic_signature"] == right["diagnostic_signature"]
    assert left["content_hash"] == right["content_hash"]


def test_every_graph_preserves_sop_step_order():
    graph = build_expert_diagnosis_graph(_case())
    step_nodes = [node for node in graph["nodes"] if node["type"] == "SOPStep"]
    precedes = [edge for edge in graph["edges"] if edge["type"] == "PRECEDES"]
    assert len(step_nodes) == 5
    assert len(precedes) == 4
    assert {(edge["src"], edge["dst"]) for edge in precedes} == {
        ("step:Q0", "step:P_TX"), ("step:P_TX", "step:R_RX"),
        ("step:R_RX", "step:F_MEDIUM"), ("step:F_MEDIUM", "step:D"),
    }


def test_absolute_tx_rx_loss_annotations_are_audited_not_compiled():
    audit = annotation_pattern_audit({
        "annotations": [
            {"pattern_id": "safe", "notes": "L1侧TX-40"},
            {"pattern_id": "unsafe", "notes": "L1和L2传输光衰（TX - RX）异常"},
        ]
    })
    assert audit["safe_for_pattern_mining_count"] == 1
    assert audit["requires_domain_confirmation_count"] == 1
    assert audit["requires_domain_confirmation"][0]["pattern_id"] == "unsafe"


def test_blackout_has_priority_over_endpoint_patterns():
    case = _case()
    for side in ("L1", "L2"):
        for metric in ("txpower", "rxpower"):
            case[metric][side] = {str(index): -40.0 for index in range(4)}
        case["media_snr"][side] = {str(index): 0.0 for index in range(4)}
    graph = build_expert_diagnosis_graph(case)
    assert graph["matched_patterns"] == ["EP_Q0_BLACKOUT"]
    assert not any(node["type"] == "CandidateRootCause" for node in graph["nodes"])


def test_receive_hard_down_is_not_an_endpoint_decision():
    case = _case()
    case["rxpower"]["L1"]["1"] = -40.0
    case["media_snr"]["L1"]["1"] = 0.0
    case["serdes_snr"]["L1"]["1"] = 1.0
    graph = build_expert_diagnosis_graph(case)
    review = review_training_case(case, graph)
    assert review["review_class"] == "direction_observed_root_not_identifiable"
    assert review["candidate_set"] == ["L1", "L2", "fiber"]


def test_local_tx_off_overrides_ambiguous_receive_symptom():
    case = _case()
    case["label"] = "L2"
    case["bias"]["L2"]["1"] = 0.0
    case["txpower"]["L2"]["1"] = -40.0
    case["rxpower"]["L1"]["1"] = -40.0
    case["media_snr"]["L1"]["1"] = 0.0
    case["serdes_snr"]["L1"]["1"] = 1.0
    graph = build_expert_diagnosis_graph(case)
    review = review_training_case(case, graph)
    assert review["candidate_set"] == ["L2"]
    assert review["label_assessment"] == "consistent_with_decisive_evidence"


def test_fiber_label_always_requests_field_evidence():
    case = _case()
    case["label"] = "fiber"
    graph = build_expert_diagnosis_graph(case)
    review = review_training_case(case, graph, unsafe_expert_reasoning=True)
    assert review["label_assessment"] == "fiber_label_requires_external_evidence"
    assert "OTDR" in review["required_evidence"]
    assert review["unsafe_expert_reasoning"] is True
