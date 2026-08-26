from rca_framework.decision_graph_policy import (
    LEARNED_PATH_CONFIGS, learned_path_match, receive_symptom_context,
)
from scripts.evaluate_filtered_rule_decision_graph import (
    acquisition_recommendations, attribution, physical_votes, resolve_votes, stripped,
)


def test_blind_input_removes_all_target_labels():
    rows = stripped([{"case_id": "x", "label": "L1", "original_label": "l1", "rxpower": {}}])
    assert rows == [{"case_id": "x", "rxpower": {}}]


def test_physical_votes_require_directional_evidence_and_do_not_bind_lane_id():
    votes = physical_votes((
        "lane:L1_to_L2:tx_down",
        "status:L1:TxLOS",
        "lane_scope:L1_to_L2:tx_down:single_lane",
    ))
    assert votes[0]["verdict"] == "L1"
    assert votes[0]["strength"] == "strong"
    assert not any(any("lane0" in evidence for evidence in vote["evidence"]) for vote in votes)


def test_label_aware_attribution_separates_missing_from_graph_gap():
    base = {
        "verdict": None,
        "telemetry_status": "partial_telemetry",
        "decision_votes": [],
        "features": {"tokens": ["topology_level:L1:rxpower_mean:low_tail"]},
        "similarity": {"exact_labels": {}},
        "expert": {"group": "expert:no_anomaly"},
    }
    assert attribution(base, "L2", "unreviewed")[0] == "missing_evidence"
    base["expert"] = {"group": "expert:multi_metric"}
    assert attribution(base, "L2", "unreviewed")[0] == "decision_graph_gap"


def test_receive_symptom_requires_opposite_tx_fault_for_endpoint_direction():
    anomalies = {"L1": {}, "L2": {}}
    assert receive_symptom_context(
        "expert:single:rxpower", "L2", "L1", (), anomalies,
    ) == "uncorroborated_receive_symptom"
    assert receive_symptom_context(
        "expert:single:rxpower", "L2", "L1", ("lane:L2_to_L1:tx_ok_rx_down",), anomalies,
    ) == "sender_tx_ok_receive_down"
    assert receive_symptom_context(
        "expert:single:rxpower", "L2", "L1", ("status:L2:TxLOS",), anomalies,
    ) == "opposite_tx_fault"


def test_moderate_physical_conflict_vetoes_strong_endpoint_vote():
    votes = [
        {"verdict": "L2", "strength": "strong"},
        {"verdict": "fiber", "strength": "moderate"},
    ]
    verdict, action, reason = resolve_votes(votes)
    assert verdict is None
    assert action == "insufficient"
    assert "冲突" in reason


def test_learned_positive_path_is_topology_rule_and_direction_specific():
    rows = [
        {"case_id": "a", "label": "L2", "topology_id": "logical8", "group": "expert:multi_metric",
         "verdict": "L2", "tokens": ["x", "y"]},
        {"case_id": "b", "label": "L2", "topology_id": "logical8", "group": "expert:multi_metric",
         "verdict": "L2", "tokens": ["x", "z"]},
        {"case_id": "c", "label": "L1", "topology_id": "logical4", "group": "expert:multi_metric",
         "verdict": "L2", "tokens": ["x", "y"]},
    ]
    match = learned_path_match(tokens=["x", "y"], topology_id="logical8",
                               group="expert:multi_metric", verdict="L2", training_rows=rows)
    assert match is not None
    assert match["purity"] == 1.0
    assert {item["case_id"] for item in match["neighbors"]} == {"a", "b"}


def test_missing_recommendations_preserve_missing_and_request_synchronized_evidence():
    advice = acquisition_recommendations("partial_telemetry", ["L1.TxLOS", "L2.rxpower"], ())
    rendered = " ".join(advice)
    assert "同一时刻" in rendered
    assert "lane级Tx/Rx" in rendered
    assert "missing" in rendered


def test_rxpower_requires_multiple_neighbors_and_serdes_is_advisory_only():
    assert LEARNED_PATH_CONFIGS["expert:single:rxpower"]["min_neighbors"] >= 2
    assert LEARNED_PATH_CONFIGS["expert:single:serdes_snr"]["terminal"] == 0
