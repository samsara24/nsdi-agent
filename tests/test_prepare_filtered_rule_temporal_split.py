from scripts.prepare_filtered_rule_temporal_split import (
    canonicalize_endpoints,
    expert_proposals,
    measurement_fingerprint,
    parse_alarm_time,
    stable_case_id,
    unified_label,
)
from rca_framework.topology import SOURCE_TOPOLOGIES


def test_parse_alarm_time_accepts_both_source_formats():
    assert parse_alarm_time("2025-09-02 10:43:02").strftime("%Y-%m") == "2025-09"
    assert parse_alarm_time("2025/9/2  10:43:02").strftime("%Y-%m") == "2025-09"


def test_measurement_fingerprint_ignores_case_and_label_metadata():
    left = {"label": "l1", "case_id": "old", "bias": {"l1": {"0": 7.2}}}
    right = {"label": "l2", "case_id": "new", "bias": {"L1": {"0": 7.2}}}
    assert measurement_fingerprint(left) == measurement_fingerprint(right)


def test_stable_case_id_does_not_change_when_label_is_adjusted():
    left = {"label": "l1", "alarm_time": "2025-06-01 00:00:00", "bias": {}}
    right = {**left, "label": "l2"}
    assert stable_case_id("all_data", left) == stable_case_id("all_data", right)


def test_source_relative_labels_share_one_label_space():
    assert unified_label("all_data", "l1") == "L1"
    assert unified_label("all_data", "l2") == "L2"
    assert unified_label("rule1_channel_not_4", "l3") == "L1"
    assert unified_label("rule1_channel_not_4", "l4") == "L2"
    assert unified_label("rule1_channel_not_4", "fiber") == "fiber"


def test_topology_contract_keeps_logical_optical_lane_pairing():
    all_data = SOURCE_TOPOLOGIES["all_data"]
    rule1 = SOURCE_TOPOLOGIES["rule1_channel_not_4"]
    assert all_data["endpoint_speeds"] == {"L1": "400G", "L2": "200G"}
    assert rule1["endpoint_speeds"] == {"L1": "400G", "L2": "400G"}
    assert all_data["same_index_optical_pairing"] is True
    assert rule1["same_index_optical_pairing"] is True
    assert all_data["absolute_link_loss_allowed"] is False
    assert rule1["absolute_link_loss_allowed"] is False


def test_endpoint_keys_are_normalized_recursively():
    case = {
        "bias": {"l3": {"0": 1.0}, "l4": {"0": 2.0}},
        "transmission": {"l3-l4": 1, "l4-l3": 2},
    }
    actual = canonicalize_endpoints(case, {"l3": "L1", "l4": "L2"})
    assert actual["bias"] == {"L1": {"0": 1.0}, "L2": {"0": 2.0}}
    assert actual["transmission"] == {"L1-L2": 1, "L2-L1": 2}


def test_expert_annotation_file_has_no_conflicting_explicit_labels(tmp_path):
    path = tmp_path / "annotations.json"
    path.write_text(
        '{"schema_version":"rca-expert-label-review-v1","annotations":['
        '{"completed":true,"left_case_id":"a","right_case_id":"b",'
        '"left_label":"L1","right_label":"keep"}]}'
    )
    proposals, summary = expert_proposals(path)
    assert proposals["a"]["explicit_label"] == "L1"
    assert proposals["b"]["explicit_label"] is None
    assert summary["completed_annotation_count"] == 1
