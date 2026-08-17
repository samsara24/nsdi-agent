from pathlib import Path

from rca_framework.expanded_evidence import (
    case_quality_state,
    is_exact_sentinel,
    measurement_state,
    physical_evidence_paths,
    quality_compatible,
)
from scripts.analyze_expanded_rca_patterns import load_cases, load_expert_annotations


def _case(values):
    case = {}
    for metric, block in values.items():
        case[metric] = block
    return case


def test_optical_sentinel_and_drop_boundary_are_distinct():
    state = measurement_state("rxpower", (-40.0, -39.5, -38.999))
    assert is_exact_sentinel(-40.0)
    assert not is_exact_sentinel(-39.999)
    assert state["sentinel_count"] == 1
    assert state["boundary_count"] == 2
    assert state["bucket"] == "partial_lanes"


def test_zero_and_serdes_one_boundaries_are_explicit():
    assert measurement_state("media_snr", (0.0, 0.001))["boundary_count"] == 1
    assert measurement_state("host_snr", (-0.001, 0.001))["boundary_count"] == 1
    serdes = measurement_state("serdes_snr", (0.0, 1.0, 1.001))
    assert serdes["boundary_count"] == 2
    assert serdes["boundary"] == 1.0


def test_blackout_precedes_physical_fault_paths():
    block = {"L1": [-40.0] * 4, "L2": [-40.0] * 4}
    case = _case({
        "txpower": block,
        "rxpower": block,
        "media_snr": {"L1": [0.0] * 4, "L2": [0.0] * 4},
        "host_snr": {"L1": [0.0] * 4, "L2": [0.0] * 4},
        "serdes_snr": {"L1": [1.0] * 4, "L2": [1.0] * 4},
    })
    assert case_quality_state(case)["quality"] == "optical_blackout"
    paths = physical_evidence_paths(case)
    assert {path["predicate_type"] for path in paths} == {"data_quality"}
    assert all("optical_blackout" in path["predicate"] for path in paths)


def test_relation_paths_never_claim_lane_pairing_or_absolute_loss():
    case = _case({
        "txpower": {"L1": [0.0] * 4, "L2": [0.0] * 4},
        "rxpower": {"L1": [-40.0, 0.0, 0.0, 0.0], "L2": [0.0] * 4},
        "media_snr": {"L1": [0.0, 20.0, 20.0, 20.0], "L2": [20.0] * 4},
        "host_snr": {"L1": [20.0] * 4, "L2": [20.0] * 4},
        "serdes_snr": {"L1": [1.0, 700000.0, 700000.0, 700000.0], "L2": [700000.0] * 4},
    })
    paths = physical_evidence_paths(case)
    relations = [path for path in paths if path["predicate_type"] == "cross_end_relation"]
    assert relations
    assert all("no lane pairing" in path["quantifier"] for path in relations)
    assert not any("TX-RX" in path["criterion"] or "loss" in path["criterion"].lower() for path in paths)


def test_quality_state_participates_in_exact_match_gate():
    valid = _case({
        "txpower": {"L1": [0.0], "L2": [0.0]}, "rxpower": {"L1": [0.0], "L2": [0.0]},
        "media_snr": {"L1": [20.0], "L2": [20.0]}, "host_snr": {"L1": [20.0], "L2": [20.0]},
        "serdes_snr": {"L1": [700000.0], "L2": [700000.0]},
    })
    missing = dict(valid)
    missing.pop("host_snr")
    assert quality_compatible(valid, valid)
    assert not quality_compatible(valid, missing)


def test_expert_pair_annotations_deduplicate_without_label_conflicts():
    root = Path(__file__).resolve().parents[1]
    expert_path = Path("/Users/ziangchen/Downloads/expert_label_annotations.json")
    if not expert_path.exists():
        return
    old = load_cases(root / "datasets/organized_rca_v2_stratified_60_40_seed42")
    new = load_cases(root / "datasets/all_data_rca_v2_stratified_60_40_seed42")
    labels = {str(case["case_id"]): str(case["label"]) for case in old + new}
    result = load_expert_annotations(expert_path, labels)
    assert result["pair_count"] == 44
    assert result["case_count"] == 66
    assert result["changed_count"] == 35
    assert result["secondary_physics_review_pair_count"] == 6
