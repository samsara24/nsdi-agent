from __future__ import annotations

import pytest

from rca_framework.evidence_pack import build_packs
from scripts.run_offline_sop_llm_experiment import (
    _gpu_memory_released,
    _quality_summary,
    _validate_split,
)


def test_gpu_release_gate_allows_only_small_driver_residue():
    before = {
        "gpus": [
            {"uuid": "gpu-1", "memory_free_mb": 20000},
            {"uuid": "gpu-2", "memory_free_mb": 19000},
        ]
    }
    released = {
        "gpus": [
            {"uuid": "gpu-1", "memory_free_mb": 19800},
            {"uuid": "gpu-2", "memory_free_mb": 18800},
        ]
    }
    leaked = {
        "gpus": [
            {"uuid": "gpu-1", "memory_free_mb": 12000},
            {"uuid": "gpu-2", "memory_free_mb": 18800},
        ]
    }
    assert _gpu_memory_released(before, released) is True
    assert _gpu_memory_released(before, leaked) is False
    assert _gpu_memory_released(before, {"gpus": []}) is None


def test_split_gate_rejects_overlap_and_wrong_sizes():
    train = [{"case_id": "a", "label": "L1"}]
    test = [{"case_id": "b", "label": "fiber"}]
    _validate_split(train, test, expected_train_size=1, expected_test_size=1)
    with pytest.raises(ValueError, match="overlap"):
        _validate_split(train, [{"case_id": "a", "label": "L2"}], expected_train_size=1, expected_test_size=1)
    with pytest.raises(ValueError, match="train split size mismatch"):
        _validate_split(train, test, expected_train_size=161, expected_test_size=1)


def test_quality_summary_reports_required_l2fixed_fields():
    cases = [
        {
            "case_id": "a",
            "label": "L2",
            "alarm_ip_interface": None,
            "Lane number": None,
            "host_snr": {"L1": {"0": 10.0}},
            "rxpower": {"L1": {str(i): -2.0 for i in range(8)}, "L2": {"0": -2.0}},
        },
        {"case_id": "b", "label": "fiber"},
    ]
    packs = build_packs(cases)
    summary = _quality_summary(cases, packs)
    assert summary["case_count"] == 2
    assert summary["missing_alarm_ip_interface"] == 2
    assert summary["missing_lane_number"] == 2
    assert summary["host_snr_present_cases"] == 1
    assert summary["l1_metric_width_over_4_cases"] == 1
    assert summary["l2_metric_width_over_4_cases"] == 0
    assert summary["label_distribution"] == {"L2": 1, "fiber": 1}
