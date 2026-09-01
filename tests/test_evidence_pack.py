"""T3 测试：证据包契约、标签隔离、缺失 / 冲突证据结构与 lane 级读数。

覆盖 Progress 第 4.2 节要求的四条路径：正常、缺失、冲突、lane 级特征。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rca_framework.anomaly import fit_thresholds
from rca_framework.data import load_cases
from rca_framework.evidence_pack import (
    CONTEXT_FIELDS,
    EVIDENCE_PACK_SCHEMA,
    EvidencePack,
    build_packs,
    labels_of,
)
from rca_framework.features.extractor import (
    detect_token_conflicts,
    extract_features,
    fit_feature_model,
)


DATA_DIR = Path("datasets/organized_rca_v2_stratified_60_40_seed42")
TRAIN_SIZE = 126


@pytest.fixture(scope="module")
def cases():
    return load_cases(DATA_DIR)


@pytest.fixture(scope="module")
def fitted(cases):
    train = cases[:TRAIN_SIZE]
    return fit_thresholds(train), fit_feature_model(build_packs(train))


def test_pack_strips_label_structurally(cases):
    """标签隔离是结构性的：证据包里根本没有标签字段可读。"""
    for case in cases[:30]:
        assert "label" in case
        pack = EvidencePack.from_case(case)
        assert not pack.has_label_field()
        assert "label" not in pack.telemetry
        assert "label" not in json.dumps(pack.to_dict(), ensure_ascii=False)


def test_labels_are_carried_separately(cases):
    train = cases[:TRAIN_SIZE]
    packs, labels = build_packs(train), labels_of(train)
    assert len(packs) == len(labels) == TRAIN_SIZE
    assert set(labels) == {"L1", "L2", "fiber"}


def test_pack_records_observed_and_missing_fields(cases):
    pack = EvidencePack.from_case(cases[0])
    assert pack.expected_field_count == 18  # 2 侧 * (5 指标 + 4 状态位)
    assert set(pack.observed_fields) & set(pack.missing_fields) == set()
    assert len(pack.observed_fields) + len(pack.missing_fields) == pack.expected_field_count
    # host_snr 在本数据集大面积缺失，见 Validation.md V9。
    assert "L1.host_snr" in pack.missing_fields
    assert "L1.host_snr" not in pack.diagnostic_missing_fields


def test_telemetry_status_separates_normal_from_absent():
    """零证据的两种相反含义必须可区分。"""
    empty = EvidencePack.from_case({"case_id": "empty"})
    assert empty.telemetry_status == "no_telemetry"
    assert empty.coverage == 0.0
    assert empty.observed_fields == ()

    partial = EvidencePack.from_case({
        "case_id": "partial",
        "rxpower": {"L1": {"0": 1.0}, "L2": {"0": 1.0}},
    })
    assert partial.telemetry_status == "partial_telemetry"
    assert 0.0 < partial.coverage < 1.0


def test_host_snr_is_optional_for_diagnostic_completeness():
    case = {"case_id": "optional-host"}
    for metric in ("bias", "txpower", "rxpower", "media_snr", "serdes_snr"):
        case[metric] = {side: {"0": 1.0} for side in ("L1", "L2")}
    for status in ("RxLOS", "RxLOL", "TxLOS", "TxLOL"):
        case[status] = {"L1": "Normal", "L2": "Normal"}
    pack = EvidencePack.from_case(case, source_dataset="all_data")
    assert pack.telemetry_status == "partial_telemetry"
    assert pack.diagnostic_telemetry_status == "full_telemetry"
    assert pack.diagnostic_missing_fields == ()


def test_readings_keep_down_sentinels_unfiltered():
    """lane 级读数不做 healthy 过滤，否则 tx_ok_rx_down 这类模式会在这一层就被抹掉。"""
    pack = EvidencePack.from_case({
        "case_id": "sentinel",
        "txpower": {"L1": {"0": 1.2, "1": -40.0}, "L2": {"0": 1.0}},
        "rxpower": {"L1": {"0": 0.9}, "L2": {"0": -40.0, "1": 0.5}},
    })
    tx = pack.reading("L1", "txpower")
    assert tx.lanes == {"0": 1.2, "1": -40.0}
    assert tx.lane_count == 2
    assert tx.observed
    assert pack.reading("L1", "media_snr").lanes == {}
    assert not pack.reading("L1", "media_snr").observed


def test_context_is_separated_from_telemetry(cases):
    pack = EvidencePack.from_case(cases[0])
    assert "alarm_name" in pack.context
    assert "Lane number" in pack.context
    for name in pack.context:
        assert name in CONTEXT_FIELDS
    # 上下文字段不进 signature，理由见 Progress 9.3。
    assert "rxpower" not in pack.context


def test_scalars_are_collected_for_constraint_checks(cases):
    pack = EvidencePack.from_case(cases[0])
    assert pack.scalars["L1.Temperature"] is not None
    assert pack.scalars["L2.Voltage"] is not None


def test_pack_round_trips_through_json(cases):
    pack = EvidencePack.from_case(cases[0], source_dataset="unit-test")
    restored = EvidencePack.from_dict(json.loads(json.dumps(pack.to_dict(), ensure_ascii=False)))
    assert restored.case_id == pack.case_id
    assert restored.observed_fields == pack.observed_fields
    assert restored.missing_fields == pack.missing_fields
    assert restored.telemetry_status == pack.telemetry_status
    assert restored.schema_version == EVIDENCE_PACK_SCHEMA
    assert restored.reading("L1", "rxpower").lanes == pack.reading("L1", "rxpower").lanes


def test_features_carry_missing_and_status_forward(cases, fitted):
    thresholds, model = fitted
    pack = EvidencePack.from_case(cases[0])
    features = extract_features(pack, thresholds, model)
    assert features.telemetry_status == pack.telemetry_status
    assert features.missing_fields == pack.missing_fields
    assert "telemetry_status" in features.to_dict()


def test_empty_signature_is_not_the_same_as_no_telemetry(cases, fitted):
    """空 signature 必须结合 telemetry_status 才能解释。"""
    thresholds, model = fitted
    blank = extract_features(EvidencePack.from_case({"case_id": "blank"}), thresholds, model)
    assert blank.is_empty
    assert blank.telemetry_status == "no_telemetry"

    train = cases[:TRAIN_SIZE]
    empty_but_measured = [
        extract_features(pack, thresholds, model) for pack in build_packs(train)
    ]
    silent = [item for item in empty_but_measured if item.is_empty]
    # T1 实测：126 条训练 case 里只有 2 条空 signature，且都是采到数但一切正常。
    assert len(silent) == 2
    assert {item.telemetry_status for item in silent} == {"partial_telemetry"}


def test_conflict_detection_flags_mutually_exclusive_buckets():
    assert detect_token_conflicts(["drop:L1:rxpower:single_lane", "drop:L1:rxpower:all_lanes"]) == [
        ("drop:L1:rxpower:all_lanes", "drop:L1:rxpower:single_lane")
    ]
    assert detect_token_conflicts(["level:L2:rxpower_mean:low_tail", "level:L2:rxpower_mean:high_tail"]) == [
        ("level:L2:rxpower_mean:high_tail", "level:L2:rxpower_mean:low_tail")
    ]
    # 不同侧、不同指标不算冲突。
    assert detect_token_conflicts(["drop:L1:rxpower:single_lane", "drop:L2:rxpower:single_lane"]) == []
    # fence 的高低可以同时成立：同一侧不同 lane 一个偏低一个偏高是合法观测。
    assert detect_token_conflicts(["fence:L1:rxpower:low", "fence:L1:rxpower:high"]) == []


def test_real_dataset_produces_no_feature_conflicts(cases, fitted):
    """抽取规则自洽性检查：真实数据上不应出现互斥分档同时成立。"""
    thresholds, model = fitted
    for pack in build_packs(cases):
        features = extract_features(pack, thresholds, model)
        assert features.conflicts == (), (features.case_id, features.conflicts)


def test_lane_level_features_survive_missing_side(fitted):
    """单侧缺失时抽取不得崩溃，且只产出有数据那一侧的 token。"""
    thresholds, model = fitted
    pack = EvidencePack.from_case({
        "case_id": "one-sided",
        "rxpower": {"L1": {"0": -40.0, "1": -40.0, "2": -40.0, "3": -40.0}},
    })
    features = extract_features(pack, thresholds, model)
    assert "drop:L1:rxpower:all_lanes" in features.tokens
    assert not any(token.startswith("drop:L2:") for token in features.tokens)
    assert "L2.rxpower" in features.missing_fields
