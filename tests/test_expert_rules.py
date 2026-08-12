"""专家决策树测试：规则忠实度、方向知识的可归因性、以及它在 M9 里的接法。

这批测试有三个层次，缺一不可：

1. **忠实度**——阈值、优先级、短路顺序必须与 `docs/EXPERT_EXPERIENCE.md` 逐字一致。
   规则一旦被「顺手改好」，它在现网被验证过这件事就不再成立。这里连文档承认的
   缺陷（字符串排序、每指标只保留一个异常）也一并锁住。
2. **可归因性**——把方向表反转后精度必须显著低于多数类。这条断言保护的不是
   某个数字，而是「+14pp 来自归因方向知识」这个结论本身可复现；若哪天反转变得
   无所谓了，说明增益已经改由别的东西提供，结论必须重写。
3. **接线**——专家候选进入 M9 级联后的行为。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest

from rca_framework.data import cases_by_manifest_split
from rca_framework.decision import (
    CANDIDATE_SOURCES,
    DecisionPolicy,
    build_candidates,
    expert_candidate,
)
from rca_framework.branches.base import BranchOutcome
from rca_framework.evidence_pack import EvidencePack, build_packs, labels_of
from rca_framework.expert import (
    ANOMALY_LEVEL,
    DOC_VARIANT,
    EXPERT_THRESHOLDS,
    SINGLE_METRIC_BASE,
    SINGLE_METRIC_DIRECTION,
    ExpertCalibration,
    ExpertVariant,
    detect_anomaly,
    diagnose,
    diagnose_many,
    diagnose_side,
)
from rca_framework.knowledge import out_of_fold_expert_predictions


DATA_DIR = Path("datasets/rca_v2_l2fixed")


@pytest.fixture(scope="module")
def train_split():
    cases = cases_by_manifest_split(DATA_DIR, "train")
    return build_packs(cases, source_dataset=str(DATA_DIR)), labels_of(cases)


@pytest.fixture(scope="module")
def test_split():
    cases = cases_by_manifest_split(DATA_DIR, "test")
    return build_packs(cases, source_dataset=str(DATA_DIR)), labels_of(cases)


def make_case(**overrides: Any) -> Dict[str, Any]:
    """一条两端四 lane 的健康 case，各项都落在专家阈值的正常带内。"""
    healthy = {
        "rxpower": 1.0,
        "txpower": 1.0,
        "media_snr": 25.0,
        "host_snr": 25.0,
        "serdes_snr": 700000.0,
    }
    case: Dict[str, Any] = {
        "case_id": "synthetic",
        "label": "L2",
        "alarm_ip_interface": "L2_ENDPOINT--200G_PORT",
        "link_side_ip_interface_map": {
            "L1": "L1_ENDPOINT--400G_PORT",
            "L2": "L2_ENDPOINT--200G_PORT",
        },
    }
    for metric, value in healthy.items():
        case[metric] = {side: {str(lane): value for lane in range(4)} for side in ("L1", "L2")}
    for key, value in overrides.items():
        case[key] = value
    return case


def pack_with(side: str, metric: str, lanes: Dict[str, float]) -> EvidencePack:
    case = make_case()
    case[metric] = dict(case[metric])
    case[metric][side] = lanes
    return EvidencePack.from_case(case)


# --------------------------------------------------------------------------
# 1. 对文档的忠实度
# --------------------------------------------------------------------------


def test_thresholds_match_expert_document():
    """§3.3 阈值表逐字核对。改动任何一格都必须先改文档。"""
    assert EXPERT_THRESHOLDS["rxpower"] == {"down": -40.0, "low": -2.5, "high": 4.6, "diff": 1.0}
    assert EXPERT_THRESHOLDS["txpower"] == {"down": -40.0, "low": -2.5, "high": 2.5, "diff": 1.3}
    assert EXPERT_THRESHOLDS["host_snr"] == {"down": 0.0, "low": 22.8, "high": 27.5, "diff": 2.5}
    assert EXPERT_THRESHOLDS["media_snr"] == {"down": 0.0, "low": 22.4, "high": 28.7, "diff": 3.0}
    assert EXPERT_THRESHOLDS["serdes_snr"] == {
        "down": 0.0,
        "low": 458750.0,
        "high": 947750.0,
        "diff": 230000.0,
    }


def test_direction_table_matches_expert_document():
    """§5.3：接收类观测指向对端，发送与本地数字侧观测指向本端。"""
    assert SINGLE_METRIC_DIRECTION == {
        "host_snr": "same",
        "serdes_snr": "same",
        "media_snr": "opposite",
        "rxpower": "opposite",
        "txpower": "same",
    }
    assert SINGLE_METRIC_BASE == {
        "host_snr": 2,
        "serdes_snr": 3,
        "media_snr": 4,
        "rxpower": 5,
        "txpower": 6,
    }
    assert ANOMALY_LEVEL == {"lane_down": 0, "low_value": 1, "high_value": 1, "lane_diff": 2}


def test_anomaly_detection_short_circuits_in_document_order():
    """§3.1：命中靠前的类型就返回，同一指标不可能同时报低值与 lane_diff。

    这是文档 §8 自己点出的表达能力缺陷。锁住它是为了保证实现的是被验证过的
    那套规则，而不是一个「看起来更合理」的变体。
    """
    assert detect_anomaly("rxpower", [-40.0, 1.0, 1.0, 1.0]) == "lane_down"
    # 同时满足低值与 lane_diff，只报低值。
    assert detect_anomaly("rxpower", [-3.0, 1.0, 1.0, 1.0]) == "low_value"
    assert detect_anomaly("rxpower", [5.0, 1.0, 1.0, 1.0]) == "high_value"
    assert detect_anomaly("rxpower", [2.5, 1.0, 1.0, 1.0]) == "lane_diff"
    assert detect_anomaly("rxpower", [1.0, 1.2, 1.1, 1.0]) is None
    assert detect_anomaly("rxpower", []) is None


def test_host_snr_all_nonpositive_is_treated_as_absent():
    """§2.3：某端 host_snr 没有任何正值时整项作废，而不是判为 0 值异常。"""
    pack = pack_with("L1", "host_snr", {str(lane): None for lane in range(4)})
    diagnosis = diagnose_side(pack, "L1")
    assert diagnosis is None


def test_txpower_lane_down_wins_and_points_to_its_own_side():
    """§5.1：发送侧断光是最高优先级，且定界在异常所在端。"""
    pack = pack_with("L1", "txpower", {"0": -40.0, "1": 1.0, "2": 1.0, "3": 1.0})
    diagnosis = diagnose_side(pack, "L1")
    assert diagnosis is not None
    assert (diagnosis.rule, diagnosis.priority, diagnosis.location) == (
        "txpower_lane_down",
        "0",
        "L1",
    )


def test_multi_metric_requires_all_three_and_points_to_far_end():
    """§5.2：三项同时异常才触发，指向对端；缺一项就退回单指标模式。"""
    case = make_case()
    case["rxpower"] = dict(case["rxpower"])
    case["media_snr"] = dict(case["media_snr"])
    case["serdes_snr"] = dict(case["serdes_snr"])
    case["rxpower"]["L1"] = {"0": -3.0, "1": 1.0, "2": 1.0, "3": 1.0}
    case["media_snr"]["L1"] = {"0": 20.0, "1": 25.0, "2": 25.0, "3": 25.0}
    case["serdes_snr"]["L1"] = {"0": 400000.0, "1": 700000.0, "2": 700000.0, "3": 700000.0}
    diagnosis = diagnose_side(EvidencePack.from_case(case), "L1")
    assert diagnosis is not None
    assert (diagnosis.rule, diagnosis.priority, diagnosis.location) == ("multi_metric", "1", "L2")

    # 去掉 serdes 这一项后不再触发组合模式，退回 host/serdes 之后的单指标模式。
    case["serdes_snr"]["L1"] = {str(lane): 700000.0 for lane in range(4)}
    fallback = diagnose_side(EvidencePack.from_case(case), "L1")
    assert fallback is not None
    assert fallback.rule != "multi_metric"


def test_priority_uses_string_ordering_as_documented():
    """§5.3 + §8：priority 是字符串拼接与字符串排序，这是文档记录在案的取值方式。"""
    pack = pack_with("L1", "media_snr", {"0": 20.0, "1": 25.0, "2": 25.0, "3": 25.0})
    diagnosis = diagnose_side(pack, "L1")
    assert diagnosis is not None
    # media_snr 基础优先级 4，低值级别 1，拼成 "41"，并指向对端。
    assert diagnosis.priority == "41"
    assert isinstance(diagnosis.priority, str)
    assert diagnosis.location == "L2"


def test_fiber_requires_equal_priority_and_different_location():
    """§6.2：两端同 priority 且定界不同才判 fiber。"""
    case = make_case()
    case["media_snr"] = dict(case["media_snr"])
    case["media_snr"]["L1"] = {"0": 20.0, "1": 25.0, "2": 25.0, "3": 25.0}
    case["media_snr"]["L2"] = {"0": 20.0, "1": 25.0, "2": 25.0, "3": 25.0}
    diagnosis = diagnose(EvidencePack.from_case(case))
    assert diagnosis.verdict == "fiber"
    assert diagnosis.group == "expert:both_anomaly"

    # 只有一端异常时不判 fiber。
    case["media_snr"]["L2"] = {str(lane): 25.0 for lane in range(4)}
    single = diagnose(EvidencePack.from_case(case))
    assert single.verdict == "L2"
    assert single.group == "expert:single:media_snr"


def test_no_anomaly_falls_back_to_alarm_side():
    """§6.2 的兜底出口指向告警端，而不是固定某一侧。"""
    diagnosis = diagnose(EvidencePack.from_case(make_case()))
    assert diagnosis.group == "expert:no_anomaly"
    assert diagnosis.verdict == "L2"
    assert diagnosis.alarm_side_resolved is True

    swapped = make_case(alarm_ip_interface="L1_ENDPOINT--400G_PORT")
    assert diagnose(EvidencePack.from_case(swapped)).verdict == "L1"


def test_fallbacks_can_be_disabled_for_ablation():
    """关掉兜底后这两个出口必须弃权，否则消融量到的「判别力」是假的。"""
    variant = ExpertVariant(name="no_fallback", use_fallbacks=False)
    assert diagnose(EvidencePack.from_case(make_case()), variant=variant).verdict is None


def test_variant_rejects_invalid_direction():
    with pytest.raises(ValueError):
        ExpertVariant(name="bad", multi_metric_direction="sideways")


# --------------------------------------------------------------------------
# 2. 增益的可归因性
# --------------------------------------------------------------------------


def _accuracy(packs, labels, variant) -> float:
    diagnoses = diagnose_many(packs, variant=variant)
    hits = sum(1 for item, truth in zip(diagnoses, labels) if item.verdict == truth)
    return hits / len(labels)


def test_direction_knowledge_carries_the_gain(test_split):
    """反转方向表后精度必须掉到多数类以下，且远低于原方向。

    这条断言是 §9.33 那个结论的可执行版本：专家规则的价值在于**方向**，
    不在于「有异常就报某一端」。三个对照组都必须输给文档方向。
    """
    packs, labels = test_split
    majority = max(labels.count(label) for label in set(labels)) / len(labels)
    doc = _accuracy(packs, labels, DOC_VARIANT)

    reversed_variant = ExpertVariant(
        name="reverse",
        single_metric_direction={
            metric: ("same" if value == "opposite" else "opposite")
            for metric, value in SINGLE_METRIC_DIRECTION.items()
        },
        multi_metric_direction="same",
        txpower_lane_down_direction="opposite",
    )
    always_same = ExpertVariant(
        name="same",
        single_metric_direction={metric: "same" for metric in SINGLE_METRIC_DIRECTION},
        multi_metric_direction="same",
        txpower_lane_down_direction="same",
    )
    always_opposite = ExpertVariant(
        name="opposite",
        single_metric_direction={metric: "opposite" for metric in SINGLE_METRIC_DIRECTION},
        multi_metric_direction="opposite",
        txpower_lane_down_direction="opposite",
    )

    assert doc > majority + 0.10
    assert _accuracy(packs, labels, reversed_variant) < majority - 0.20
    assert _accuracy(packs, labels, always_same) < majority
    assert _accuracy(packs, labels, always_opposite) < majority


def test_expert_rules_do_not_overfit_the_train_split(train_split, test_split):
    """规则不含拟合参数，因此测试集不应当系统性地更差。"""
    train_packs, train_labels = train_split
    test_packs, test_labels = test_split
    train_accuracy = _accuracy(train_packs, train_labels, DOC_VARIANT)
    test_accuracy = _accuracy(test_packs, test_labels, DOC_VARIANT)
    assert test_accuracy >= train_accuracy - 0.05


def test_alarm_side_resolution_is_not_the_source_of_the_gain(test_split):
    """去掉告警端解析后精度不应显著下降，否则增益来自 alarm_ip_interface 而非物理知识。"""
    packs, labels = test_split
    baseline = _accuracy(packs, labels, DOC_VARIANT)
    without = _accuracy(packs, labels, ExpertVariant(name="no_alarm", resolve_alarm_side=False))
    assert without >= baseline - 0.02


# --------------------------------------------------------------------------
# 3. 标定与 M9 接线
# --------------------------------------------------------------------------


def test_calibration_groups_rules_not_cases(train_split):
    packs, labels = train_split
    calibration = ExpertCalibration.fit(diagnose_many(packs), labels, source="unit-test")
    groups = calibration.to_dict()["groups"]
    assert "expert:multi_metric" in groups
    assert sum(item["total"] for item in groups.values()) == len(labels)
    for group, item in groups.items():
        assert 0.0 <= item["wilson_lower_bound"] <= item["accuracy"] + 1e-9


def test_out_of_fold_calibration_is_not_more_optimistic(train_split):
    """折外标定给出的下界不应当整体高于全量标定，否则说明折外实现有泄漏。"""
    packs, labels = train_split
    in_sample = ExpertCalibration.fit(diagnose_many(packs), labels)
    oof = out_of_fold_expert_predictions(packs, labels)
    assert len(oof) == len(labels)
    answered = [item for item in oof if item is not None]
    assert answered
    mean_oof = sum(item["confidence"] for item in answered) / len(answered)
    mean_in = sum(
        in_sample.confidence(item["group"]) for item in answered
    ) / len(answered)
    assert mean_oof <= mean_in + 1e-6


def test_expert_is_a_registered_candidate_source():
    assert "expert" in CANDIDATE_SOURCES
    DecisionPolicy(candidate_order=("expert", "branch", "sop"))


def test_expert_candidate_wins_cascade_when_it_passes_gate():
    outcome = BranchOutcome(
        case_id="c1",
        branch="N5c",
        verdict=None,
        confidence=0.0,
        confidence_lower_bound=0.0,
        calibration_group="N5c_no_exclusion",
        calibration_support=0,
    )
    prediction = {
        "verdict": "L1",
        "confidence": 0.78,
        "confidence_lower_bound": 0.55,
        "support": 18,
        "group": "expert:single:rxpower",
        "reason": "L2 侧命中 single:rxpower，指向 L1",
    }
    candidate = expert_candidate(prediction)
    assert candidate is not None
    assert (candidate.source, candidate.verdict, candidate.group) == (
        "expert",
        "L1",
        "expert:single:rxpower",
    )

    policy = DecisionPolicy(
        final_lower_bound=0.5, minimum_support=10, candidate_order=("expert", "branch", "sop")
    )
    candidates = build_candidates(
        outcome,
        sop_prediction={
            "verdict": "L2",
            "confidence": 0.66,
            "confidence_lower_bound": 0.56,
            "support": 100,
            "leaf_id": "root",
        },
        expert_prediction=prediction,
        policy=policy,
    )
    assert [item.source for item in candidates] == ["expert", "sop"]


def test_expert_candidate_is_absent_without_calibration():
    assert expert_candidate(None) is None
    assert expert_candidate({"verdict": None}) is None
