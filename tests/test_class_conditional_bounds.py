"""迭代 2：按类别校准门限必须真的救回少数类，且不能靠多数类掩盖风险。

迭代 1 的实测形态是：统一门限 0.4104 下整体选择性风险 29.58% 达标，
但 L1 召回只有 6.25%、平衡召回 0.2596——门限把 L1 候选整体挡在门外，
整体精度全靠先验 62.1% 的 L2 撑着。这里用一份复刻该结构的最小数据
（L2 候选置信度普遍高于 L1）验证两件事：
统一门限会挡掉 L1；按类别校准会放开 L1 且各类风险各自达标。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rca_framework.decision import (
    DecisionCandidate,
    DecisionPolicy,
    fit_decision_policy,
    refine_per_label_bounds,
    simulate_gate,
)


def candidate(verdict: str, lower_bound: float, support: int = 20) -> DecisionCandidate:
    return DecisionCandidate(
        source="sop",
        verdict=verdict,
        confidence=lower_bound + 0.1,
        confidence_lower_bound=lower_bound,
        support=support,
        group=f"sop:{verdict}",
        reason="fixture",
    )


def band(verdict: str, base: float, correct: int, total: int = 10):
    """构造一段下界连续、命中数为 `correct` 的同类候选。"""
    return [
        (
            [candidate(verdict, base + 0.007 * index)],
            verdict if index < correct else ("L1" if verdict == "L2" else "L2"),
        )
        for index in range(total)
    ]


def build_rows():
    """复刻单一门限失效的真实结构：两类的可用区间**互不嵌套**。

    L2 候选分两段：0.55 以上纯度 70%（刚好压在 30% 风险目标上），
    0.40~0.47 那段全错；L1 候选同样分两段：0.30~0.36 纯度 70%（同样达标），
    0.20 那段纯度只有 30%。

    一个门限只能表达「下界高于 t 全收」：要收进 0.30 那段可用的 L1，
    就必然连 0.40 那段全错的 L2 一起收进来，整体风险随即超标；
    于是统一拟合只能退到 0.55，把纯度本来达标的 L1 整体挡在门外。
    按类别校准能同时表达「L2 收紧到 0.55、L1 放宽到 0.30」，
    这是单一标量无法表达的工作点。
    """
    return (
        band("L2", 0.55, correct=7)
        + band("L2", 0.40, correct=0)
        + band("L1", 0.30, correct=7)
        + band("L1", 0.20, correct=3)
    )


def test_single_threshold_blocks_the_minority_class():
    rows = build_rows()
    policy, fit = fit_decision_policy(
        rows, target_selective_risk=0.30, minimum_support=10, candidate_order=("sop",)
    )
    stats = simulate_gate(rows, policy)
    # 门限被不可用的那段 L2 顶到 0.55，纯度达标的 L1 一条都过不去。
    assert policy.final_lower_bound >= 0.55
    assert "L1" not in stats["by_predicted_label"]
    assert stats["by_predicted_label"]["L2"]["answered"] == 10
    # 整体风险达标，代价是覆盖率只有 1/4。
    assert stats["selective_risk"] <= 0.30
    assert stats["coverage"] == 0.25
    assert fit["feasible"] is True


def test_class_conditional_bounds_recover_the_minority_class():
    rows = build_rows()
    policy, fit = fit_decision_policy(
        rows,
        target_selective_risk=0.30,
        minimum_support=10,
        candidate_order=("sop",),
        class_conditional=True,
    )
    stats = simulate_gate(rows, policy)
    assert policy.per_label_lower_bound["L1"] < policy.per_label_lower_bound["L2"]
    assert stats["by_predicted_label"]["L1"]["answered"] == 10
    # 覆盖率从 25% 翻到 50%，且新增的全部是此前被整体牺牲的 L1。
    assert stats["coverage"] == 0.5
    assert stats["balanced_recall"] > simulate_gate(
        rows,
        DecisionPolicy(final_lower_bound=0.55, minimum_support=10, candidate_order=("sop",)),
    )["balanced_recall"]
    # 关键：放开 L1 之后，**每一类**的风险都要各自达标，
    # 不允许用 L2 的正确率抵消 L1 的错误。
    for row in stats["by_predicted_label"].values():
        assert row["selective_risk"] <= 0.30
    assert fit["class_conditional"]["labels_missing_target"] == []


def test_a_class_that_cannot_meet_the_target_is_reported_not_hidden():
    """某一类无论怎么调门限都达不到目标时，必须显式记录在 labels_missing_target。

    这里 L1 候选的正确率只有 30%，任何门限都无法把风险压到 30% 以下。
    正确行为是把它标出来（并保持在最严格门限），而不是悄悄放宽目标。
    """
    rows = []
    for index in range(10):
        rows.append(([candidate("L2", 0.55 + 0.007 * index)], "L2" if index < 8 else "L1"))
    for index in range(10):
        rows.append(([candidate("L1", 0.30 + 0.008 * index)], "L1" if index < 3 else "L2"))

    base = DecisionPolicy(final_lower_bound=0.5, minimum_support=10, candidate_order=("sop",))
    refined, report = refine_per_label_bounds(rows, base, target_selective_risk=0.30)
    assert "L1" not in report["final"]["by_predicted_label"] or (
        report["final"]["by_predicted_label"]["L1"]["answered"] == 0
    )
    assert refined.per_label_lower_bound["L1"] >= base.final_lower_bound


def test_per_label_bounds_round_trip_through_serialisation():
    policy = DecisionPolicy(
        final_lower_bound=0.41,
        minimum_support=10,
        candidate_order=("branch", "sop"),
        per_label_lower_bound={"L1": 0.31, "L2": 0.58},
    )
    payload = policy.to_dict()
    assert payload["per_label_lower_bound"] == {"L1": 0.31, "L2": 0.58}
    assert policy.lower_bound_for("L1") == 0.31
    assert policy.lower_bound_for("fiber") == 0.41
    assert policy.lower_bound_for(None) == 0.41
