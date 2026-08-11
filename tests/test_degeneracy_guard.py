"""迭代 1 守护：退化解必须在指标里露出来。

`coverage` 和 `precision_when_answered` 在 rca_v2_l2fixed 上会给「一律报 L2」
打出 100% 覆盖、62.6% 精度、0% 人工干预的成绩单，看起来三项全优，
实际对 L1 与 fiber 完全无用。`degeneracy_guard` 就是为这件事加的，
所以它自己必须有测试——否则守护本身失效时没人知道。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rca_framework.decision import FinalDecision
from rca_framework.types import ROOT_CAUSES
from scripts.evaluate_routing import class_metrics, degeneracy_guard


def decision(case_id: str, verdict: str | None) -> FinalDecision:
    return FinalDecision(
        case_id=case_id,
        branch="N5c",
        action="final" if verdict else "human_review",
        verdict=verdict,
        proposed_verdict=verdict,
        confidence=0.7 if verdict else 0.0,
        confidence_lower_bound=0.6 if verdict else 0.0,
        calibration_group="test",
        calibration_support=20,
        reason="fixture",
    )


#: 6 条 L2、3 条 L1、1 条 fiber，比例接近 rca_v2_l2fixed 的类别先验。
TRUTHS = ["L2"] * 6 + ["L1"] * 3 + ["fiber"]


def guard_for(verdicts, truths=TRUTHS, sop_predictions=None):
    decisions = [decision(f"case_{index}", verdict) for index, verdict in enumerate(verdicts)]
    return degeneracy_guard(
        decisions,
        truths,
        class_metrics([item.verdict for item in decisions], truths),
        sop_predictions=sop_predictions,
    )


def test_always_predicting_the_majority_class_shows_zero_lift():
    guard = guard_for(["L2"] * len(TRUTHS))
    assert guard["majority_label"] == "L2"
    # 全部作答且全部报多数类，同子集基线与自身完全相同。
    assert guard["majority_on_kept"] == 0.6
    assert guard["lift_over_majority_on_kept"] == 0.0
    # 只召回一个类，平衡召回被钉在 1/3 附近，而 precision_when_answered 会是 0.6。
    assert guard["balanced_recall"] < 0.34


def test_a_system_that_actually_separates_classes_shows_positive_lift():
    guard = guard_for(["L2"] * 6 + ["L1"] * 3 + ["fiber"])
    assert guard["lift_over_majority_on_kept"] == 0.4
    assert guard["balanced_recall"] == 1.0


def test_abstaining_only_on_cases_the_fallback_would_miss_scores_high():
    """弃答有效性衡量的是「人工有没有用在对的地方」。

    这里让系统在 3 条 case 上弃答，而 SOP 兜底在这 3 条上全错，
    有效性应当是 1.0；如果换成兜底全对，弃答就是纯浪费，有效性为 0。
    """
    verdicts = ["L2"] * 6 + [None, None, None] + ["fiber"]
    wrong_fallback = [{"verdict": "L2"} for _ in TRUTHS]  # 真值是 L1，兜底会全错
    guard = guard_for(verdicts, sop_predictions=wrong_fallback)
    assert guard["abstention_effectiveness"]["abstained"] == 3
    assert guard["abstention_effectiveness"]["with_sop_fallback"] == 3
    assert guard["abstention_effectiveness"]["effectiveness"] == 1.0

    right_fallback = [{"verdict": truth} for truth in TRUTHS]
    guard = guard_for(verdicts, sop_predictions=right_fallback)
    assert guard["abstention_effectiveness"]["effectiveness"] == 0.0


def test_sop_fallback_accepts_both_dataclass_and_mapping():
    class Prediction:
        verdict = "L2"

    verdicts = ["L2"] * 6 + [None, None, None] + ["fiber"]
    from_objects = guard_for(verdicts, sop_predictions=[Prediction() for _ in TRUTHS])
    from_mappings = guard_for(verdicts, sop_predictions=[{"verdict": "L2"} for _ in TRUTHS])
    assert (
        from_objects["abstention_effectiveness"]["effectiveness"]
        == from_mappings["abstention_effectiveness"]["effectiveness"]
    )


def test_guard_survives_an_all_abstain_run():
    """MVP 的真实形态：107 条全部转人工。守护不能在这种情况下崩。"""
    guard = guard_for([None] * len(TRUTHS))
    assert guard["majority_on_kept"] is None
    assert guard["lift_over_majority_on_kept"] is None
    assert guard["balanced_recall"] == 0.0
    assert set(ROOT_CAUSES) == {"L1", "L2", "fiber"}
