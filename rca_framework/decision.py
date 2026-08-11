"""M9 统一置信度、LLM 标定与降级决策。

历史匹配分支的置信度来自训练集留一法；LLM 分支不能直接把模型自报的 confidence
当作正确率。本模块把 LLM 输出按分支与 confidence 分桶，在独立的训练留一法输出上
统计正确率和 Wilson 下界，再用同一套出口决定最终结论、补采或人工介入。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .branches.base import BranchOutcome, wilson_lower_bound


DECISION_POLICY_VERSION = "decision-policy-v1"
LLM_CONFIDENCE_BINS: Tuple[float, ...] = (0.0, 0.5, 0.7, 0.9, 1.0)
DECISION_ACTIONS: Tuple[str, ...] = ("final", "request_evidence", "human_review")


def llm_calibration_group(branch: str, confidence: float) -> str:
    value = min(1.0, max(0.0, float(confidence)))
    for lower, upper in zip(LLM_CONFIDENCE_BINS[:-1], LLM_CONFIDENCE_BINS[1:]):
        if value < upper or upper == 1.0:
            closing = "]" if upper == 1.0 else ")"
            return f"llm:{branch}:[{lower:.1f},{upper:.1f}{closing}"
    return f"llm:{branch}:[0.9,1.0]"


@dataclass(frozen=True)
class LLMCalibration:
    """LLM 已回答样本的独立可靠性标定表。"""

    counts: Mapping[str, Tuple[int, int]] = field(default_factory=dict)
    source: str = ""

    @classmethod
    def fit(
        cls,
        outcomes: Sequence[BranchOutcome],
        traces: Sequence[Optional[Any]],
        labels: Sequence[str],
        *,
        source: str = "train-loo",
    ) -> "LLMCalibration":
        if not (len(outcomes) == len(traces) == len(labels)):
            raise ValueError("outcomes, traces and labels must be the same length")
        tally: Dict[str, list[int]] = {}
        for outcome, trace, truth in zip(outcomes, traces, labels):
            accepted = getattr(trace, "accepted", None) if trace is not None else None
            if accepted is None or accepted.verdict is None:
                continue
            group = llm_calibration_group(outcome.branch, accepted.confidence)
            row = tally.setdefault(group, [0, 0])
            row[0] += int(accepted.verdict == truth)
            row[1] += 1
        return cls(
            counts={key: (value[0], value[1]) for key, value in sorted(tally.items())},
            source=source,
        )

    def confidence(self, group: str) -> float:
        correct, total = self.counts.get(group, (0, 0))
        return round(correct / total, 6) if total else 0.0

    def lower_bound(self, group: str) -> float:
        correct, total = self.counts.get(group, (0, 0))
        return wilson_lower_bound(correct, total)

    def support(self, group: str) -> int:
        return self.counts.get(group, (0, 0))[1]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "confidence_bins": list(LLM_CONFIDENCE_BINS),
            "groups": {
                key: {
                    "correct": correct,
                    "total": total,
                    "accuracy": round(correct / total, 6) if total else 0.0,
                    "wilson_lower_bound": wilson_lower_bound(correct, total),
                }
                for key, (correct, total) in sorted(self.counts.items())
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LLMCalibration":
        return cls(
            counts={
                str(key): (int(item["correct"]), int(item["total"]))
                for key, item in value.get("groups", {}).items()
            },
            source=str(value.get("source", "")),
        )


def apply_llm_calibration(
    outcome: BranchOutcome,
    trace: Optional[Any],
    calibration: Optional[LLMCalibration],
) -> BranchOutcome:
    """把模型自报 confidence 替换为独立标定频率；无标定时保留原值但下界置零。"""
    accepted = getattr(trace, "accepted", None) if trace is not None else None
    if accepted is None or accepted.verdict is None:
        return outcome
    group = llm_calibration_group(outcome.branch, accepted.confidence)
    support = calibration.support(group) if calibration is not None else 0
    if support:
        return replace(
            outcome,
            confidence=calibration.confidence(group),
            confidence_lower_bound=calibration.lower_bound(group),
            calibration_group=group,
            calibration_support=support,
        )
    caveat = "LLM 结论尚无独立标定样本；模型自报置信度仅供记录，不能直接作为最终可靠性"
    return replace(
        outcome,
        confidence=accepted.confidence,
        confidence_lower_bound=0.0,
        calibration_group=f"uncalibrated:{group}",
        calibration_support=0,
        caveats=outcome.caveats + (() if caveat in outcome.caveats else (caveat,)),
    )


@dataclass(frozen=True)
class DecisionPolicy:
    """N6 统一出口策略。阈值保持可配置，便于 T10 做选择性风险消融。"""

    version: str = DECISION_POLICY_VERSION
    final_lower_bound: float = 0.5
    minimum_support: int = 10

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "final_lower_bound": self.final_lower_bound,
            "minimum_support": self.minimum_support,
        }


DEFAULT_DECISION_POLICY = DecisionPolicy()


@dataclass(frozen=True)
class FinalDecision:
    case_id: str
    branch: str
    action: str
    verdict: Optional[str]
    proposed_verdict: Optional[str]
    confidence: float
    confidence_lower_bound: float
    calibration_group: str
    calibration_support: int
    reason: str
    requested_evidence: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "branch": self.branch,
            "action": self.action,
            "verdict": self.verdict,
            "proposed_verdict": self.proposed_verdict,
            "confidence": self.confidence,
            "confidence_lower_bound": self.confidence_lower_bound,
            "calibration_group": self.calibration_group,
            "calibration_support": self.calibration_support,
            "reason": self.reason,
            "requested_evidence": list(self.requested_evidence),
        }


def decide(
    outcome: BranchOutcome,
    policy: DecisionPolicy = DEFAULT_DECISION_POLICY,
) -> FinalDecision:
    """把任意 N5 输出收敛成最终结论、补采请求或人工介入。"""
    reliable = (
        outcome.verdict is not None
        and outcome.calibration_support >= policy.minimum_support
        and outcome.confidence_lower_bound >= policy.final_lower_bound
    )
    if reliable:
        return FinalDecision(
            case_id=outcome.case_id,
            branch=outcome.branch,
            action="final",
            verdict=outcome.verdict,
            proposed_verdict=outcome.verdict,
            confidence=outcome.confidence,
            confidence_lower_bound=outcome.confidence_lower_bound,
            calibration_group=outcome.calibration_group,
            calibration_support=outcome.calibration_support,
            reason=(
                f"Wilson 95% 下界 {outcome.confidence_lower_bound:.2%} 达到"
                f"阈值 {policy.final_lower_bound:.2%}，且标定支持数"
                f" {outcome.calibration_support} >= {policy.minimum_support}"
            ),
        )

    if outcome.missing_evidence:
        action = "request_evidence"
        reason = "当前结论未通过可靠性门槛；先补齐分支列出的缺失证据再重新诊断"
    else:
        action = "human_review"
        if outcome.verdict is None:
            reason = "当前路径未形成可校验结论，且没有明确的自动补采项，转人工介入"
        else:
            reason = (
                f"候选结论未通过可靠性门槛（Wilson 下界 {outcome.confidence_lower_bound:.2%}，"
                f"支持数 {outcome.calibration_support}），转人工复核"
            )
    return FinalDecision(
        case_id=outcome.case_id,
        branch=outcome.branch,
        action=action,
        verdict=None,
        proposed_verdict=outcome.verdict,
        confidence=outcome.confidence,
        confidence_lower_bound=outcome.confidence_lower_bound,
        calibration_group=outcome.calibration_group,
        calibration_support=outcome.calibration_support,
        reason=reason,
        requested_evidence=outcome.missing_evidence if action == "request_evidence" else (),
    )


def decide_many(
    outcomes: Sequence[BranchOutcome],
    policy: DecisionPolicy = DEFAULT_DECISION_POLICY,
) -> Tuple[FinalDecision, ...]:
    return tuple(decide(outcome, policy) for outcome in outcomes)

