"""N5c 低匹配（未见模式）处理器。

这个分支处理的是「历史上没见过」的 case，因此不能靠复用历史结论，只能靠物理约束推理。
T6 会接入 LLM，本模块现在做的是 T6 之前必须先做对的三件事：

1. **筛出与本 case 实际相关的约束**，而不是把 14 条全量塞进 prompt。
   不相关的约束会稀释注意力，也会让模型去讨论根本没有观测到的量。
2. **先做可执行的确定性排除**，再交给 LLM。约束库里 `kind="exclusion"` 且能对
   `ROOT_CAUSES` 直接生效的部分，用代码判定比用 LLM 判定更可靠也更便宜。
   目前只有 `C6_tx_down_excludes_medium` 满足这个条件（`C3` / `C4` 是排除
   「温度 / 电压致因」，不映射到 L1 / L2 / fiber 三分类，因此不参与缩减）。
3. **在 LLM 缺席时不硬猜。** 确定性排除通常只能把 3 个候选缩到 2 个，
   缩不到 1 个就返回弃权（`verdict=None`），而不是退回类别先验。
   阶段 1 已经证明「零证据也给个答案」正是 legacy 的失败模式。

`DiagnosisRequest` 是本模块交给 T6 的接口：它把证据、约束、候选集打包成
一个与具体 LLM 后端无关的结构，prompt 模板在 M8 里渲染它。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple  # noqa: F401

from ..anomaly import DOWN_THRESHOLDS, lane_values
from ..constraints.library import CONSTRAINT_LIBRARY, Constraint, ConstraintLibrary
from ..evidence_graph.match import MatchResult
from ..evidence_graph.router import RoutingDecision
from ..evidence_pack import EvidencePack
from ..types import ROOT_CAUSES, SIDES
from .base import BranchCalibration, BranchOutcome, EvidenceLink

if False:  # pragma: no cover - typing only without importing optional module at runtime
    from ..features.extractor import CaseFeatures
    from ..sop import LearnedSOP


BRANCH = "N5c"

#: token 家族前缀 -> 相关约束类别。用来从 14 条里筛出与本 case 有关的那几条。
FAMILY_TO_CATEGORY: Dict[str, Tuple[str, ...]] = {
    "drop:": (
        "tx_power", "rx_power", "bias_current",
        "lane_directional_consistency", "attribution_direction",
    ),
    "status:": ("lane_directional_consistency", "attribution_direction"),
    "imbalance:": ("lane_directional_consistency", "rx_power", "attribution_direction"),
    "lane:": (
        "lane_directional_consistency", "tx_power", "rx_power", "attribution_direction",
    ),
    "level:": ("rx_power", "signal_quality", "attribution_direction"),
    "serdes:": ("signal_quality", "measurement_validity"),
    "telemetry:": ("measurement_validity",),
}

#: 量测有效性与可识别性类约束（caveat）无条件注入：它们的作用是阻止无效推理，
#: 而无效推理恰恰最可能发生在「没有相关观测」的时候。
#: `identifiability` 必须无条件注入的理由更直接：C20 说的是「不要给 fiber 结论」，
#: 如果只在命中疑似介质 token 时才注入，模型恰好在最想猜 fiber 的场景下看不到它。
ALWAYS_INJECTED_CATEGORIES: Tuple[str, ...] = ("measurement_validity", "identifiability")


@dataclass(frozen=True)
class Exclusion:
    """一次确定性排除。`constraint_id` 让报告可以追溯到具体哪条物理约束。"""

    root_cause: str
    constraint_id: str
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {"root_cause": self.root_cause, "constraint_id": self.constraint_id, "reason": self.reason}


@dataclass(frozen=True)
class DiagnosisRequest:
    """交给 M8 渲染 prompt 的载荷。与具体 LLM 后端无关。"""

    case_id: str
    evidence_tokens: Tuple[str, ...]
    missing_fields: Tuple[str, ...]
    telemetry_status: str
    candidate_root_causes: Tuple[str, ...]
    exclusions: Tuple[Exclusion, ...]
    constraint_ids: Tuple[str, ...]
    nearest_similarity: float
    branch: str
    routing_reason: str
    historical_case_ids: Tuple[str, ...]
    historical_label_distribution: Tuple[Tuple[str, int], ...]
    sop_prediction: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "evidence_tokens": list(self.evidence_tokens),
            "missing_fields": list(self.missing_fields),
            "telemetry_status": self.telemetry_status,
            "candidate_root_causes": list(self.candidate_root_causes),
            "exclusions": [item.to_dict() for item in self.exclusions],
            "constraint_ids": list(self.constraint_ids),
            "nearest_similarity": self.nearest_similarity,
            "branch": self.branch,
            "routing_reason": self.routing_reason,
            "historical_case_ids": list(self.historical_case_ids),
            "historical_label_distribution": {
                label: count for label, count in self.historical_label_distribution
            },
            "sop_prediction": dict(self.sop_prediction) if self.sop_prediction is not None else None,
        }


def tx_down_sides(pack: EvidencePack) -> Tuple[str, ...]:
    """找出发送光功率处于断光哨兵的侧。这是 `C6` 的触发条件。"""
    sentinel = DOWN_THRESHOLDS["txpower"]
    down: List[str] = []
    for side in SIDES:
        values = [value for value in lane_values(pack.telemetry, "txpower", side).values() if value is not None]
        if values and any(value <= sentinel for value in values):
            down.append(side)
    return tuple(down)


def deterministic_exclusions(pack: EvidencePack) -> Tuple[Exclusion, ...]:
    """执行约束库中可对 `ROOT_CAUSES` 直接生效的排除。

    目前只有 C6：本端没有发出光时，该方向的故障不可能是光纤造成的——
    光纤只能衰减已经进入它的光，不能解释一束从未被发出的光。

    C15 是 C6 的前置条件，必须先判。不加这个前置条件时，C6 在全量 211 条上触发 14 次、
    其中 2 次排掉了真实的 fiber 标签。追下去发现那 2 条属于全链路遥测失效，
    断光哨兵表示「读不到」而不是「没有光」，C6 的前提根本不成立。
    加上前置条件后，C6 触发 8 次、排错 0 次。
    """
    if pack.optical_blackout:
        return ()
    sides = tx_down_sides(pack)
    if not sides:
        return ()
    return (
        Exclusion(
            root_cause="fiber",
            constraint_id="C6_tx_down_excludes_medium",
            reason=(
                f"{'、'.join(sides)} 侧存在发送光功率处于断光哨兵（<= {DOWN_THRESHOLDS['txpower']:.0f} dBm）的 lane，"
                f"光纤无法解释一束从未被发出的光，故排除介质根因"
            ),
        ),
    )


def relevant_constraints(
    tokens: Sequence[str],
    library: ConstraintLibrary = CONSTRAINT_LIBRARY,
) -> Tuple[Constraint, ...]:
    """按本 case 实际触发的特征家族筛约束，再无条件补上量测有效性类。"""
    categories = set(ALWAYS_INJECTED_CATEGORIES)
    for token in tokens:
        for prefix, mapped in FAMILY_TO_CATEGORY.items():
            if token.startswith(prefix):
                categories.update(mapped)
    return tuple(item for item in library.constraints if item.category in categories)


def calibration_group(pack: EvidencePack, exclusions: Sequence[Exclusion]) -> str:
    if not pack.observed_fields:
        return "N5c_no_telemetry"
    return "N5c_with_exclusion" if exclusions else "N5c_no_exclusion"


def handle(
    result: MatchResult,
    decision: RoutingDecision,
    calibration: BranchCalibration,
    pack: EvidencePack,
    *,
    library: ConstraintLibrary = CONSTRAINT_LIBRARY,
    trace: Optional[Any] = None,
    features: Optional[Any] = None,
    sop_model: Optional[Any] = None,
) -> BranchOutcome:
    """`trace` 是 `llm.reason.ReasoningTrace`。为 None 时表示没接 LLM，本分支只做确定性排除。"""
    exclusions = deterministic_exclusions(pack)
    excluded = {item.root_cause for item in exclusions}
    candidates = tuple(label for label in ROOT_CAUSES if label not in excluded)
    constraints = relevant_constraints(result.query_tokens, library)
    group = calibration_group(pack, exclusions)
    confidence = calibration.confidence(group)
    confidence_lower_bound = calibration.lower_bound(group)
    calibration_support = calibration.support(group)

    chain: List[EvidenceLink] = [
        EvidenceLink(
            kind="no_historical_match",
            statement=decision.reason,
            tokens=result.query_tokens,
            source="evidence_graph",
        )
    ]
    for item in exclusions:
        chain.append(
            EvidenceLink(
                kind="constraint_exclusion",
                statement=item.reason,
                source=item.constraint_id,
            )
        )
    chain.append(
        EvidenceLink(
            kind="constraint_context",
            statement=(
                f"注入 {len(constraints)} 条与本 case 相关的物理约束"
                f"（{library.version}，全部待专家审核）"
            ),
            tokens=tuple(item.constraint_id for item in constraints),
            source=library.version,
        )
    )

    # 确定性排除通常把 3 个候选缩到 2 个，缩不到 1 个就不给结论。
    verdict = candidates[0] if len(candidates) == 1 else None
    caveats: List[str] = []
    missing = list(result.missing_evidence)
    if sop_model is not None and features is not None:
        prediction = sop_model.predict(features)
        chain.append(EvidenceLink(
            kind="learned_sop",
            statement=prediction.reason,
            tokens=tuple(prediction.path),
            source=sop_model.version,
        ))
        if prediction.verdict in candidates:
            # SOP 是训练集归纳的先验路径。没有 LLM 时允许它作为 dry-run
            # 候选；正式 SOP+LLM 路径必须由受约束 LLM 给出最终结论。
            if trace is None:
                verdict = prediction.verdict
                confidence = prediction.confidence
                confidence_lower_bound = prediction.confidence_lower_bound
                calibration_support = prediction.support
                group = f"sop:{prediction.leaf_id}"
        else:
            caveats.append(
                f"learned SOP 候选 {prediction.verdict} 已被确定性约束排除或不可用，转入补采/人工复核"
            )

    if trace is not None and trace.accepted is not None and trace.accepted.verdict is not None:
        verdict = trace.accepted.verdict
        confidence = trace.accepted.confidence
        confidence_lower_bound = 0.0
        calibration_support = 0
        group = f"llm_raw:{BRANCH}"
        for index, step in enumerate(trace.accepted.steps):
            chain.append(EvidenceLink(
                kind="llm_step",
                statement=f"第 {index + 1} 步：{step.claim}",
                tokens=tuple(step.cited_evidence),
                source="|".join(step.cited_constraints) or "llm",
            ))
        missing.extend(item for item in trace.accepted.missing_information if item not in missing)
        if trace.rewrote:
            caveats.append(
                f"该结论经过 {trace.attempt_count} 轮生成才通过物理约束校验，前几轮的违规记录见推理日志"
            )
    elif trace is not None:
        caveats.append(f"LLM 未给出可用结论：{trace.abstain_reason}")

    if verdict is None:
        caveats.append(
            f"物理约束只能把候选缩小到 {len(candidates)} 个（{'、'.join(candidates)}），"
            f"确定性推理无法定论"
            + ("" if trace is not None else "，需要 LLM 在约束内进一步推理")
        )
    if not pack.observed_fields:
        caveats.append("本 case 没有任何遥测观测，任何结论都不可信")
    if trace is not None and trace.evidence_check is not None and trace.evidence_check.violations:
        caveats.append(
            "证据包本身存在 "
            f"{len(trace.evidence_check.violations)} 条量测有效性告警，相关读数需谨慎采信"
        )

    return BranchOutcome(
        case_id=result.query_case_id,
        branch=BRANCH,
        verdict=verdict,
        confidence=confidence,
        confidence_lower_bound=confidence_lower_bound,
        calibration_group=group,
        calibration_support=calibration_support,
        evidence_chain=tuple(chain),
        reused_case_ids=(),
        missing_evidence=tuple(missing),
        caveats=tuple(caveats),
        needs_llm=trace is None,
        needs_human=not pack.observed_fields or (trace is not None and verdict is None),
    )


def build_request(
    result: MatchResult,
    pack: EvidencePack,
    *,
    decision: Optional[RoutingDecision] = None,
    library: ConstraintLibrary = CONSTRAINT_LIBRARY,
    features: Optional[Any] = None,
    sop_model: Optional[Any] = None,
) -> DiagnosisRequest:
    """构造 T6/T7 的约束推理或历史冲突仲裁载荷。"""
    exclusions = deterministic_exclusions(pack)
    excluded = {item.root_cause for item in exclusions}
    label_counts: Dict[str, int] = {}
    for candidate in result.top_candidates:
        if candidate.label is not None:
            label_counts[candidate.label] = label_counts.get(candidate.label, 0) + 1
    sop_prediction = (
        sop_model.predict(features).to_dict()
        if sop_model is not None and features is not None
        else None
    )
    return DiagnosisRequest(
        case_id=result.query_case_id,
        evidence_tokens=result.query_tokens,
        missing_fields=pack.missing_fields,
        telemetry_status=pack.telemetry_status,
        candidate_root_causes=tuple(label for label in ROOT_CAUSES if label not in excluded),
        exclusions=exclusions,
        constraint_ids=tuple(item.constraint_id for item in relevant_constraints(result.query_tokens, library)),
        nearest_similarity=result.max_similarity,
        branch=decision.branch if decision is not None else BRANCH,
        routing_reason=decision.reason if decision is not None else "历史匹配不足，走约束推理",
        historical_case_ids=tuple(item.case_id for item in result.top_candidates),
        historical_label_distribution=tuple(sorted(label_counts.items())),
        sop_prediction=sop_prediction,
    )
