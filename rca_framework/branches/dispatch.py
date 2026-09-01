"""把 N4 路由与 N5a / N5b / N5c / N6 处理器接起来。

置信度标定（`fit_calibration`）必须只用训练集留一法。用留出测试集标定等于把测试集
信息漏进置信度，报告出来的置信度会系统性偏高。这一点有单测锁定。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, MutableMapping, Optional, Sequence, Tuple

from ..constraints.library import CONSTRAINT_LIBRARY, ConstraintLibrary
from ..evidence_graph.match import MatchResult
from ..evidence_graph.router import (
    DEFAULT_POLICY,
    RoutingDecision,
    RoutingPolicy,
    route,
    route_many,
)
from ..evidence_pack import EvidencePack
from . import exact, general, partial
from .base import BranchCalibration, BranchOutcome, EvidenceLink, majority_label

if TYPE_CHECKING:
    from ..decision import LLMCalibration


def calibration_group_of(
    result: MatchResult,
    decision: RoutingDecision,
    pack: EvidencePack,
) -> str:
    """一个 case 属于哪个标定分组。路由与标定必须用同一套分组，否则置信度对不上号。"""
    if decision.branch == "N5a":
        return exact.calibration_group(result)
    if decision.branch == "N5b":
        return partial.calibration_group(
            result,
            use_dual_similarity=decision.policy_name == "filtered-rule-three-channel-v2",
        )
    if decision.branch == "N5c":
        return general.calibration_group(pack, general.deterministic_exclusions(pack))
    return "N6_abstain"


def provisional_verdict(
    result: MatchResult,
    decision: RoutingDecision,
    pack: EvidencePack,
) -> Optional[str]:
    """标定用的「如果按本分支的规则判，会判成什么」。

    N5c 在 T6 之前没有 LLM，确定性排除缩不到 1 个候选时它本就不给结论，
    因此这里也不给——标定表里 N5c 的准确率会如实反映这一点，不会被虚高的猜测撑起来。
    """
    if decision.branch == "N6":
        return None
    if decision.branch == "N5c":
        excluded = {item.root_cause for item in general.deterministic_exclusions(pack)}
        remaining = [label for label in ("L1", "L2", "fiber") if label not in excluded]
        return remaining[0] if len(remaining) == 1 else None
    candidates = (
        result.dual_top_candidates
        if decision.policy_name == "filtered-rule-three-channel-v2" and result.dual_top_candidates
        else result.top_candidates
    )
    return majority_label([item.label for item in candidates if item.label is not None])


def fit_calibration(
    results: Sequence[MatchResult],
    packs: Sequence[EvidencePack],
    labels: Sequence[str],
    *,
    policy: RoutingPolicy = DEFAULT_POLICY,
    source: str = "train-loo",
) -> BranchCalibration:
    """在训练集留一法结果上标定各分组的实测准确率。"""
    if not (len(results) == len(packs) == len(labels)):
        raise ValueError("results, packs and labels must be the same length")
    decisions = route_many(results, policy)
    groups: List[str] = []
    flags: List[bool] = []
    for result, decision, pack, actual in zip(results, decisions, packs, labels):
        groups.append(calibration_group_of(result, decision, pack))
        flags.append(provisional_verdict(result, decision, pack) == actual)
    return BranchCalibration.fit(groups, flags, source=f"{source}:{policy.name}")


def low_evidence_outcome(
    result: MatchResult,
    decision: RoutingDecision,
    calibration: BranchCalibration,
    *,
    trace: Optional[Any] = None,
) -> BranchOutcome:
    verdict = None
    confidence = 0.0
    confidence_breakdown = None
    self_reported_confidence = 0.0
    fallback_source = ""
    compliance_penalties = ()
    caveats = ["证据极低或全链路遥测失效，结论只能作为低置信候选"]
    chain = [
        EvidenceLink(kind="low_evidence_route", statement=decision.reason, source="router"),
    ]
    if trace is not None and trace.accepted is not None:
        verdict = trace.accepted.verdict
        confidence = trace.accepted.confidence
        confidence_breakdown = trace.accepted.confidence_breakdown.to_dict()
        self_reported_confidence = trace.accepted.self_reported_confidence
        fallback_source = trace.accepted.fallback_source
        compliance_penalties = trace.accepted.compliance_penalties
        for index, step in enumerate(trace.accepted.steps):
            chain.append(EvidenceLink(
                kind="llm_low_evidence_step",
                statement=f"低证据第 {index + 1} 步：{step.claim}",
                tokens=tuple(step.cited_evidence),
                source="|".join(step.cited_constraints) or "llm",
            ))
    return BranchOutcome(
        case_id=result.query_case_id,
        branch="N6",
        verdict=verdict,
        confidence=confidence,
        confidence_lower_bound=0.0,
        calibration_group="N6_abstain",
        calibration_support=calibration.support("N6_abstain"),
        evidence_chain=tuple(chain),
        caveats=tuple(caveats),
        needs_llm=trace is None,
        needs_human=True,
        confidence_breakdown=confidence_breakdown,
        self_reported_confidence=self_reported_confidence,
        history_verdict=None,
        fallback_source=fallback_source,
        compliance_penalties=compliance_penalties,
    )


abstain_outcome = low_evidence_outcome


def handle(
    result: MatchResult,
    pack: EvidencePack,
    calibration: BranchCalibration,
    *,
    policy: RoutingPolicy = DEFAULT_POLICY,
    library: ConstraintLibrary = CONSTRAINT_LIBRARY,
    trace: Optional[Any] = None,
    features: Optional[Any] = None,
    sop_model: Optional[Any] = None,
) -> Tuple[RoutingDecision, BranchOutcome]:
    decision = route(result, policy)
    if decision.branch == "N5a":
        return decision, exact.handle(result, decision, calibration, trace=trace)
    if decision.branch == "N5b":
        return decision, partial.handle(
            result,
            decision,
            calibration,
            trace=trace,
            use_dual_similarity=policy.use_dual_similarity,
        )
    if decision.branch == "N5c":
        return decision, general.handle(
            result,
            decision,
            calibration,
            pack,
            library=library,
            trace=trace,
            features=features,
            sop_model=sop_model,
        )
    return decision, low_evidence_outcome(result, decision, calibration, trace=trace)


def handle_many(
    results: Sequence[MatchResult],
    packs: Sequence[EvidencePack],
    calibration: BranchCalibration,
    *,
    policy: RoutingPolicy = DEFAULT_POLICY,
    library: ConstraintLibrary = CONSTRAINT_LIBRARY,
    reasoner: Optional[Any] = None,
    llm_calibration: Optional["LLMCalibration"] = None,
    trace_collector: Optional[MutableMapping[str, Any]] = None,
    features: Optional[Sequence[Any]] = None,
    sop_model: Optional[Any] = None,
) -> List[Tuple[RoutingDecision, BranchOutcome]]:
    """`reasoner` 是 `llm.ConstrainedReasoner`。给了就批量处理所有 case。

    刻意做成批量：先用确定性处理器识别 N5a 混合桶、N5b 关键缺失/标签冲突和 N5c，
    一次性生成，再把 trace 分发回去。这样 `needs_llm` 不再只是一个无人消费的标志。
    """
    decisions = route_many(results, policy)
    traces: Dict[str, Any] = {}
    if reasoner is not None:
        targets = [
            (index, result, pack, decision)
            for index, (result, pack, decision)
            in enumerate(zip(results, packs, decisions))
        ]
        if targets:
            requests = [
                general.build_request(
                    result,
                    pack,
                    decision=decision,
                    library=library,
                    features=features[index] if features is not None else None,
                    sop_model=sop_model,
                )
                for index, result, pack, decision in targets
            ]
            for trace in reasoner.reason_many(requests, [pack for _, _, pack, _ in targets]):
                traces[trace.case_id] = trace
    if trace_collector is not None:
        trace_collector.update(traces)

    paired = [
        handle(result, pack, calibration, policy=policy, library=library,
               trace=traces.get(result.query_case_id),
               features=features[index] if features is not None else None,
               sop_model=sop_model)
        for index, (result, pack) in enumerate(zip(results, packs))
    ]
    from ..decision import apply_llm_calibration

    return [
        (decision, apply_llm_calibration(outcome, traces.get(outcome.case_id), llm_calibration))
        for decision, outcome in paired
    ]
