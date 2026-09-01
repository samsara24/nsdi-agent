"""N5b 部分匹配处理器。

N5b 的核心不是用 token 前缀猜“缺失是否关键”，而是把缺失证据交给物理约束：
若某个缺失 token 会改变纯物理归因 / 排除路径，则缺它就不能直接复用历史链路，
需要补采或 LLM 仲裁。量测契约是否决不可信推理的 veto，训练集区间与叶节点统计
也不参与关键证据判定；P4 明确声明的「正常带内发送电平」同理不算关键缺失。
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

from ..constraints.measurement import MeasurementContract
from ..constraints.physics import PHYSICS_LIBRARY, PhysicalConstraint
from ..evidence_graph.match import MatchResult
from ..evidence_graph.router import RoutingDecision
from .base import BranchCalibration, BranchOutcome, EvidenceLink, majority_label


BRANCH = "N5b"

# 缺失时会改变归因 / 排除路径的物理约束。下面两类故意排除：
# 1) P2/P3/P4：规格带或「正常带内发送电平不是归因证据」——缺它们不能触发仲裁。
# 2) 量测契约：它们是否决「怎么推理」的 veto，不是「必须采到这条证据」的清单。
_KEY_WHEN_MISSING_PHYSICS_IDS = frozenset(
    {
        "P1_bias_zero_means_laser_off",
        "P5_tx_down_excludes_medium",
        "P6_rx_has_continuous_degradation",
        "P7_tx_ok_rx_down_means_path_loss",
        "P8_bidirectional_symmetry_points_shared_path",
        "P9_scope_all_lanes_vs_single_lane",
        "P10_receive_symptom_points_to_far_transmit_chain",
        "P11_single_lane_does_not_exclude_fiber",
        "P12_receive_lane_imbalance_removes_common_mode",
        "P13_local_signal_metrics_point_local",
    }
)


def _matches_any_prefix(token: str, prefixes: Sequence[str]) -> bool:
    return any(token.startswith(prefix) for prefix in prefixes)


def physical_key_reasons(
    token: str,
    *,
    physical_constraints: Sequence[PhysicalConstraint] = PHYSICS_LIBRARY.constraints,
    measurement_contracts: Sequence[MeasurementContract] = (),
) -> Tuple[str, ...]:
    """Return physics IDs that make a missing token critical for historical reuse.

    Measurement contracts remain available for callers that need veto text, but they
    do not participate in N5b key-evidence gating by default.
    """

    del measurement_contracts  # retained for call-site compatibility; vetoes are not key evidence
    reasons = [
        constraint.constraint_id
        for constraint in physical_constraints
        if constraint.constraint_id in _KEY_WHEN_MISSING_PHYSICS_IDS
        and _matches_any_prefix(token, constraint.applies_to_token_prefixes)
    ]
    return tuple(sorted(set(reasons)))


def critical_missing(missing: Tuple[str, ...]) -> Tuple[str, ...]:
    return tuple(token for token in missing if physical_key_reasons(token))


def _branch_candidates(result: MatchResult, *, use_dual_similarity: bool) -> Tuple[Any, ...]:
    if use_dual_similarity and result.dual_top_candidates:
        return result.dual_top_candidates
    return result.top_candidates


def _missing_for(candidates: Sequence[Any]) -> Tuple[str, ...]:
    if not candidates:
        return ()
    common = set(candidates[0].missing_evidence)
    for candidate in candidates[1:]:
        common &= set(candidate.missing_evidence)
    return tuple(sorted(common))


def calibration_group(result: MatchResult, *, use_dual_similarity: bool = False) -> str:
    missing = _missing_for(_branch_candidates(result, use_dual_similarity=use_dual_similarity))
    return "N5b_critical_gap" if critical_missing(missing) else "N5b_minor_gap"


def handle(
    result: MatchResult,
    decision: RoutingDecision,
    calibration: BranchCalibration,
    *,
    trace: Optional[Any] = None,
    use_dual_similarity: bool = False,
) -> BranchOutcome:
    top = _branch_candidates(result, use_dual_similarity=use_dual_similarity)
    labels = [candidate.label for candidate in top if candidate.label is not None]
    verdict = majority_label(labels)
    history_verdict = verdict
    missing = _missing_for(top)
    critical = critical_missing(missing)
    group = calibration_group(result, use_dual_similarity=use_dual_similarity)
    confidence = calibration.confidence(group)
    confidence_lower_bound = calibration.lower_bound(group)
    calibration_support = calibration.support(group)

    chain = [
        EvidenceLink(
            kind="partial_match",
            statement=(
                f"与 {len(top)} 条历史 case 部分匹配（相似度 {result.max_similarity:.2f}，"
                f"证据覆盖率 {result.evidence_coverage:.0%}）"
            ),
            tokens=tuple(top[0].shared_evidence) if top else (),
            source="evidence_graph",
        )
    ]

    caveats: List[str] = []
    if missing:
        chain.append(
            EvidenceLink(
                kind="missing_evidence",
                statement=f"历史 case 具备但本 case 未观测到的证据共 {len(missing)} 条，建议补采",
                tokens=missing,
                source="evidence_graph.missing",
            )
        )
    if critical:
        reason_rows = [
            f"{token}({', '.join(physical_key_reasons(token))})"
            for token in critical
        ]
        caveats.append(
            f"缺失的 {len(critical)} 条证据被物理归因约束判为关键，"
            f"结论需要补采后复核：{'、'.join(reason_rows)}"
        )
    if top and top[0].evidence_chain_summary:
        chain.append(
            EvidenceLink(
                kind="historical_chain_context",
                statement=(
                    "最高相似历史 case 的证据链路："
                    + " -> ".join(top[0].evidence_chain_summary)
                ),
                source=top[0].case_id,
            )
        )
    if top and top[0].missing_chain_steps:
        chain.append(
            EvidenceLink(
                kind="missing_chain_steps",
                statement=(
                    "当前 case 未覆盖的历史链路步骤："
                    + " -> ".join(top[0].missing_chain_steps)
                ),
                source=top[0].case_id,
            )
        )
    label_pure = len(set(labels)) == 1
    if not label_pure:
        caveats.append("并列候选的历史标签不一致，结论取多数投票")

    extra = tuple(top[0].extra_evidence) if top else ()
    if extra:
        chain.append(
            EvidenceLink(
                kind="extra_evidence",
                statement=f"本 case 有 {len(extra)} 条历史 case 不具备的证据，可能比历史更严重或场景不同",
                tokens=extra,
                source="evidence_graph.extra",
            )
        )

    arbitration_required = bool(critical) or not label_pure
    needs_human = False
    confidence_breakdown = None
    self_reported_confidence = 0.0
    fallback_source = ""
    compliance_penalties = ()
    if trace is not None:
        if trace.accepted is not None and trace.accepted.verdict is not None:
            verdict = trace.accepted.verdict
            confidence = trace.accepted.confidence
            confidence_breakdown = trace.accepted.confidence_breakdown.to_dict()
            self_reported_confidence = trace.accepted.self_reported_confidence
            fallback_source = trace.accepted.fallback_source
            compliance_penalties = trace.accepted.compliance_penalties
            confidence_lower_bound = 0.0
            calibration_support = 0
            group = f"llm_raw:{BRANCH}"
            for index, step in enumerate(trace.accepted.steps):
                chain.append(
                    EvidenceLink(
                        kind="llm_arbitration_step",
                        statement=f"仲裁第 {index + 1} 步：{step.claim}",
                        tokens=tuple(step.cited_evidence),
                        source="|".join(step.cited_constraints) or "llm",
                    )
                )
        else:
            verdict = None
            needs_human = True
            caveats.append(f"LLM 仲裁未形成可用结论：{trace.abstain_reason}")

    return BranchOutcome(
        case_id=result.query_case_id,
        branch=BRANCH,
        verdict=verdict,
        confidence=confidence,
        confidence_lower_bound=confidence_lower_bound,
        calibration_group=group,
        calibration_support=calibration_support,
        evidence_chain=tuple(chain),
        reused_case_ids=tuple(candidate.case_id for candidate in top),
        missing_evidence=missing,
        caveats=tuple(caveats),
        needs_llm=trace is None,
        needs_human=needs_human,
        confidence_breakdown=confidence_breakdown,
        self_reported_confidence=self_reported_confidence,
        history_verdict=history_verdict,
        fallback_source=fallback_source,
        compliance_penalties=compliance_penalties,
    )
