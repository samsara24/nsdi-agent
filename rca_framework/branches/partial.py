"""N5b 部分匹配处理器。

画板对这个分支的定义是「缺非关键证据时补齐，缺关键证据或候选冲突时触发 LLM 仲裁」。
落到实现上，需要回答两个问题：**缺了什么**，以及**缺的算不算关键**。

「缺了什么」由 M3 给出：`missing_evidence` 是并列候选共同要求、而本 case 没有的 token，
取交集是为了只让人去补每个候选都要求的证据，而不是把所有候选的差异列一遍。

「算不算关键」这里用一个可解释的判据：缺失证据如果落在**状态类**家族
（`status:` 与 `drop:` 前缀，即链路通断与光功率丢失），就是关键证据——
它们直接决定链路是否可用，缺了就没法判。落在**分档类**家族（`level:`、`fence:`）
的缺失是程度信息，不影响定性，属于非关键。

这个判据来自约束库的分类而不是拟合：`C6` / `C8`（发光 / 收光全无即该侧失效）
是 invariant 类约束，`C11`（SNR 下界）是 indicator 类。invariant 缺了不能推，
indicator 缺了只是弱一点。
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from ..evidence_graph.match import MatchResult
from ..evidence_graph.router import RoutingDecision
from .base import BranchCalibration, BranchOutcome, EvidenceLink, majority_label


BRANCH = "N5b"

#: 关键证据的 token 前缀。缺这些等于链路通断状态未知，不能仅凭历史相似度下结论。
CRITICAL_PREFIXES: Tuple[str, ...] = ("status:", "drop:")


def critical_missing(missing: Tuple[str, ...]) -> Tuple[str, ...]:
    return tuple(token for token in missing if token.startswith(CRITICAL_PREFIXES))


def calibration_group(result: MatchResult) -> str:
    return "N5b_critical_gap" if critical_missing(result.missing_evidence) else "N5b_minor_gap"


def handle(
    result: MatchResult,
    decision: RoutingDecision,
    calibration: BranchCalibration,
    *,
    trace: Optional[Any] = None,
) -> BranchOutcome:
    top = result.top_candidates
    labels = [candidate.label for candidate in top if candidate.label is not None]
    verdict = majority_label(labels)
    missing = result.missing_evidence
    critical = critical_missing(missing)
    group = calibration_group(result)
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
        caveats.append(
            f"缺失的 {len(critical)} 条证据属于链路通断类，缺了无法确定该侧是否失效，"
            f"结论需要补采后复核：{'、'.join(critical)}"
        )
    if not result.is_label_pure:
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

    arbitration_required = bool(critical) or not result.is_label_pure
    needs_human = False
    if arbitration_required and trace is not None:
        if trace.accepted is not None and trace.accepted.verdict is not None:
            verdict = trace.accepted.verdict
            confidence = trace.accepted.confidence
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
        needs_llm=arbitration_required and trace is None,
        needs_human=needs_human,
    )
