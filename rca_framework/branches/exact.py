"""N5a 完全匹配处理器。

AGENTS.md 对这个分支有一条硬要求：**必须先校验命中的 signature 是否标签纯净，
不允许只报完全匹配数量**。理由在阶段 1 的基线里：legacy 空间下 N5a 有 46 条，
听上去覆盖不错，但其中大量 signature 底下挂着不同标签的历史 case，
「完全匹配」并不意味着「结论唯一」。

因此本处理器把 N5a 拆成两个标定分组：

- `N5a_pure`：命中桶里的历史 case 标签一致，可以直接复用结论。
- `N5a_mixed`：命中桶里标签不一致，说明这套证据在历史上就区分不开。
  仍然给出多数投票结论，但降置信度、加 caveat，并把它标记为需要 LLM 仲裁。

在 v1 特征空间的实测（训练集留一法）：`N5a_pure` 12/14 = 85.71%，
`N5a_mixed` 4/6 = 66.67%。两者差 19 个百分点，证明这个拆分是必要的。
"""

from __future__ import annotations

from collections import Counter
from typing import Any, List, Optional, Tuple

from ..evidence_graph.match import MatchResult
from ..evidence_graph.router import RoutingDecision
from .base import BranchCalibration, BranchOutcome, EvidenceLink, majority_label


BRANCH = "N5a"


def calibration_group(result: MatchResult) -> str:
    return "N5a_pure" if result.is_label_pure else "N5a_mixed"


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
    group = calibration_group(result)
    confidence = calibration.confidence(group)
    confidence_lower_bound = calibration.lower_bound(group)
    calibration_support = calibration.support(group)

    chain = [
        EvidenceLink(
            kind="exact_match",
            statement=(
                f"本 case 的 {len(result.query_tokens)} 条证据与 {len(top)} 条历史 case 完全一致"
            ),
            tokens=result.query_tokens,
            source="evidence_graph",
        )
    ]

    caveats: List[str] = []
    if not result.is_label_pure:
        distribution = Counter(labels)
        breakdown = "、".join(f"{label} {count} 条" for label, count in sorted(distribution.items()))
        caveats.append(
            f"命中桶的历史标签不纯（{breakdown}），这套证据在历史上就区分不开这几个根因，"
            f"结论取多数投票，需要额外证据才能定论"
        )
        chain.append(
            EvidenceLink(
                kind="purity_warning",
                statement=f"命中桶标签分布：{breakdown}",
                source="evidence_graph.purity",
            )
        )
    else:
        chain.append(
            EvidenceLink(
                kind="purity_check",
                statement=f"命中桶的 {len(labels)} 条历史 case 标签一致，均为 {verdict}",
                source="evidence_graph.purity",
            )
        )

    arbitration_required = not result.is_label_pure
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
        missing_evidence=(),
        caveats=tuple(caveats),
        needs_llm=arbitration_required and trace is None,
        needs_human=needs_human,
    )
