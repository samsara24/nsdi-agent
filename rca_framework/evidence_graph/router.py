"""M4 分流路由：把一次检索结果分到 N5a / N5b / N5c / N6。

路由规则是可配置的，因为它现在有两套且哪套进论文还没定（见 `Validation.md` V1）：

- `BOARD_POLICY`：画板定稿的 `sim = 100%` / `70% <= sim < 100%` / `sim < 70%`。
- `COVERAGE_POLICY`：T4 标定得出的替代规则，N5b 的入口条件从「相似度 >= 0.7」
  换成「证据全覆盖」，并把零证据 case 直接送 N6。

T4 的标定结论是：在特征字典 v1 的空间里，相似度一旦低于 1.0，它的具体数值就不再
携带准确率信息（`[0.5,0.7)` 与 `[0.7,0.8)` 与 `[0.8,0.9)` 三档准确率都在类别先验附近
来回摆，两个切分上排序还不一致）。因此 `BOARD_POLICY` 的 0.7 这条线切出来的是噪声。
`COVERAGE_POLICY` 用的「证据全覆盖」是唯一在训练集留一法和留出测试集上都能复现的分档信号。

两套都保留、都有测试锁定当前数字，这样 V1 往哪边拍板都不需要返工。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .match import MatchResult


#: 分支名。N6 不是「第四个分支」，它是三个分支之外的显式弃权出口。
BRANCHES: Tuple[str, ...] = ("N5a", "N5b", "N5c", "N6")


@dataclass(frozen=True)
class RoutingPolicy:
    """一套分流规则。所有阈值都在这里声明，不散落在判断语句里。"""

    name: str
    description: str
    exact_similarity: float = 1.0
    #: N5b 的相似度下界。`None` 表示不用相似度切 N5b。
    partial_similarity: Optional[float] = 0.7
    #: N5b 是否要求「历史上存在一个 case 包含我的全部证据」。
    partial_requires_full_coverage: bool = False
    #: 零证据 case 是否直接进 N6 而不是掉进 N5c。
    abstain_on_empty_evidence: bool = False
    #: 全链路遥测失效（约束 C15）的 case 是否直接进 N6。
    #: 这类 case 会产出十几个 token 看起来证据充分，但它们全部来自同一条失效的采集通道。
    abstain_on_optical_blackout: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "exact_similarity": self.exact_similarity,
            "partial_similarity": self.partial_similarity,
            "partial_requires_full_coverage": self.partial_requires_full_coverage,
            "abstain_on_empty_evidence": self.abstain_on_empty_evidence,
            "abstain_on_optical_blackout": self.abstain_on_optical_blackout,
        }


BOARD_POLICY = RoutingPolicy(
    name="board-100-70",
    description="画板定稿规则：sim=100% / 70%<=sim<100% / sim<70%。在 legacy anomaly_id 空间上定的。",
    exact_similarity=1.0,
    partial_similarity=0.7,
    partial_requires_full_coverage=False,
    abstain_on_empty_evidence=False,
)

COVERAGE_POLICY = RoutingPolicy(
    name="coverage-v2",
    description=(
        "T4 标定规则：sim=1.0 进 N5a；证据全覆盖（历史见过我的全部证据）进 N5b；"
        "零证据与全链路遥测失效直接进 N6；其余进 N5c。不使用 0.7 这条无数据支持的线。"
    ),
    exact_similarity=1.0,
    partial_similarity=None,
    partial_requires_full_coverage=True,
    abstain_on_empty_evidence=True,
    abstain_on_optical_blackout=True,
)

DEFAULT_POLICY = COVERAGE_POLICY

POLICIES: Dict[str, RoutingPolicy] = {
    BOARD_POLICY.name: BOARD_POLICY,
    COVERAGE_POLICY.name: COVERAGE_POLICY,
}


def policy_for(name: str) -> RoutingPolicy:
    if name not in POLICIES:
        raise KeyError(f"unknown routing policy: {name}; available={sorted(POLICIES)}")
    return POLICIES[name]


@dataclass(frozen=True)
class RoutingDecision:
    """一次分流的结果。`reason` 会原样进报告，所以必须是人话。"""

    case_id: str
    branch: str
    reason: str
    policy_name: str
    max_similarity: float
    evidence_coverage: float
    tie_count: int
    missing_evidence: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "branch": self.branch,
            "reason": self.reason,
            "policy": self.policy_name,
            "max_similarity": self.max_similarity,
            "evidence_coverage": self.evidence_coverage,
            "tie_count": self.tie_count,
            "missing_evidence": list(self.missing_evidence),
        }


def route(result: MatchResult, policy: RoutingPolicy = DEFAULT_POLICY) -> RoutingDecision:
    branch, reason = _decide(result, policy)
    return RoutingDecision(
        case_id=result.query_case_id,
        branch=branch,
        reason=reason,
        policy_name=policy.name,
        max_similarity=result.max_similarity,
        evidence_coverage=result.evidence_coverage,
        tie_count=len(result.top_candidates),
        missing_evidence=result.missing_evidence,
    )


def _decide(result: MatchResult, policy: RoutingPolicy) -> Tuple[str, str]:
    if policy.abstain_on_empty_evidence and not result.query_tokens:
        # 零证据有两种成因，处置相同但报告措辞必须不同，否则运维不知道该去补数据还是去现场。
        if result.query_telemetry_status == "no_telemetry":
            return "N6", "没有采到任何遥测数据，既无历史可复用也无证据可推理，转人工介入"
        return "N6", "遥测已采集但未触发任何特征，无可用证据支撑推断，转人工介入"

    if policy.abstain_on_optical_blackout and result.query_optical_blackout:
        return "N6", (
            f"两端收发光功率全部处于断光哨兵而 TxLOS 仍报 Normal（约束 C15），"
            f"这 {len(result.query_tokens)} 条特征全部来自同一条失效的采集通道，"
            f"看似证据充分实则无一有效，转人工现场确认"
        )

    if result.max_similarity >= policy.exact_similarity:
        return "N5a", (
            f"与 {len(result.top_candidates)} 条历史 case 的证据完全一致"
            f"（相似度 {result.max_similarity:.2f}），复用历史结论"
        )

    if policy.partial_requires_full_coverage and result.evidence_coverage >= 1.0:
        return "N5b", (
            f"历史上存在包含本 case 全部证据的 case（相似度 {result.max_similarity:.2f}，"
            f"证据覆盖率 100%），差异在于历史 case 还有 {len(result.missing_evidence)} 条本 case 未观测到的证据"
        )

    if policy.partial_similarity is not None and result.max_similarity >= policy.partial_similarity:
        return "N5b", (
            f"与历史 case 部分匹配（相似度 {result.max_similarity:.2f} >= {policy.partial_similarity:.2f}，"
            f"证据覆盖率 {result.evidence_coverage:.0%}）"
        )

    if result.max_similarity <= 0.0:
        return "N5c", "历史证据图中没有任何证据重叠的 case，走约束与 SOP 的通用排障"
    return "N5c", (
        f"历史匹配度不足（相似度 {result.max_similarity:.2f}，"
        f"证据覆盖率 {result.evidence_coverage:.0%}），存在历史未见的证据，走约束与 SOP 的通用排障"
    )


def route_many(
    results: Sequence[MatchResult],
    policy: RoutingPolicy = DEFAULT_POLICY,
) -> List[RoutingDecision]:
    return [route(result, policy) for result in results]


def routing_summary(decisions: Sequence[RoutingDecision]) -> Dict[str, Any]:
    """分流分布统计。三分支的数量与占比必须可统计，这是 T5 的验收项之一。"""
    counts = Counter(item.branch for item in decisions)
    total = len(decisions)
    return {
        "policy": decisions[0].policy_name if decisions else None,
        "case_count": total,
        "counts": {branch: counts.get(branch, 0) for branch in BRANCHES},
        "ratios": {
            branch: (round(counts.get(branch, 0) / total, 6) if total else 0.0)
            for branch in BRANCHES
        },
        "cases_with_missing_evidence": sum(1 for item in decisions if item.missing_evidence),
    }
