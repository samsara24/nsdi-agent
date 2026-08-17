"""M3 Top-N 检索：相似度、覆盖率、缺失证据、多余证据、冲突证据。

相似度内核与 legacy `retrieval.retrieve` 一致（IDF 加权 Jaccard），差别在返回结构。
legacy 只返回相似度和重叠 token，这回答不了 N5b 的核心问题「还缺什么证据」，
也回答不了 N6 的「够不够判」。这里把差集拆成三类：

- `missing_evidence`：候选有、当前 case 没有的 token。它就是补采清单。
- `extra_evidence`：当前 case 有、候选没有的 token。它说明当前 case 比历史更严重或场景不同。
- `conflicting_evidence`：同一维度上两边取了互斥分档的 token 对。
  这是最强的信号——不是「少了点证据」，而是「两边直接矛盾」，不应按相似 case 处理。

一个空 signature 与另一个空 signature 的相似度定义为 0 而不是 1。
否则所有零证据 case 会互相 100% 命中，N5a 会被一群「什么都没测到」的 case 填满，
而它们恰恰是最该走人工介入的那批（阶段 1 结论 2）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from ..features.extractor import MUTUALLY_EXCLUSIVE_PREFIXES, CaseFeatures
from .store import CaseDiagnosis, EvidenceGraph, GraphCase


@dataclass(frozen=True)
class Candidate:
    """一条 Top-N 候选。`label` 在 `hide_labels=True` 时为 None。"""

    case_id: str
    similarity: float
    shared_evidence: Tuple[str, ...]
    missing_evidence: Tuple[str, ...]
    extra_evidence: Tuple[str, ...]
    conflicting_evidence: Tuple[Tuple[str, str], ...]
    evidence_chain_summary: Tuple[str, ...] = ()
    missing_chain_steps: Tuple[str, ...] = ()
    label: Optional[str] = None

    @property
    def has_conflict(self) -> bool:
        return bool(self.conflicting_evidence)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "label": self.label,
            "similarity": self.similarity,
            "shared_evidence": list(self.shared_evidence),
            "missing_evidence": list(self.missing_evidence),
            "extra_evidence": list(self.extra_evidence),
            "conflicting_evidence": [list(pair) for pair in self.conflicting_evidence],
            "evidence_chain_summary": list(self.evidence_chain_summary),
            "missing_chain_steps": list(self.missing_chain_steps),
        }


@dataclass
class MatchResult:
    """一次 N3 检索的完整产出，也是 N4 路由的唯一输入。"""

    query_case_id: str
    query_tokens: Tuple[str, ...]
    query_telemetry_status: str
    candidates: Tuple[Candidate, ...] = ()
    graph_version: str = ""
    query_optical_blackout: bool = False

    @property
    def max_similarity(self) -> float:
        return max((item.similarity for item in self.candidates), default=0.0)

    @property
    def top_candidates(self) -> Tuple[Candidate, ...]:
        """与最高相似度并列的全部候选。N5a 的「桶」就是这个集合。"""
        best = self.max_similarity
        if best <= 0.0:
            return ()
        return tuple(item for item in self.candidates if item.similarity == best)

    @property
    def evidence_coverage(self) -> float:
        """当前 case 的证据被最佳候选覆盖了多少。

        与相似度的区别：相似度是对称的，覆盖率是单向的。
        覆盖率低说明当前 case 有一堆历史上没见过的证据，即使相似度还行也要当心。
        """
        top = self.top_candidates
        if not top or not self.query_tokens:
            return 0.0
        shared = max(len(item.shared_evidence) for item in top)
        return round(shared / len(self.query_tokens), 6)

    @property
    def missing_evidence(self) -> Tuple[str, ...]:
        """并列候选共同要求、而当前 case 没有的证据。这是补采清单。"""
        top = self.top_candidates
        if not top:
            return ()
        common: Optional[Set[str]] = None
        for item in top:
            current = set(item.missing_evidence)
            common = current if common is None else (common & current)
        return tuple(sorted(common or set()))

    @property
    def has_conflict(self) -> bool:
        return any(item.has_conflict for item in self.top_candidates)

    @property
    def tie_labels(self) -> Tuple[str, ...]:
        return tuple(item.label for item in self.top_candidates if item.label is not None)

    @property
    def is_label_pure(self) -> bool:
        labels = set(self.tie_labels)
        return len(labels) == 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_case_id": self.query_case_id,
            "query_tokens": list(self.query_tokens),
            "query_telemetry_status": self.query_telemetry_status,
            "query_optical_blackout": self.query_optical_blackout,
            "graph_version": self.graph_version,
            "max_similarity": self.max_similarity,
            "evidence_coverage": self.evidence_coverage,
            "tie_count": len(self.top_candidates),
            "missing_evidence": list(self.missing_evidence),
            "has_conflict": self.has_conflict,
            "candidates": [item.to_dict() for item in self.candidates],
        }


def weighted_jaccard(query: Set[str], candidate: Set[str], idf: Mapping[str, float]) -> float:
    union = query | candidate
    if not union:
        return 0.0
    overlap = query & candidate
    # 固定求和顺序，理由同 store._idf。
    numerator = sum(idf.get(token, 1.0) for token in sorted(overlap))
    denominator = sum(idf.get(token, 1.0) for token in sorted(union))
    return round(numerator / denominator, 8) if denominator else 0.0


def find_conflicts(query: Set[str], candidate: Set[str]) -> Tuple[Tuple[str, str], ...]:
    """找出两个 signature 在同一维度上取了互斥分档的 token 对。"""

    def by_dimension(tokens: Set[str]) -> Dict[str, str]:
        return {
            token.rsplit(":", 1)[0]: token
            for token in sorted(tokens)
            if token.startswith(MUTUALLY_EXCLUSIVE_PREFIXES)
        }

    left, right = by_dimension(query), by_dimension(candidate)
    return tuple(
        (left[dimension], right[dimension])
        for dimension in sorted(set(left) & set(right))
        if left[dimension] != right[dimension]
    )


def match(
    graph: EvidenceGraph,
    features: CaseFeatures,
    *,
    top_k: int = 5,
    hide_labels: bool = False,
    exclude_case_ids: Sequence[str] = (),
) -> MatchResult:
    """在证据图里检索与当前 case 最相似的历史 case。

    `exclude_case_ids` 用于留一法评估：把 query 自身排除，否则相似度恒为 1.0。
    """
    query = set(features.tokens)
    excluded = set(exclude_case_ids)
    rows: List[Candidate] = []
    for case in graph.cases:
        if case.case_id in excluded:
            continue
        rows.append(
            _build_candidate(
                case,
                query,
                graph.idf,
                diagnosis=graph.diagnosis_for(case.case_id),
                hide_labels=hide_labels,
            )
        )

    rows.sort(key=lambda item: (-item.similarity, item.case_id))
    return MatchResult(
        query_case_id=features.case_id,
        query_tokens=features.tokens,
        query_telemetry_status=features.telemetry_status,
        candidates=tuple(rows[:top_k]) if top_k > 0 else tuple(rows),
        graph_version=graph.version,
        query_optical_blackout=features.optical_blackout,
    )


def match_many(
    graph: EvidenceGraph,
    features: Sequence[CaseFeatures],
    *,
    top_k: int = 5,
    hide_labels: bool = False,
    leave_one_out: bool = False,
) -> List[MatchResult]:
    """批量检索。`leave_one_out=True` 时自动把每个 query 自身排除。"""
    return [
        match(
            graph,
            item,
            top_k=top_k,
            hide_labels=hide_labels,
            exclude_case_ids=(item.case_id,) if leave_one_out else (),
        )
        for item in features
    ]


def _build_candidate(
    case: GraphCase,
    query: Set[str],
    idf: Mapping[str, float],
    *,
    diagnosis: Optional[CaseDiagnosis] = None,
    hide_labels: bool,
) -> Candidate:
    candidate_tokens = set(case.tokens)
    chain_summary, missing_steps = _diagnosis_chain_summary(diagnosis, query)
    return Candidate(
        case_id=case.case_id,
        label=None if hide_labels else case.label,
        similarity=weighted_jaccard(query, candidate_tokens, idf),
        shared_evidence=tuple(sorted(query & candidate_tokens)),
        missing_evidence=tuple(sorted(candidate_tokens - query)),
        extra_evidence=tuple(sorted(query - candidate_tokens)),
        conflicting_evidence=find_conflicts(query, candidate_tokens),
        evidence_chain_summary=chain_summary,
        missing_chain_steps=missing_steps,
    )


def _diagnosis_chain_summary(
    diagnosis: Optional[CaseDiagnosis],
    query: Set[str],
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    if diagnosis is None:
        return (), ()
    token_nodes = {
        node.node_id: str(node.attrs.get("token", ""))
        for node in diagnosis.nodes
        if node.node_type == "FeatureToken"
    }
    step_nodes = [
        node for node in diagnosis.nodes
        if node.node_type in {"SOPStep", "ConstraintCheck"}
    ]
    summary: List[str] = []
    missing: List[str] = []
    for node in sorted(step_nodes, key=lambda item: item.node_id):
        attrs = node.attrs
        title = str(attrs.get("statement") or attrs.get("title") or attrs.get("constraint_id") or node.node_id)
        tokens = tuple(str(token) for token in attrs.get("tokens", ()) if token)
        if not tokens:
            linked_tokens = tuple(
                token_nodes.get(edge.dst, "")
                for edge in diagnosis.edges
                if edge.src == node.node_id and edge.edge_type == "uses_token"
            )
            tokens = tuple(token for token in linked_tokens if token)
        summary.append(title)
        if tokens and not set(tokens).issubset(query):
            missing.append(title)
    return tuple(summary[:8]), tuple(missing[:8])
