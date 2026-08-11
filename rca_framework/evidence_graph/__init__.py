"""M2 / M3 证据图：历史 case 的索引与 Top-N 检索。

与 legacy `graph.AnomalyKnowledgeGraph` 的关系：检索内核（IDF 加权 Jaccard）算法一致，
但索引建在特征字典 v1 的 token 上而不是 `anomaly_id` 上，并且检索结果多带三样东西——
缺失证据、多余证据、冲突证据——这三样是 N5b「补什么证据」和 N6「够不够判」的直接输入。
legacy 的 `retrieve` 只返回相似度和重叠列表，回答不了这两个问题。
"""

from .match import (
    Candidate,
    MatchResult,
    match,
    match_many,
)
from .router import (
    BOARD_POLICY,
    BRANCHES,
    COVERAGE_POLICY,
    DEFAULT_POLICY,
    POLICIES,
    RoutingDecision,
    RoutingPolicy,
    policy_for,
    route,
    route_many,
    routing_summary,
)
from .store import (
    EVIDENCE_GRAPH_SCHEMA,
    EVIDENCE_GRAPH_V2_SCHEMA,
    CaseDiagnosis,
    DiagnosisEdge,
    DiagnosisNode,
    EvidenceGraph,
    GraphCase,
)

__all__ = [
    "BOARD_POLICY",
    "BRANCHES",
    "COVERAGE_POLICY",
    "DEFAULT_POLICY",
    "EVIDENCE_GRAPH_SCHEMA",
    "EVIDENCE_GRAPH_V2_SCHEMA",
    "POLICIES",
    "Candidate",
    "CaseDiagnosis",
    "DiagnosisEdge",
    "DiagnosisNode",
    "EvidenceGraph",
    "GraphCase",
    "MatchResult",
    "RoutingDecision",
    "RoutingPolicy",
    "match",
    "match_many",
    "policy_for",
    "route",
    "route_many",
    "routing_summary",
]
