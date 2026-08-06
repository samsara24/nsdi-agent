from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Set, Tuple

from .types import EvidenceItem, ROOT_CAUSES


AGREEMENT_TYPES: Tuple[str, ...] = (
    "independent_agreement",   # 各路结论一致，且所依据的 anomaly 互不相交
    "same_source_agreement",   # 各路结论一致，但依据的是同一批 anomaly
    "conflict",                # 不同路给出不同结论
    "no_evidence",             # 只有先验，没有 case 特异证据
)

# legacy 的"两路"口径：KG_RAG_LLM 路同时使用图路径与 KG feature rule，
# KG_RCA 路使用互斥符号规则。同源判定必须按路聚合，否则同一路内的多条证据
# 会被误判成多路互证。
ROUTE_OF_SOURCE: Dict[str, str] = {
    "kg_path": "kg",
    "kg_feature_rule": "kg",
    "symbolic_rule": "symbolic",
    "anomaly": "anomaly",
    "lane_loss": "lane",
    "retrieval": "retrieval",
    "playbook": "playbook",
}


@dataclass
class EvidenceView:
    """一个 case 的证据视图。

    与 `fusion.fuse_results` 的关键差别：这里保留了每条证据的来源 anomaly，
    因此可以回答"两路是否真的独立"。legacy 融合只比较两路的 prediction，
    而在 `backend=none` 下两路读的是同一批 anomaly，结论一致几乎是必然的，
    把它称作"两条独立推理链结论一致"并不成立。
    """

    per_label: Dict[str, List[EvidenceItem]] = field(default_factory=dict)
    independent_evidence_count: int = 0
    agreement_type: str = "no_evidence"
    shared_anomalies: Tuple[str, ...] = ()
    prior_only: bool = True
    conflict_strength: float = 0.0
    top_label: str = ""
    route_labels: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agreement_type": self.agreement_type,
            "independent_evidence_count": self.independent_evidence_count,
            "shared_anomalies": list(self.shared_anomalies),
            "prior_only": self.prior_only,
            "conflict_strength": round(self.conflict_strength, 8),
            "top_label": self.top_label,
            "route_labels": dict(self.route_labels),
            "evidence_count": {label: len(items) for label, items in self.per_label.items()},
        }


def _independent_groups(items: Sequence[EvidenceItem]) -> List[List[EvidenceItem]]:
    """按共享 anomaly 把证据并成组：只要引用了同一个 anomaly，就不算互相独立。"""
    parent: Dict[str, str] = {}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for item in items:
        for anomaly_id in item.origin_anomalies:
            parent.setdefault(anomaly_id, anomaly_id)
        for anomaly_id in item.origin_anomalies[1:]:
            union(item.origin_anomalies[0], anomaly_id)

    grouped: Dict[str, List[EvidenceItem]] = defaultdict(list)
    for index, item in enumerate(items):
        key = find(item.origin_anomalies[0]) if item.origin_anomalies else f"__standalone_{index}"
        grouped[key].append(item)
    return [grouped[key] for key in sorted(grouped)]


def _route_label(items: Sequence[EvidenceItem]) -> str:
    strength: Dict[str, float] = defaultdict(float)
    for item in items:
        strength[item.supports] += item.strength
    ordering = sorted(strength.items(), key=lambda row: (-row[1], ROOT_CAUSES.index(row[0])))
    return ordering[0][0]


def aggregate_evidence(items: Sequence[EvidenceItem]) -> EvidenceView:
    """聚合证据项，判定一致性类型与独立证据数。

    判定顺序是先看各路结论是否一致，再看支持该结论的证据是否共享 anomaly：

    - 各路结论不同 -> `conflict`。
    - 结论一致，且支持它的证据可以切成两组以上互不相交的 anomaly、路与路之间
      也没有共享 anomaly -> `independent_agreement`。
    - 结论一致但依据同一批 anomaly -> `same_source_agreement`，下游不得把它
      当成两路互证。

    `is_prior_only` 的证据不参与判定：先验不是证据。
    """
    real = [item for item in items if not item.is_prior_only and item.supports in ROOT_CAUSES]
    if not real:
        return EvidenceView(per_label={}, agreement_type="no_evidence", prior_only=True)

    per_label = {
        label: [item for item in real if item.supports == label]
        for label in ROOT_CAUSES
        if any(item.supports == label for item in real)
    }
    by_route: Dict[str, List[EvidenceItem]] = defaultdict(list)
    for item in real:
        by_route[ROUTE_OF_SOURCE.get(item.source, item.source)].append(item)
    route_labels = {route: _route_label(rows) for route, rows in sorted(by_route.items())}

    strength = {label: sum(item.strength for item in rows) for label, rows in per_label.items()}
    ordering = sorted(strength.items(), key=lambda row: (-row[1], ROOT_CAUSES.index(row[0])))
    top_label, top_strength = ordering[0]
    runner_up = ordering[1][1] if len(ordering) > 1 else 0.0

    routes_agree = len(set(route_labels.values())) == 1
    if routes_agree:
        top_label = next(iter(route_labels.values()))

    supporting = per_label.get(top_label, [])
    groups = _independent_groups(supporting)
    anomalies_by_route: Dict[str, Set[str]] = defaultdict(set)
    for item in supporting:
        anomalies_by_route[ROUTE_OF_SOURCE.get(item.source, item.source)].update(item.origin_anomalies)
    shared = set.intersection(*anomalies_by_route.values()) if len(anomalies_by_route) >= 2 else set()

    if not routes_agree:
        agreement_type = "conflict"
    elif len(groups) >= 2 and not shared:
        agreement_type = "independent_agreement"
    else:
        agreement_type = "same_source_agreement"

    return EvidenceView(
        per_label=per_label,
        independent_evidence_count=len(groups),
        agreement_type=agreement_type,
        shared_anomalies=tuple(sorted(shared)),
        prior_only=False,
        conflict_strength=runner_up / top_strength if top_strength else 0.0,
        top_label=top_label,
        route_labels=route_labels,
    )
