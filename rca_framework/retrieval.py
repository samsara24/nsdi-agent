from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

from .types import CaseEvidence


def retrieve(
    train_index: Sequence[CaseEvidence],
    idf: Mapping[str, float],
    query: CaseEvidence,
    top_k: int,
    *,
    hide_labels: bool = False,
) -> List[Dict[str, Any]]:
    """IDF 加权 Jaccard 检索。算法与 legacy `AnomalyKnowledgeGraph.retrieve` 完全一致。

    `hide_labels=True` 时不返回 `root_cause`，用于避免检索结果把训练集标签
    间接泄漏进 prompt：屏蔽了聚合 KG 分数却仍带标签检索，类别先验依然会渗入。
    """
    query_ids = query.anomaly_ids
    rows: List[Dict[str, Any]] = []
    for case in train_index:
        candidate_ids = case.anomaly_ids
        union = query_ids | candidate_ids
        overlap = query_ids & candidate_ids
        # 固定求和顺序，否则集合迭代顺序会让相似度出现浮点级抖动，artifacts 不可复现。
        numerator = sum(idf.get(item, 1.0) for item in sorted(overlap))
        denominator = sum(idf.get(item, 1.0) for item in sorted(union))
        similarity = numerator / denominator if denominator else 0.0
        row = {
            "case_id": case.case_id,
            "root_cause": case.label,
            "similarity": round(similarity, 8),
            "overlap_anomalies": sorted(overlap),
            "supporting_evidence": [item.evidence for item in case.anomalies if item.anomaly_id in overlap][:5],
        }
        if hide_labels:
            row.pop("root_cause")
        rows.append(row)
    return sorted(rows, key=lambda row: (-row["similarity"], row["case_id"]))[:top_k]


def max_similarity(rows: Sequence[Mapping[str, Any]]) -> float:
    return max((float(row.get("similarity", 0.0)) for row in rows), default=0.0)
