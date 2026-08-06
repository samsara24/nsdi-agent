from __future__ import annotations

import itertools
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Sequence, Tuple

from . import retrieval
from .types import Anomaly, CaseEvidence, EvidenceItem, ROOT_CAUSES, normalize_scores, rank_scores


COVERAGE_STATES: Tuple[str, ...] = (
    "covered_pair",       # 命中至少一条双特征 KG 规则，说明该异常组合训练集见过
    "covered_singleton",  # 只命中单特征 KG 规则，只说明命中了一个已知判别特征
    "covered_exemplar",   # 无规则命中，但检索到足够相似的训练 case
    "partial",            # 有原子路径，无规则、无高相似 exemplar
    "uncovered",          # 无路径、无规则，聚合分数退化为类别先验
)

# 声明式门限，尚未标定。`CoverageReport.max_retrieval_similarity` 会逐 case 报出，
# 便于后续用 artifacts 里的分布回头校准这个值。
EXEMPLAR_SIMILARITY_THRESHOLD = 0.5


@dataclass
class GraphEdge:
    source: str
    relation: str
    target: str
    root_cause: str
    anomaly_id: str
    count: int
    root_cause_frequency: float
    precision: float
    lift: float
    weight: float


@dataclass
class GraphPath:
    root_cause: str
    anomaly_id: str
    anomaly_noun: str
    path: List[str]
    score: float
    edge_statistics: Dict[str, Any]


@dataclass
class FeatureRule:
    """An automatically learned, human-readable KG rule."""

    rule_id: str
    root_cause: str
    all_of: Tuple[str, ...]
    class_frequency: float
    precision: float
    lift: float
    support: float
    exclusivity_margin: float
    matched_training_cases: int
    strength: float
    selection: str = "characteristic"

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["all_of"] = list(self.all_of)
        return value

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "FeatureRule":
        copied = dict(value)
        copied["all_of"] = tuple(copied.get("all_of", []))
        return cls(**copied)


@dataclass(frozen=True)
class CoverageReport:
    """当前 case 落在 KG 的哪一档覆盖状态，以及支撑该判断的原始计数。"""

    state: str
    anomaly_count: int
    path_count: int
    matched_rule_count: int
    matched_pair_rule_count: int
    max_retrieval_similarity: float
    prior_only: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _score_composition(prior_floor: float, path_evidence: float, rule_evidence: float) -> Dict[str, float]:
    """`query` 的 raw 分数由三部分构成，这里报出各部分占比。

    `prior_floor` 占比为 1.0 时，`scores` 归一化后恰好等于训练集类别先验，
    即这份"候选分布"不含任何 case 特异信息。
    """
    total = prior_floor + path_evidence + rule_evidence
    if total <= 0:
        return {"prior_floor": 0.0, "path_evidence": 0.0, "rule_evidence": 0.0}
    return {
        "prior_floor": round(prior_floor / total, 8),
        "path_evidence": round(path_evidence / total, 8),
        "rule_evidence": round(rule_evidence / total, 8),
    }


def classify_coverage(
    case: CaseEvidence,
    graph_result: Dict[str, Any],
    retrieval_result: Sequence[Dict[str, Any]] | None = None,
    *,
    exemplar_similarity_threshold: float = EXEMPLAR_SIMILARITY_THRESHOLD,
) -> CoverageReport:
    """把 KG 响应的结构翻译成五档覆盖状态。

    分档只看结构，不看标签，因此可复现。命中一条单特征规则与命中一条双特征规则
    的证据强度差异很大，必须分开，否则下游无法只给 `covered_singleton` 弱结论。
    """
    matched_rules = graph_result.get("matched_feature_rules") or {}
    matched_rule_count = sum(len(items) for items in matched_rules.values())
    pair_rule_count = sum(
        1 for items in matched_rules.values() for rule in items if len(rule.get("all_of", [])) >= 2
    )
    rows = graph_result.get("retrieved_cases") or [] if retrieval_result is None else retrieval_result
    top_similarity = retrieval.max_similarity(rows)
    path_count = int(graph_result.get("path_count", len(graph_result.get("paths") or [])))
    if pair_rule_count > 0:
        state = "covered_pair"
    elif matched_rule_count > 0:
        state = "covered_singleton"
    elif top_similarity >= exemplar_similarity_threshold:
        state = "covered_exemplar"
    elif path_count > 0:
        state = "partial"
    else:
        state = "uncovered"
    return CoverageReport(
        state=state,
        anomaly_count=len(case.anomalies),
        path_count=path_count,
        matched_rule_count=matched_rule_count,
        matched_pair_rule_count=pair_rule_count,
        max_retrieval_similarity=round(top_similarity, 8),
        prior_only=bool(graph_result.get("prior_only", path_count == 0 and matched_rule_count == 0)),
    )


def evidence_items(graph_result: Dict[str, Any]) -> List[EvidenceItem]:
    """把 KG 路径与命中的 feature rule 转成带来源的证据项。

    路径只覆盖 `query` 返回的 top_k_paths，与注入 prompt 的内容保持一致。
    """
    items: List[EvidenceItem] = []
    for path in graph_result.get("paths") or []:
        anomaly_id = str(path.get("anomaly_id", ""))
        items.append(EvidenceItem(
            evidence_id=f"kg_path:{path.get('root_cause')}:{anomaly_id}",
            source="kg_path",
            supports=str(path.get("root_cause")),
            strength=float(path.get("score", 0.0)),
            origin_anomalies=(anomaly_id,),
            detail={"anomaly_noun": path.get("anomaly_noun", ""), "edge_statistics": path.get("edge_statistics", {})},
        ))
    for label, rules in (graph_result.get("matched_feature_rules") or {}).items():
        for rule in rules:
            items.append(EvidenceItem(
                evidence_id=str(rule.get("rule_id", "")),
                source="kg_feature_rule",
                supports=label,
                strength=float(rule.get("strength", 0.0)),
                origin_anomalies=tuple(rule.get("all_of", [])),
                detail={
                    "precision": rule.get("precision"),
                    "lift": rule.get("lift"),
                    "matched_training_cases": rule.get("matched_training_cases"),
                    "selection": rule.get("selection"),
                },
            ))
    if not items:
        items.append(EvidenceItem(
            evidence_id="kg_prior_only",
            source="kg_path",
            supports=str(graph_result.get("prediction", ROOT_CAUSES[0])),
            strength=float(graph_result.get("confidence", 0.0)),
            origin_anomalies=(),
            is_prior_only=True,
            detail={"note": "无路径、无规则命中，KG 分数等于训练集类别先验"},
        ))
    return items


class AnomalyKnowledgeGraph:
    """A feature-centered KG learned from labeled case patterns.

    Besides individual anomaly edges, the graph stores class profiles and
    characteristic feature rules.  These rules are learned from the training
    cases only and are used directly during query scoring.
    """

    def __init__(self) -> None:
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: List[GraphEdge] = []
        self.edge_index: Dict[str, List[GraphEdge]] = defaultdict(list)
        self.train_index: List[CaseEvidence] = []
        self.idf: Dict[str, float] = {}
        self.priors: Dict[str, float] = {label: 1.0 / len(ROOT_CAUSES) for label in ROOT_CAUSES}
        self.feature_profiles: Dict[str, List[Dict[str, Any]]] = {label: [] for label in ROOT_CAUSES}
        self.feature_rules: Dict[str, List[FeatureRule]] = {label: [] for label in ROOT_CAUSES}

    def fit(self, cases: Sequence[CaseEvidence], min_edge_count: int = 1) -> "AnomalyKnowledgeGraph":
        labeled = [case for case in cases if case.label in ROOT_CAUSES]
        if not labeled:
            raise ValueError("knowledge graph requires labeled training cases")
        self.train_index = list(labeled)
        class_counts = Counter(case.label for case in labeled)
        anomaly_counts: Counter[str] = Counter()
        joint_counts: Counter[tuple[str, str]] = Counter()
        exemplar: Dict[str, Anomaly] = {}
        for case in labeled:
            for item in case.anomalies:
                anomaly_counts[item.anomaly_id] += 1
                joint_counts[(case.label, item.anomaly_id)] += 1
                exemplar.setdefault(item.anomaly_id, item)
        total = len(labeled)
        self.priors = {label: class_counts[label] / total for label in ROOT_CAUSES}
        self.nodes = {
            f"root_cause:{label}": {"id": f"root_cause:{label}", "node_type": "RootCause", "noun": f"{label}根因", "label": label}
            for label in ROOT_CAUSES
        }
        for item in exemplar.values():
            self.nodes[f"anomaly:{item.anomaly_id}"] = {
                "id": f"anomaly:{item.anomaly_id}",
                "node_type": item.node_type,
                "noun": item.noun,
                "side": item.side,
                "metric": item.metric,
            }
        self.edges = []
        self.edge_index = defaultdict(list)
        for label in ROOT_CAUSES:
            prior = self.priors[label]
            for anomaly_id, feature_count in anomaly_counts.items():
                count = joint_counts[(label, anomaly_id)]
                if count < min_edge_count:
                    continue
                root_frequency = count / class_counts[label]
                precision = count / feature_count
                lift = precision / prior if prior else 0.0
                weight = root_frequency * precision * max(0.1, math.log1p(lift)) * math.log1p(count)
                item = exemplar[anomaly_id]
                edge = GraphEdge(
                    source=f"root_cause:{label}", relation=item.relation, target=f"anomaly:{anomaly_id}",
                    root_cause=label, anomaly_id=anomaly_id, count=count,
                    root_cause_frequency=round(root_frequency, 8), precision=round(precision, 8),
                    lift=round(lift, 8), weight=round(weight, 8),
                )
                self.edges.append(edge)
                self.edge_index[anomaly_id].append(edge)
        document_frequency = Counter(anomaly_id for case in labeled for anomaly_id in sorted(case.anomaly_ids))
        # 按 anomaly_id 排序落库，使导出的 idf 键序不依赖集合迭代顺序。
        self.idf = {key: math.log((total + 1) / (count + 1)) + 1.0 for key, count in sorted(document_frequency.items())}
        self._fit_feature_rules(labeled, class_counts, anomaly_counts, joint_counts, exemplar, min_edge_count)
        return self

    def _fit_feature_rules(
        self,
        labeled: Sequence[CaseEvidence],
        class_counts: Counter[str],
        feature_counts: Counter[str],
        joint_counts: Counter[tuple[str, str]],
        exemplars: Dict[str, Anomaly],
        min_count: int,
    ) -> None:
        """Summarize per-class features and retain discriminative KG rules."""
        self.feature_profiles = {label: [] for label in ROOT_CAUSES}
        self.feature_rules = {label: [] for label in ROOT_CAUSES}
        total = len(labeled)
        pair_counts: Counter[Tuple[str, ...]] = Counter()
        pair_class_counts: Counter[Tuple[Tuple[str, ...], str]] = Counter()
        for case in labeled:
            for pair in itertools.combinations(sorted(case.anomaly_ids), 2):
                pair_counts[pair] += 1
                pair_class_counts[(pair, case.label)] += 1
        for label in ROOT_CAUSES:
            prior = self.priors.get(label, 0.0)
            profile_rows = []
            for feature, total_count in feature_counts.items():
                hit = joint_counts[(label, feature)]
                if not hit:
                    continue
                class_frequency = hit / class_counts[label]
                precision = hit / total_count
                lift = precision / prior if prior else 0.0
                profile_rows.append({
                    "feature": feature,
                    "node_type": exemplars[feature].node_type,
                    "noun": exemplars[feature].noun,
                    "side": exemplars[feature].side,
                    "metric": exemplars[feature].metric,
                    "matched_training_cases": hit,
                    "class_frequency": round(class_frequency, 8),
                    "precision": round(precision, 8),
                    "lift": round(lift, 8),
                    # Lift alone is unsafe for the minority fiber class: a
                    # common feature can have a large lift simply because
                    # fiber has a small prior.  A usable KG rule must also
                    # have reasonable global precision.
                    "characteristic": bool(
                        hit >= min_count and class_frequency >= 0.15
                        and precision >= 0.35 and lift >= 1.10
                    ),
                })
            self.feature_profiles[label] = sorted(
                profile_rows,
                key=lambda row: (-row["characteristic"], -row["lift"], -row["class_frequency"], row["feature"]),
            )

            candidates: List[FeatureRule] = []
            for row in self.feature_profiles[label]:
                if not row["characteristic"]:
                    continue
                feature = row["feature"]
                discriminative = row["class_frequency"] * math.log1p(max(0.0, row["lift"]))
                candidates.append(FeatureRule(
                    rule_id="",
                    root_cause=label,
                    all_of=(feature,),
                    class_frequency=row["class_frequency"],
                    precision=row["precision"],
                    lift=row["lift"],
                    support=round(row["matched_training_cases"] / total, 8),
                    exclusivity_margin=round(discriminative, 8),
                    matched_training_cases=row["matched_training_cases"],
                    strength=round(discriminative * math.log1p(row["matched_training_cases"]), 8),
                ))
            candidates.sort(key=lambda rule: (-rule.strength, -rule.lift, rule.all_of))
            for pair, total_pair_count in pair_counts.items():
                hit = pair_class_counts[(pair, label)]
                if hit < min_count:
                    continue
                class_frequency = hit / class_counts[label]
                precision = hit / total_pair_count
                lift = precision / prior if prior else 0.0
                if class_frequency < 0.10 or precision < 0.40 or lift < 1.10:
                    continue
                discriminative = class_frequency * precision * math.log1p(max(0.0, lift))
                candidates.append(FeatureRule(
                    rule_id="",
                    root_cause=label,
                    all_of=pair,
                    class_frequency=round(class_frequency, 8),
                    precision=round(precision, 8),
                    lift=round(lift, 8),
                    support=round(hit / total, 8),
                    exclusivity_margin=round(discriminative, 8),
                    matched_training_cases=hit,
                    strength=round(discriminative * math.log1p(hit) * 1.25, 8),
                    selection="characteristic_pair",
                ))
            candidates.sort(key=lambda rule: (-rule.strength, -len(rule.all_of), -rule.lift, rule.all_of))
            self.feature_rules[label] = candidates[:20]
            for index, rule in enumerate(self.feature_rules[label], start=1):
                rule.rule_id = f"KG_RULE_{label}_{index:04d}"

    def query(
        self,
        case: CaseEvidence,
        top_k_paths: int = 12,
        top_k_cases: int = 5,
        *,
        include_retrieval: bool = True,
    ) -> Dict[str, Any]:
        raw_scores = {label: 0.05 * self.priors.get(label, 0.0) for label in ROOT_CAUSES}
        prior_floor_total = sum(raw_scores.values())
        path_evidence_total = 0.0
        paths: List[GraphPath] = []
        anomaly_map = {item.anomaly_id: item for item in case.anomalies}
        for anomaly_id, item in anomaly_map.items():
            for edge in self.edge_index.get(anomaly_id, []):
                path_score = edge.weight * max(0.25, min(3.0, item.severity))
                raw_scores[edge.root_cause] += path_score
                path_evidence_total += path_score
                paths.append(GraphPath(
                    root_cause=edge.root_cause,
                    anomaly_id=anomaly_id,
                    anomaly_noun=item.noun,
                    path=[f"query:{case.case_id}", "EXHIBITS", f"anomaly:{anomaly_id}", "INDICATES", f"root_cause:{edge.root_cause}"],
                    score=round(path_score, 8),
                    edge_statistics={
                        "training_count": edge.count,
                        "precision": edge.precision,
                        "lift": edge.lift,
                        "root_cause_frequency": edge.root_cause_frequency,
                    },
                ))
        matched_feature_rules: Dict[str, List[Dict[str, Any]]] = {label: [] for label in ROOT_CAUSES}
        feature_scores = {label: 0.0 for label in ROOT_CAUSES}
        for label in ROOT_CAUSES:
            for rule in self.feature_rules.get(label, []):
                if set(rule.all_of).issubset(case.anomaly_ids):
                    feature_scores[label] += rule.strength
                    matched_feature_rules[label].append(rule.to_dict())
            # Profile evidence is part of the KG score, with a conservative
            # weight so a single common anomaly cannot dominate the paths.
            raw_scores[label] += 0.12 * feature_scores[label]
        scores = normalize_scores(raw_scores)
        ranking = rank_scores(scores)
        paths.sort(key=lambda item: (-item.score, item.root_cause, item.anomaly_id))
        neighbors = self.retrieve(case, top_k_cases) if include_retrieval else []
        margin = ranking[0][1] - ranking[1][1]
        path_factor = min(1.0, len(paths) / 4.0)
        confidence = (0.55 * ranking[0][1] + 0.45 * margin) * (0.5 + 0.5 * path_factor) * (0.5 + 0.5 * case.coverage)
        rule_evidence_total = 0.12 * sum(feature_scores.values())
        matched_rule_count = sum(len(items) for items in matched_feature_rules.values())
        return {
            "prediction": ranking[0][0],
            "confidence": round(min(1.0, confidence), 8),
            "scores": scores,
            "paths": [asdict(item) for item in paths[:top_k_paths]],
            "retrieved_cases": neighbors,
            "evidence_coverage": round(case.coverage, 8),
            "path_count": len(paths),
            "feature_profile_scores": normalize_scores(feature_scores),
            "matched_feature_rules": matched_feature_rules,
            # 以下为说明性字段，不参与 scores 计算。
            "score_composition": _score_composition(prior_floor_total, path_evidence_total, rule_evidence_total),
            "prior_only": len(paths) == 0 and matched_rule_count == 0,
        }

    def retrieve(self, query: CaseEvidence, top_k: int, *, hide_labels: bool = False) -> List[Dict[str, Any]]:
        return retrieval.retrieve(self.train_index, self.idf, query, top_k, hide_labels=hide_labels)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "label-centered-anomaly-graph-v2",
            "nodes": list(self.nodes.values()),
            "edges": [asdict(edge) for edge in self.edges],
            "train_index": [case.to_dict() for case in self.train_index],
            "idf": self.idf,
            "priors": self.priors,
            "feature_profiles": self.feature_profiles,
            "feature_rules": {label: [rule.to_dict() for rule in rules] for label, rules in self.feature_rules.items()},
            "invariant": "Every edge target is an anomaly node; normal observations and test labels are absent.",
        }

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "AnomalyKnowledgeGraph":
        graph = cls()
        graph.nodes = {item["id"]: item for item in value.get("nodes", [])}
        graph.edges = [GraphEdge(**item) for item in value.get("edges", [])]
        graph.edge_index = defaultdict(list)
        for edge in graph.edges:
            graph.edge_index[edge.anomaly_id].append(edge)
        graph.train_index = [CaseEvidence.from_dict(item) for item in value.get("train_index", [])]
        graph.idf = {key: float(item) for key, item in value.get("idf", {}).items()}
        graph.priors = {key: float(item) for key, item in value.get("priors", {}).items()}
        graph.feature_profiles = {label: list(value.get("feature_profiles", {}).get(label, [])) for label in ROOT_CAUSES}
        graph.feature_rules = {
            label: [FeatureRule.from_dict(item) for item in value.get("feature_rules", {}).get(label, [])]
            for label in ROOT_CAUSES
        }
        return graph
