from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Tuple


ROOT_CAUSES: Tuple[str, ...] = ("L1", "L2", "fiber")
SIDES: Tuple[str, ...] = ("L1", "L2")

EVIDENCE_STATUSES: Tuple[str, ...] = (
    "anomalies_found",
    "all_metrics_normal",
    "partial_telemetry",
    "no_telemetry",
)


@dataclass(frozen=True)
class Anomaly:
    anomaly_id: str
    node_type: str
    noun: str
    relation: str
    side: str = "link"
    metric: str = ""
    severity: float = 1.0
    evidence: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "Anomaly":
        return cls(**value)


@dataclass
class CaseEvidence:
    case_id: str
    label: str
    anomalies: List[Anomaly]
    observed_fields: int
    expected_fields: int
    missing_fields: List[str] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    @property
    def anomaly_ids(self) -> set[str]:
        return {item.anomaly_id for item in self.anomalies}

    @property
    def coverage(self) -> float:
        if not self.expected_fields:
            return 0.0
        return min(1.0, self.observed_fields / self.expected_fields)

    @property
    def evidence_status(self) -> str:
        """把"零异常"从空集合升级为一等结论。

        `anomalies == []` 现在可以有三种完全相反的含义：所有指标都正常、遥测本身
        缺失、或只采到一部分指标。判定只用已有的 `observed_fields` /
        `expected_fields`，不引入任何新阈值。

        实现为派生属性而不是存储字段，因为它完全由现有字段决定：这样
        `to_dict` 不变，`model.json` 的 `label-centered-anomaly-graph-v2` schema
        与既有 artifacts 都不受影响，也不会出现字段与事实不同步。
        """
        if self.observed_fields <= 0:
            return "no_telemetry"
        if self.anomalies:
            return "anomalies_found"
        if self.observed_fields >= self.expected_fields:
            return "all_metrics_normal"
        return "partial_telemetry"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "label": self.label,
            "anomalies": [item.to_dict() for item in self.anomalies],
            "observed_fields": self.observed_fields,
            "expected_fields": self.expected_fields,
            "missing_fields": self.missing_fields,
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "CaseEvidence":
        return cls(
            case_id=value["case_id"],
            label=value.get("label", ""),
            anomalies=[Anomaly.from_dict(item) for item in value.get("anomalies", [])],
            observed_fields=int(value.get("observed_fields", 0)),
            expected_fields=int(value.get("expected_fields", 0)),
            missing_fields=list(value.get("missing_fields", [])),
            summary=dict(value.get("summary", {})),
        )


def normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    clean = {label: max(0.0, float(scores.get(label, 0.0))) for label in ROOT_CAUSES}
    total = sum(clean.values())
    if total <= 0:
        return {label: 1.0 / len(ROOT_CAUSES) for label in ROOT_CAUSES}
    return {label: value / total for label, value in clean.items()}


def rank_scores(scores: Dict[str, float]) -> List[Tuple[str, float]]:
    normalized = normalize_scores(scores)
    return sorted(normalized.items(), key=lambda item: (-item[1], ROOT_CAUSES.index(item[0])))


# 以下为 Agent 化协议类型，只增不改：legacy 路径不读取它们，因此不影响 58/85。

DECISIONS: Tuple[str, ...] = ROOT_CAUSES + ("abstain",)
SUFFICIENCY: Tuple[str, ...] = ("sufficient", "weak", "insufficient")
EVIDENCE_SOURCES: Tuple[str, ...] = (
    "anomaly",
    "lane_loss",
    "kg_path",
    "kg_feature_rule",
    "symbolic_rule",
    "retrieval",
    "playbook",
)
NO_SUPPORT = "none"


@dataclass(frozen=True)
class EvidenceItem:
    """一条带来源的证据。

    `origin_anomalies` 是同源判定的唯一依据：两条证据只要引用同一批 anomaly，
    就不能算作互相独立的确认。`is_prior_only` 标记那些只反映训练集类别先验、
    不含 case 特异信息的"证据"，它们不得参与充分性判定。
    """

    evidence_id: str
    source: str
    supports: str
    strength: float
    origin_anomalies: Tuple[str, ...] = ()
    is_prior_only: bool = False
    detail: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.source not in EVIDENCE_SOURCES:
            raise ValueError(f"unknown evidence source: {self.source}")
        if self.supports not in ROOT_CAUSES + (NO_SUPPORT,):
            raise ValueError(f"evidence must support a root cause or '{NO_SUPPORT}': {self.supports}")

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["origin_anomalies"] = list(self.origin_anomalies)
        return value


@dataclass
class Verdict:
    """诊断出口。与 legacy 的三分类不同，`decision` 允许取 `abstain`。"""

    decision: str
    confidence: float
    sufficiency: str
    supporting: List[EvidenceItem] = field(default_factory=list)
    conflicting: List[EvidenceItem] = field(default_factory=list)
    requested_evidence: List[Dict[str, Any]] = field(default_factory=list)
    abstain_reason: str = ""
    trace_id: str = ""

    def __post_init__(self) -> None:
        if self.decision not in DECISIONS:
            raise ValueError(f"decision must be one of {DECISIONS}: {self.decision}")
        if self.sufficiency not in SUFFICIENCY:
            raise ValueError(f"sufficiency must be one of {SUFFICIENCY}: {self.sufficiency}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "confidence": self.confidence,
            "sufficiency": self.sufficiency,
            "supporting": [item.to_dict() for item in self.supporting],
            "conflicting": [item.to_dict() for item in self.conflicting],
            "requested_evidence": list(self.requested_evidence),
            "abstain_reason": self.abstain_reason,
            "trace_id": self.trace_id,
        }
