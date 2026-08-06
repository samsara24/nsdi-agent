from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Tuple


ROOT_CAUSES: Tuple[str, ...] = ("L1", "L2", "fiber")
SIDES: Tuple[str, ...] = ("L1", "L2")


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
