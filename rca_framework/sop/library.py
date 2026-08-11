"""Train-only learned SOP for RCA v2.

The model in this module is deliberately named "learned SOP": it is a shallow
decision tree inferred from training labels, not an expert-authored playbook.
Each leaf carries support and a Wilson lower bound so downstream decisions can
abstain when the learned path is weak or mixed.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..branches.base import majority_label, wilson_lower_bound
from ..features.extractor import CaseFeatures
from ..types import ROOT_CAUSES


LEARNED_SOP_VERSION = "learned-sop-v1"


@dataclass(frozen=True)
class SOPPrediction:
    verdict: Optional[str]
    confidence: float
    confidence_lower_bound: float
    support: int
    leaf_id: str
    path: Tuple[str, ...]
    label_distribution: Tuple[Tuple[str, int], ...]
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict,
            "confidence": self.confidence,
            "confidence_lower_bound": self.confidence_lower_bound,
            "support": self.support,
            "leaf_id": self.leaf_id,
            "path": list(self.path),
            "label_distribution": dict(self.label_distribution),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class SOPNode:
    node_id: str
    samples: int
    label_counts: Tuple[Tuple[str, int], ...]
    prediction: Optional[str]
    token: Optional[str] = None
    present: Optional["SOPNode"] = None
    absent: Optional["SOPNode"] = None

    @property
    def is_leaf(self) -> bool:
        return self.token is None

    def to_dict(self) -> Dict[str, Any]:
        value: Dict[str, Any] = {
            "node_id": self.node_id,
            "samples": self.samples,
            "label_counts": dict(self.label_counts),
            "prediction": self.prediction,
        }
        if self.token is not None:
            value["token"] = self.token
            value["present"] = self.present.to_dict() if self.present is not None else None
            value["absent"] = self.absent.to_dict() if self.absent is not None else None
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SOPNode":
        return cls(
            node_id=str(value["node_id"]),
            samples=int(value.get("samples", 0)),
            label_counts=tuple(sorted((str(k), int(v)) for k, v in value.get("label_counts", {}).items())),
            prediction=value.get("prediction"),
            token=value.get("token"),
            present=cls.from_dict(value["present"]) if value.get("present") else None,
            absent=cls.from_dict(value["absent"]) if value.get("absent") else None,
        )


@dataclass(frozen=True)
class LearnedSOP:
    version: str
    root: SOPNode
    dictionary_version: str
    dictionary_hash: str
    training_case_count: int
    max_depth: int
    min_leaf_size: int
    source: str = ""

    def predict(self, features: CaseFeatures) -> SOPPrediction:
        tokens = set(features.tokens)
        node = self.root
        path: List[str] = []
        while not node.is_leaf and node.token is not None:
            if node.token in tokens:
                path.append(f"present:{node.token}")
                if node.present is None:
                    break
                node = node.present
            else:
                path.append(f"absent:{node.token}")
                if node.absent is None:
                    break
                node = node.absent
        counts = dict(node.label_counts)
        total = sum(counts.values())
        verdict = node.prediction
        correct = counts.get(verdict, 0) if verdict is not None else 0
        confidence = round(correct / total, 6) if total else 0.0
        lower = wilson_lower_bound(correct, total)
        if verdict is None:
            reason = "learned SOP leaf has no majority label"
        else:
            reason = f"learned SOP leaf predicts {verdict} from train distribution {counts}"
        return SOPPrediction(
            verdict=verdict,
            confidence=confidence,
            confidence_lower_bound=lower,
            support=total,
            leaf_id=node.node_id,
            path=tuple(path),
            label_distribution=tuple(sorted(counts.items())),
            reason=reason,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "dictionary_version": self.dictionary_version,
            "dictionary_hash": self.dictionary_hash,
            "training_case_count": self.training_case_count,
            "max_depth": self.max_depth,
            "min_leaf_size": self.min_leaf_size,
            "source": self.source,
            "content_hash": self.content_hash(),
            "root": self.root.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LearnedSOP":
        return cls(
            version=str(value.get("version", LEARNED_SOP_VERSION)),
            root=SOPNode.from_dict(value["root"]),
            dictionary_version=str(value.get("dictionary_version", "")),
            dictionary_hash=str(value.get("dictionary_hash", "")),
            training_case_count=int(value.get("training_case_count", 0)),
            max_depth=int(value.get("max_depth", 0)),
            min_leaf_size=int(value.get("min_leaf_size", 1)),
            source=str(value.get("source", "")),
        )

    def content_hash(self) -> str:
        payload = {
            "version": self.version,
            "dictionary_version": self.dictionary_version,
            "dictionary_hash": self.dictionary_hash,
            "training_case_count": self.training_case_count,
            "max_depth": self.max_depth,
            "min_leaf_size": self.min_leaf_size,
            "source": self.source,
            "root": self.root.to_dict(),
        }
        return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def learn_sop(
    features: Sequence[CaseFeatures],
    labels: Sequence[str],
    *,
    max_depth: int = 3,
    min_leaf_size: int = 5,
    source: str = "train-only",
) -> LearnedSOP:
    if len(features) != len(labels):
        raise ValueError("features and labels must be the same length")
    unknown = sorted({label for label in labels if label not in ROOT_CAUSES})
    if unknown:
        raise ValueError(f"unsupported labels for learned SOP: {unknown}")
    rows = [(feature, label) for feature, label in zip(features, labels)]
    dictionary_version = features[0].dictionary_version if features else ""
    dictionary_hash = features[0].dictionary_hash if features else ""
    root = _build_node(rows, depth=0, max_depth=max_depth, min_leaf_size=min_leaf_size, node_id="root")
    return LearnedSOP(
        version=LEARNED_SOP_VERSION,
        root=root,
        dictionary_version=dictionary_version,
        dictionary_hash=dictionary_hash,
        training_case_count=len(features),
        max_depth=max_depth,
        min_leaf_size=min_leaf_size,
        source=source,
    )


def _build_node(
    rows: Sequence[Tuple[CaseFeatures, str]],
    *,
    depth: int,
    max_depth: int,
    min_leaf_size: int,
    node_id: str,
) -> SOPNode:
    counts = Counter(label for _, label in rows)
    label_counts = tuple((label, counts.get(label, 0)) for label in ROOT_CAUSES)
    prediction = majority_label([label for _, label in rows])
    if depth >= max_depth or len(rows) < min_leaf_size * 2 or len([v for v in counts.values() if v]) <= 1:
        return SOPNode(node_id=node_id, samples=len(rows), label_counts=label_counts, prediction=prediction)

    token = _best_split(rows, min_leaf_size=min_leaf_size)
    if token is None:
        return SOPNode(node_id=node_id, samples=len(rows), label_counts=label_counts, prediction=prediction)

    present_rows = [(feature, label) for feature, label in rows if token in feature.tokens]
    absent_rows = [(feature, label) for feature, label in rows if token not in feature.tokens]
    return SOPNode(
        node_id=node_id,
        samples=len(rows),
        label_counts=label_counts,
        prediction=prediction,
        token=token,
        present=_build_node(
            present_rows,
            depth=depth + 1,
            max_depth=max_depth,
            min_leaf_size=min_leaf_size,
            node_id=f"{node_id}.present",
        ),
        absent=_build_node(
            absent_rows,
            depth=depth + 1,
            max_depth=max_depth,
            min_leaf_size=min_leaf_size,
            node_id=f"{node_id}.absent",
        ),
    )


def _best_split(rows: Sequence[Tuple[CaseFeatures, str]], *, min_leaf_size: int) -> Optional[str]:
    tokens = sorted({token for feature, _ in rows for token in feature.tokens})
    if not tokens:
        return None
    parent_entropy = _entropy(label for _, label in rows)
    best_token: Optional[str] = None
    best_gain = 0.0
    for token in tokens:
        present = [label for feature, label in rows if token in feature.tokens]
        absent = [label for feature, label in rows if token not in feature.tokens]
        if len(present) < min_leaf_size or len(absent) < min_leaf_size:
            continue
        weighted = (len(present) / len(rows)) * _entropy(present) + (len(absent) / len(rows)) * _entropy(absent)
        gain = parent_entropy - weighted
        if gain > best_gain:
            best_gain = gain
            best_token = token
    return best_token


def _entropy(labels: Iterable[str]) -> float:
    counts = Counter(labels)
    total = sum(counts.values())
    if not total:
        return 0.0
    value = 0.0
    for count in counts.values():
        probability = count / total
        value -= probability * math.log2(probability)
    return value
