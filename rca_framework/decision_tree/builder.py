"""CART-style numeric decision tree for RCA.

The former learned SOP split on token presence.  This tree splits on readable
numeric intervals, preferring expert engineering thresholds before falling back
to train-set quantile candidates.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from ..branches.base import majority_label
from ..expert import EXPERT_THRESHOLDS
from ..types import ROOT_CAUSES, wilson_lower_bound
from .features import NumericFeatureRow


NUMERIC_DECISION_TREE_VERSION = "numeric-decision-tree-v1"


@dataclass(frozen=True)
class NumericTreePrediction:
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
            "model": NUMERIC_DECISION_TREE_VERSION,
        }


@dataclass(frozen=True)
class DecisionTreeNode:
    node_id: str
    samples: int
    label_counts: Tuple[Tuple[str, int], ...]
    prediction: Optional[str]
    feature: Optional[str] = None
    threshold: Optional[float] = None
    threshold_source: str = ""
    leq: Optional["DecisionTreeNode"] = None
    gt: Optional["DecisionTreeNode"] = None
    pruned_reason: str = ""

    @property
    def is_leaf(self) -> bool:
        return self.feature is None

    def to_dict(self) -> Dict[str, Any]:
        value: Dict[str, Any] = {
            "node_id": self.node_id,
            "samples": self.samples,
            "label_counts": dict(self.label_counts),
            "prediction": self.prediction,
            "pruned_reason": self.pruned_reason,
        }
        if not self.is_leaf:
            value.update({
                "feature": self.feature,
                "threshold": self.threshold,
                "threshold_source": self.threshold_source,
                "leq": self.leq.to_dict() if self.leq is not None else None,
                "gt": self.gt.to_dict() if self.gt is not None else None,
            })
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DecisionTreeNode":
        return cls(
            node_id=str(value["node_id"]),
            samples=int(value.get("samples", 0)),
            label_counts=tuple(sorted((str(k), int(v)) for k, v in value.get("label_counts", {}).items())),
            prediction=value.get("prediction"),
            feature=value.get("feature"),
            threshold=float(value["threshold"]) if value.get("threshold") is not None else None,
            threshold_source=str(value.get("threshold_source", "")),
            leq=cls.from_dict(value["leq"]) if value.get("leq") else None,
            gt=cls.from_dict(value["gt"]) if value.get("gt") else None,
            pruned_reason=str(value.get("pruned_reason", "")),
        )


@dataclass(frozen=True)
class NumericDecisionTree:
    version: str
    root: DecisionTreeNode
    training_case_count: int
    max_depth: int
    min_leaf_size: int
    feature_schema: str
    source: str = ""
    dictionary_version: str = "numeric"
    dictionary_hash: str = "numeric"

    def predict(self, row: Any) -> NumericTreePrediction:
        if not isinstance(row, NumericFeatureRow):
            counts = dict(self.root.label_counts)
            total = sum(counts.values())
            verdict = self.root.prediction
            correct = counts.get(verdict, 0) if verdict else 0
            return NumericTreePrediction(
                verdict=verdict,
                confidence=round(correct / total, 6) if total else 0.0,
                confidence_lower_bound=wilson_lower_bound(correct, total),
                support=total,
                leaf_id=self.root.node_id,
                path=("numeric features unavailable; used root distribution",),
                label_distribution=tuple(sorted(counts.items())),
                reason="numeric decision tree received a token-only feature row; used root distribution",
            )
        node = self.root
        path = []
        while not node.is_leaf and node.feature is not None and node.threshold is not None:
            value = row.get(node.feature)
            if value is None:
                path.append(f"missing:{node.feature} -> <= {node.threshold:g} fallback")
                if node.leq is None:
                    break
                node = node.leq
                continue
            if value <= node.threshold:
                path.append(f"{node.feature} <= {node.threshold:g} ({value:g}; {node.threshold_source})")
                if node.leq is None:
                    break
                node = node.leq
            else:
                path.append(f"{node.feature} > {node.threshold:g} ({value:g}; {node.threshold_source})")
                if node.gt is None:
                    break
                node = node.gt
        counts = dict(node.label_counts)
        total = sum(counts.values())
        verdict = node.prediction
        correct = counts.get(verdict, 0) if verdict else 0
        reason = (
            node.pruned_reason
            if node.pruned_reason
            else f"numeric tree leaf predicts {verdict} from train distribution {counts}"
        )
        return NumericTreePrediction(
            verdict=verdict,
            confidence=round(correct / total, 6) if total else 0.0,
            confidence_lower_bound=wilson_lower_bound(correct, total),
            support=total,
            leaf_id=node.node_id,
            path=tuple(path),
            label_distribution=tuple(sorted(counts.items())),
            reason=reason,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "root": self.root.to_dict(),
            "training_case_count": self.training_case_count,
            "max_depth": self.max_depth,
            "min_leaf_size": self.min_leaf_size,
            "feature_schema": self.feature_schema,
            "source": self.source,
            "dictionary_version": self.dictionary_version,
            "dictionary_hash": self.dictionary_hash,
            "content_hash": self.content_hash(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "NumericDecisionTree":
        return cls(
            version=str(value.get("version", NUMERIC_DECISION_TREE_VERSION)),
            root=DecisionTreeNode.from_dict(value["root"]),
            training_case_count=int(value.get("training_case_count", 0)),
            max_depth=int(value.get("max_depth", 0)),
            min_leaf_size=int(value.get("min_leaf_size", 1)),
            feature_schema=str(value.get("feature_schema", "")),
            source=str(value.get("source", "")),
            dictionary_version=str(value.get("dictionary_version", "numeric")),
            dictionary_hash=str(value.get("dictionary_hash", "numeric")),
        )

    def content_hash(self) -> str:
        payload = {
            "version": self.version,
            "root": self.root.to_dict(),
            "training_case_count": self.training_case_count,
            "max_depth": self.max_depth,
            "min_leaf_size": self.min_leaf_size,
            "feature_schema": self.feature_schema,
            "source": self.source,
        }
        return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def fit_numeric_decision_tree(
    rows: Sequence[NumericFeatureRow],
    labels: Sequence[str],
    *,
    max_depth: int = 3,
    min_leaf_size: int = 5,
    source: str = "train-only",
) -> NumericDecisionTree:
    if len(rows) != len(labels):
        raise ValueError("rows and labels must be the same length")
    unknown = sorted({label for label in labels if label not in ROOT_CAUSES})
    if unknown:
        raise ValueError(f"unsupported labels for numeric decision tree: {unknown}")
    paired = tuple((row, label) for row, label in zip(rows, labels))
    root = _build_node(paired, depth=0, max_depth=max_depth, min_leaf_size=min_leaf_size, node_id="root")
    from .pruning import prune_physical_contradictions

    root = prune_physical_contradictions(root)
    return NumericDecisionTree(
        version=NUMERIC_DECISION_TREE_VERSION,
        root=root,
        training_case_count=len(rows),
        max_depth=max_depth,
        min_leaf_size=min_leaf_size,
        feature_schema=rows[0].schema_version if rows else "",
        source=source,
    )


def _build_node(
    rows: Sequence[Tuple[NumericFeatureRow, str]],
    *,
    depth: int,
    max_depth: int,
    min_leaf_size: int,
    node_id: str,
) -> DecisionTreeNode:
    counts = Counter(label for _, label in rows)
    label_counts = tuple((label, counts.get(label, 0)) for label in ROOT_CAUSES)
    prediction = majority_label([label for _, label in rows])
    if depth >= max_depth or len(rows) < min_leaf_size * 2 or len([v for v in counts.values() if v]) <= 1:
        return DecisionTreeNode(node_id=node_id, samples=len(rows), label_counts=label_counts, prediction=prediction)
    split = _best_split(rows, min_leaf_size=min_leaf_size)
    if split is None:
        return DecisionTreeNode(node_id=node_id, samples=len(rows), label_counts=label_counts, prediction=prediction)
    feature, threshold, source = split
    leq_rows = [(row, label) for row, label in rows if (row.get(feature) is None or row.get(feature) <= threshold)]
    gt_rows = [(row, label) for row, label in rows if (row.get(feature) is not None and row.get(feature) > threshold)]
    return DecisionTreeNode(
        node_id=node_id,
        samples=len(rows),
        label_counts=label_counts,
        prediction=prediction,
        feature=feature,
        threshold=threshold,
        threshold_source=source,
        leq=_build_node(
            leq_rows,
            depth=depth + 1,
            max_depth=max_depth,
            min_leaf_size=min_leaf_size,
            node_id=f"{node_id}.leq",
        ),
        gt=_build_node(
            gt_rows,
            depth=depth + 1,
            max_depth=max_depth,
            min_leaf_size=min_leaf_size,
            node_id=f"{node_id}.gt",
        ),
    )


def _best_split(
    rows: Sequence[Tuple[NumericFeatureRow, str]],
    *,
    min_leaf_size: int,
) -> Optional[Tuple[str, float, str]]:
    candidates = _candidate_thresholds(rows)
    parent_entropy = _entropy(label for _, label in rows)
    best: Optional[Tuple[str, float, str]] = None
    best_gain = 0.0
    for feature, thresholds in sorted(candidates.items()):
        for threshold, source in thresholds:
            leq = [label for row, label in rows if (row.get(feature) is None or row.get(feature) <= threshold)]
            gt = [label for row, label in rows if (row.get(feature) is not None and row.get(feature) > threshold)]
            if len(leq) < min_leaf_size or len(gt) < min_leaf_size:
                continue
            weighted = (len(leq) / len(rows)) * _entropy(leq) + (len(gt) / len(rows)) * _entropy(gt)
            gain = parent_entropy - weighted
            # Prefer expert thresholds for equal gains, then deterministic name ordering.
            if gain > best_gain or (math.isclose(gain, best_gain) and best is not None and source == "expert"):
                best_gain = gain
                best = (feature, threshold, source)
    return best


def _candidate_thresholds(
    rows: Sequence[Tuple[NumericFeatureRow, str]],
) -> Dict[str, Tuple[Tuple[float, str], ...]]:
    by_feature: Dict[str, set[float]] = {}
    for row, _ in rows:
        for feature, value in row.values.items():
            by_feature.setdefault(feature, set()).add(float(value))
    out: Dict[str, Tuple[Tuple[float, str], ...]] = {}
    for feature, values in by_feature.items():
        candidates: Dict[float, str] = {}
        for threshold in _expert_thresholds_for_feature(feature):
            if min(values) < threshold < max(values):
                candidates[threshold] = "expert"
        sorted_values = sorted(values)
        for left, right in zip(sorted_values[:-1], sorted_values[1:]):
            if left == right:
                continue
            threshold = (left + right) / 2.0
            candidates.setdefault(threshold, "quantile")
        if candidates:
            out[feature] = tuple(sorted(candidates.items(), key=lambda item: (0 if item[1] == "expert" else 1, item[0])))
    return out


def _expert_thresholds_for_feature(feature: str) -> Tuple[float, ...]:
    parts = feature.split(".")
    if len(parts) != 3:
        return ()
    _, metric, statistic = parts
    config = EXPERT_THRESHOLDS.get(metric)
    if not config:
        return ()
    if statistic == "down_count":
        return (0.5,)
    if statistic == "spread" and "diff" in config:
        return (float(config["diff"]),)
    if statistic in ("min", "mean", "max"):
        values = []
        for key in ("down", "low", "high"):
            if key in config:
                values.append(float(config[key]))
        return tuple(values)
    return ()


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


def replace_node(node: DecisionTreeNode, **changes: Any) -> DecisionTreeNode:
    return replace(node, **changes)
