"""Physical-consistency pruning for numeric RCA trees."""

from __future__ import annotations

from dataclasses import replace
from typing import Tuple

from .builder import DecisionTreeNode


Condition = Tuple[str, str, float]


def prune_physical_contradictions(node: DecisionTreeNode) -> DecisionTreeNode:
    """Prune leaves whose statistical verdict violates pure physics.

    The first mechanical rule is C6/P5: if a path has established that either
    side has one or more txpower-down lanes, a leaf may not automatically
    predict `fiber`.
    """

    def walk(current: DecisionTreeNode, path: Tuple[Condition, ...]) -> DecisionTreeNode:
        if current.is_leaf:
            if current.prediction == "fiber" and _path_has_tx_down(path):
                return replace(
                    current,
                    prediction=None,
                    pruned_reason=(
                        "pruned by P5_tx_down_excludes_medium: path contains txpower.down_count > 0, "
                        "so fiber cannot be an automatic verdict"
                    ),
                )
            return current
        if current.feature is None or current.threshold is None:
            return current
        leq_path = path + ((current.feature, "<=", current.threshold),)
        gt_path = path + ((current.feature, ">", current.threshold),)
        return replace(
            current,
            leq=walk(current.leq, leq_path) if current.leq is not None else None,
            gt=walk(current.gt, gt_path) if current.gt is not None else None,
        )

    return walk(node, ())


def _path_has_tx_down(path: Tuple[Condition, ...]) -> bool:
    for feature, op, threshold in path:
        if not feature.endswith(".txpower.down_count"):
            continue
        if op == ">" and threshold < 1.0:
            return True
    return False
