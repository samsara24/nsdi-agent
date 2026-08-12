"""三分支处理器的公共输出结构与置信度标定。

**置信度不是拍脑袋写的常数，是训练集留一法上的实测频率。**

这是本模块最重要的设计决定。常见做法是给 N5a 写 0.9、N5b 写 0.7、N5c 写 0.5，
但这些数字与实际正确率没有任何关系，报告里写出来只会误导运维。
`BranchCalibration.fit` 在训练集上跑一遍留一法，统计每个分支（N5a 再按桶纯净度细分）
实际判对了多少，把这个频率作为置信度。

同时给出 Wilson 95% 置信下界。T4 的分档样本量只有几十条，点估计的抖动很大：
例如 N5a 纯桶在训练集上 12/14 = 85.71%，但 14 个样本的 95% 下界只有 60.06%。
M9 的降级策略应当按下界而不是点估计来卡，否则会被小样本的偶然高分骗过去。
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..types import ROOT_CAUSES, wilson_lower_bound


#: 从 `types` re-export：实现下沉到叶子模块以打破 features -> expert -> branches
#: -> features 的循环，调用点与行为都不变。
wilson_lower_bound = wilson_lower_bound


def majority_label(labels: Sequence[str]) -> Optional[str]:
    """并列打破规则固定为 `ROOT_CAUSES` 顺序取最小。

    任何确定性的规则都可以，但必须固定，否则同一份输入在不同运行里会给出不同答案。
    这条规则从 T1 起全仓库统一，`scripts/` 与 `branches/` 用的是同一套。
    """
    if not labels:
        return None
    vote = Counter(labels)
    top = max(vote.values())
    return min((label for label in vote if vote[label] == top), key=ROOT_CAUSES.index)


@dataclass(frozen=True)
class BranchCalibration:
    """各分支的实测准确率表。键是标定分组名，值是 (判对数, 总数)。"""

    counts: Mapping[str, Tuple[int, int]] = field(default_factory=dict)
    source: str = ""

    @classmethod
    def fit(
        cls,
        groups: Sequence[str],
        correct_flags: Sequence[bool],
        *,
        source: str = "train-loo",
    ) -> "BranchCalibration":
        if len(groups) != len(correct_flags):
            raise ValueError("groups and correct_flags must be the same length")
        tally: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
        for group, correct in zip(groups, correct_flags):
            tally[group][0] += int(bool(correct))
            tally[group][1] += 1
        return cls(counts={key: (value[0], value[1]) for key, value in sorted(tally.items())}, source=source)

    def confidence(self, group: str) -> float:
        """点估计。报告里要和 `support` 一起显示，单独看没有意义。"""
        correct, total = self.counts.get(group, (0, 0))
        return round(correct / total, 6) if total else 0.0

    def lower_bound(self, group: str) -> float:
        correct, total = self.counts.get(group, (0, 0))
        return wilson_lower_bound(correct, total)

    def support(self, group: str) -> int:
        return self.counts.get(group, (0, 0))[1]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "groups": {
                key: {
                    "correct": correct,
                    "total": total,
                    "accuracy": round(correct / total, 6) if total else 0.0,
                    "wilson_lower_bound": wilson_lower_bound(correct, total),
                }
                for key, (correct, total) in sorted(self.counts.items())
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BranchCalibration":
        return cls(
            counts={
                key: (item["correct"], item["total"])
                for key, item in value.get("groups", {}).items()
            },
            source=value.get("source", ""),
        )


@dataclass(frozen=True)
class EvidenceLink:
    """证据链的一环。报告直接渲染它，所以每一环都要能独立读懂。"""

    kind: str
    statement: str
    tokens: Tuple[str, ...] = ()
    source: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "statement": self.statement,
            "tokens": list(self.tokens),
            "source": self.source,
        }


@dataclass(frozen=True)
class BranchOutcome:
    """三分支的统一输出。N6 的弃权也用这个结构，只是 `verdict=None`。"""

    case_id: str
    branch: str
    verdict: Optional[str]
    confidence: float
    confidence_lower_bound: float
    calibration_group: str
    calibration_support: int
    evidence_chain: Tuple[EvidenceLink, ...] = ()
    reused_case_ids: Tuple[str, ...] = ()
    missing_evidence: Tuple[str, ...] = ()
    caveats: Tuple[str, ...] = ()
    needs_llm: bool = False
    needs_human: bool = False

    @property
    def is_abstained(self) -> bool:
        return self.verdict is None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "branch": self.branch,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "confidence_lower_bound": self.confidence_lower_bound,
            "calibration_group": self.calibration_group,
            "calibration_support": self.calibration_support,
            "evidence_chain": [item.to_dict() for item in self.evidence_chain],
            "reused_case_ids": list(self.reused_case_ids),
            "missing_evidence": list(self.missing_evidence),
            "caveats": list(self.caveats),
            "needs_llm": self.needs_llm,
            "needs_human": self.needs_human,
        }
