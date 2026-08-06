from __future__ import annotations

import itertools
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .types import CaseEvidence, EvidenceItem, ROOT_CAUSES, normalize_scores, rank_scores


SUPPORT_TIERS: Dict[str, str] = {
    "strong": "matched_training_cases >= 5 and confidence >= 0.50",
    "moderate": "matched_training_cases >= 3",
    "low_support": "matched_training_cases <= 2 or selection == 'minority_fallback'",
}


def support_tier(rule: Mapping[str, Any]) -> str:
    """把一条规则的训练支持度分档。

    `minority_fallback` 规则一律算低支持：它们是某个类别（实际主要是 `fiber`）
    没有任何达标规则时放宽条件取来的，`matched_training_cases` 可能只有 2，
    但在 `match` 里与强规则同权叠加 `strength`。不分档，决策层就无法区分
    "fiber 有规则支持"和"fiber 有两个样本的巧合支持"。
    """
    matched_cases = int(rule.get("matched_training_cases", 0))
    if rule.get("selection") == "minority_fallback" or matched_cases <= 2:
        return "low_support"
    if matched_cases >= 5 and float(rule.get("confidence", 0.0)) >= 0.50:
        return "strong"
    if matched_cases >= 3:
        return "moderate"
    return "low_support"


def evidence_items(match_result: Mapping[str, Any]) -> List[EvidenceItem]:
    """把 `match` 命中的规则转成带来源的证据项，`origin_anomalies` 即规则前件。"""
    items: List[EvidenceItem] = []
    for label, rules in (match_result.get("matched_rules") or {}).items():
        for rule in rules:
            items.append(EvidenceItem(
                evidence_id=str(rule.get("rule_id", "")),
                source="symbolic_rule",
                supports=label,
                strength=float(rule.get("strength", 0.0)),
                origin_anomalies=tuple(rule.get("all_of", [])),
                detail={
                    "confidence": rule.get("confidence"),
                    "lift": rule.get("lift"),
                    "matched_training_cases": rule.get("matched_training_cases"),
                    "selection": rule.get("selection"),
                    "support_tier": rule.get("support_tier", support_tier(rule)),
                },
            ))
    if not items:
        items.append(EvidenceItem(
            evidence_id="symbolic_prior_only",
            source="symbolic_rule",
            supports=str(match_result.get("prediction", ROOT_CAUSES[0])),
            strength=float(match_result.get("confidence", 0.0)),
            origin_anomalies=(),
            is_prior_only=True,
            detail={"note": "无规则命中，符号分数等于 0.02 倍训练集类别先验"},
        ))
    return items


@dataclass
class SymbolicRule:
    rule_id: str
    root_cause: str
    all_of: Tuple[str, ...]
    confidence: float
    lift: float
    support: float
    exclusivity_margin: float
    matched_training_cases: int
    strength: float
    selection: str = "strict"

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["all_of"] = list(self.all_of)
        return value

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "SymbolicRule":
        copied = dict(value)
        copied["all_of"] = tuple(copied["all_of"])
        return cls(**copied)


class SymbolicRuleEngine:
    """Learns three antecedent-disjoint positive rule sets."""

    def __init__(self) -> None:
        self.rule_sets: Dict[str, List[SymbolicRule]] = {label: [] for label in ROOT_CAUSES}
        self.priors: Dict[str, float] = {label: 1.0 / len(ROOT_CAUSES) for label in ROOT_CAUSES}
        self.training_case_count = 0

    def fit(
        self,
        cases: Sequence[CaseEvidence],
        min_count: int = 2,
        min_confidence: float = 0.35,
        min_lift: float = 1.05,
        min_margin: float = 0.03,
        max_rules_per_class: int = 40,
    ) -> "SymbolicRuleEngine":
        labeled = [case for case in cases if case.label in ROOT_CAUSES]
        if not labeled:
            raise ValueError("symbolic rule learner requires labeled training cases")
        self.training_case_count = len(labeled)
        class_counts = Counter(case.label for case in labeled)
        self.priors = {label: class_counts[label] / len(labeled) for label in ROOT_CAUSES}
        antecedent_total: Counter[Tuple[str, ...]] = Counter()
        antecedent_class: Counter[Tuple[Tuple[str, ...], str]] = Counter()
        for case in labeled:
            ids = sorted(case.anomaly_ids)
            candidates = [(item,) for item in ids]
            candidates.extend(tuple(pair) for pair in itertools.combinations(ids[:24], 2))
            for antecedent in candidates:
                antecedent_total[antecedent] += 1
                antecedent_class[(antecedent, case.label)] += 1

        owned: Dict[str, List[SymbolicRule]] = defaultdict(list)
        relaxed: Dict[str, List[SymbolicRule]] = defaultdict(list)
        for antecedent, total_count in antecedent_total.items():
            if total_count < min_count:
                continue
            stats = []
            for label in ROOT_CAUSES:
                hit = antecedent_class[(antecedent, label)]
                confidence = hit / total_count
                lift = confidence / self.priors[label] if self.priors[label] else 0.0
                discriminative = confidence * math.log1p(max(0.0, lift))
                stats.append((label, discriminative, confidence, lift, hit))
            stats.sort(key=lambda item: (-item[1], ROOT_CAUSES.index(item[0])))
            best, second = stats[0], stats[1]
            label, discriminative, confidence, lift, hit = best
            margin = discriminative - second[1]
            support = hit / len(labeled)
            strength = discriminative * math.log1p(hit) * (1.15 if len(antecedent) == 2 else 1.0)
            rule = SymbolicRule(
                rule_id=f"RULE_{label}_{len(owned[label]) + len(relaxed[label]) + 1:04d}",
                root_cause=label,
                all_of=antecedent,
                confidence=round(confidence, 8),
                lift=round(lift, 8),
                support=round(support, 8),
                exclusivity_margin=round(margin, 8),
                matched_training_cases=hit,
                strength=round(strength, 8),
                selection="strict",
            )
            relaxed[label].append(rule)
            if hit >= min_count and confidence >= min_confidence and lift >= min_lift and margin >= min_margin:
                owned[label].append(rule)

        self.rule_sets = {label: [] for label in ROOT_CAUSES}
        claimed: set[Tuple[str, ...]] = set()
        for label in ROOT_CAUSES:
            candidates = sorted(owned[label], key=lambda rule: (-rule.strength, -len(rule.all_of), rule.all_of))
            if not candidates:
                candidates = sorted(relaxed[label], key=lambda rule: (-rule.strength, -len(rule.all_of), rule.all_of))[:5]
                for rule in candidates:
                    rule.selection = "minority_fallback"
            for rule in candidates:
                if rule.all_of in claimed:
                    continue
                self.rule_sets[label].append(rule)
                claimed.add(rule.all_of)
                if len(self.rule_sets[label]) >= max_rules_per_class:
                    break
        self._renumber()
        return self

    def _renumber(self) -> None:
        for label in ROOT_CAUSES:
            for index, rule in enumerate(self.rule_sets[label], start=1):
                rule.rule_id = f"RULE_{label}_{index:04d}"

    def match(self, case: CaseEvidence) -> Dict[str, Any]:
        query = case.anomaly_ids
        raw = {label: 0.02 * self.priors.get(label, 0.0) for label in ROOT_CAUSES}
        matched: Dict[str, List[Dict[str, Any]]] = {label: [] for label in ROOT_CAUSES}
        covered: set[str] = set()
        tier_counts: Counter[str] = Counter()
        for label in ROOT_CAUSES:
            for rule in self.rule_sets[label]:
                if set(rule.all_of).issubset(query):
                    raw[label] += rule.strength
                    covered.update(rule.all_of)
                    row = rule.to_dict()
                    # 只在 match 输出上附加分档，不写进 `to_dict` 的模型 schema。
                    row["support_tier"] = support_tier(row)
                    tier_counts[row["support_tier"]] += 1
                    matched[label].append(row)
        scores = normalize_scores(raw)
        ranking = rank_scores(scores)
        matched_count = sum(len(items) for items in matched.values())
        rule_coverage = len(covered) / len(query) if query else 0.0
        margin = ranking[0][1] - ranking[1][1]
        confidence = (0.5 * ranking[0][1] + 0.5 * margin) * min(1.0, matched_count / 2.0) * (0.5 + 0.5 * rule_coverage)
        return {
            "prediction": ranking[0][0],
            "confidence": round(confidence, 8),
            "scores": scores,
            "matched_rules": matched,
            "matched_rule_count": matched_count,
            "rule_coverage": round(rule_coverage, 8),
            # 以下为说明性字段，不参与 scores 计算。
            "support_tier_counts": {tier: tier_counts[tier] for tier in SUPPORT_TIERS},
        }

    def support_audit(self) -> Dict[str, Any]:
        """报出每个类别的规则里有多少条只有弱训练支持。

        回答的是"`fiber` 的规则中有多少来自 `minority_fallback`"这类问题：
        规则条数相同不代表证据强度相同。
        """
        report: Dict[str, Any] = {}
        for label in ROOT_CAUSES:
            rows = [rule.to_dict() for rule in self.rule_sets[label]]
            tiers = Counter(support_tier(row) for row in rows)
            selections = Counter(str(row.get("selection", "")) for row in rows)
            report[label] = {
                "rule_count": len(rows),
                "support_tier": {tier: tiers[tier] for tier in SUPPORT_TIERS},
                "selection": dict(sorted(selections.items())),
            }
        return report

    def overlap_audit(self) -> Dict[str, Any]:
        sets = {label: {rule.all_of for rule in rules} for label, rules in self.rule_sets.items()}
        pairwise = {}
        for left, right in itertools.combinations(ROOT_CAUSES, 2):
            pairwise[f"{left}__{right}"] = [list(item) for item in sorted(sets[left] & sets[right])]
        return {
            "pairwise_overlap": pairwise,
            "total_overlap_count": sum(len(items) for items in pairwise.values()),
            "invariant": "An antecedent is owned by at most one root-cause rule set.",
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "exclusive-symbolic-rules-v2",
            "training_case_count": self.training_case_count,
            "priors": self.priors,
            "rule_sets": {label: [rule.to_dict() for rule in rules] for label, rules in self.rule_sets.items()},
            "overlap_audit": self.overlap_audit(),
        }

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "SymbolicRuleEngine":
        engine = cls()
        engine.training_case_count = int(value.get("training_case_count", 0))
        engine.priors = {key: float(item) for key, item in value.get("priors", {}).items()}
        engine.rule_sets = {
            label: [SymbolicRule.from_dict(item) for item in value.get("rule_sets", {}).get(label, [])]
            for label in ROOT_CAUSES
        }
        return engine
