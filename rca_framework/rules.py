from __future__ import annotations

import itertools
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Sequence, Tuple

from .types import CaseEvidence, ROOT_CAUSES, normalize_scores, rank_scores


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
        for label in ROOT_CAUSES:
            for rule in self.rule_sets[label]:
                if set(rule.all_of).issubset(query):
                    raw[label] += rule.strength
                    covered.update(rule.all_of)
                    matched[label].append(rule.to_dict())
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
        }

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
