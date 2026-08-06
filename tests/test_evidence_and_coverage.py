"""阶段 1 观测层单测：协议类型、覆盖分档、同源判定、支持度分级、检索拆分。

这些测试只针对新增的观测能力。legacy 数值是否漂移由 `test_baseline_lock.py` 负责。
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from rca_framework import evidence as evidence_module
from rca_framework import graph as graph_module
from rca_framework import retrieval
from rca_framework import rules as rules_module
from rca_framework.types import (
    Anomaly,
    CaseEvidence,
    DECISIONS,
    EvidenceItem,
    ROOT_CAUSES,
    SUFFICIENCY,
    Verdict,
)


def make_anomaly(anomaly_id: str, side: str = "L1") -> Anomaly:
    return Anomaly(
        anomaly_id=anomaly_id, node_type="signal_drop", noun=anomaly_id, relation="INDICATES",
        side=side, metric="rxpower", severity=1.0, evidence=anomaly_id,
    )


def make_case(case_id: str, label: str, anomaly_ids: List[str]) -> CaseEvidence:
    return CaseEvidence(case_id, label, [make_anomaly(item) for item in anomaly_ids], 10, 10)


def kg_path_item(anomaly_id: str, supports: str, strength: float = 1.0) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=f"kg_path:{supports}:{anomaly_id}", source="kg_path", supports=supports,
        strength=strength, origin_anomalies=(anomaly_id,),
    )


def symbolic_item(anomaly_ids: tuple[str, ...], supports: str, strength: float = 1.0) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=f"RULE_{supports}", source="symbolic_rule", supports=supports,
        strength=strength, origin_anomalies=anomaly_ids,
    )


def graph_result(
    *,
    paths: List[Dict[str, Any]] | None = None,
    matched: Dict[str, List[Dict[str, Any]]] | None = None,
    similarity: float = 0.0,
) -> Dict[str, Any]:
    rows = paths or []
    matched_rules = matched or {label: [] for label in ROOT_CAUSES}
    matched_count = sum(len(items) for items in matched_rules.values())
    return {
        "prediction": "L2",
        "confidence": 0.1,
        "paths": rows,
        "path_count": len(rows),
        "matched_feature_rules": matched_rules,
        "retrieved_cases": [{"case_id": "train-1", "similarity": similarity}],
        "prior_only": not rows and not matched_count,
    }


def test_decisions_extend_root_causes_with_abstain() -> None:
    assert DECISIONS[: len(ROOT_CAUSES)] == ROOT_CAUSES
    assert DECISIONS[len(ROOT_CAUSES):] == ("abstain",)
    assert "abstain" not in ROOT_CAUSES
    assert SUFFICIENCY == ("sufficient", "weak", "insufficient")


def test_evidence_item_rejects_unknown_source_and_support() -> None:
    with pytest.raises(ValueError):
        EvidenceItem(evidence_id="x", source="telepathy", supports="L1", strength=1.0)
    with pytest.raises(ValueError):
        EvidenceItem(evidence_id="x", source="kg_path", supports="L3", strength=1.0)


def test_verdict_allows_abstain_but_rejects_unknown_decision() -> None:
    verdict = Verdict(decision="abstain", confidence=0.0, sufficiency="insufficient", abstain_reason="no telemetry")
    assert verdict.to_dict()["decision"] == "abstain"
    with pytest.raises(ValueError):
        Verdict(decision="unknown", confidence=0.0, sufficiency="insufficient")


def test_same_source_agreement_is_not_counted_as_two_routes() -> None:
    view = evidence_module.aggregate_evidence([
        kg_path_item("a", "L1"),
        symbolic_item(("a",), "L1"),
    ])
    assert view.agreement_type == "same_source_agreement"
    assert view.independent_evidence_count == 1
    assert view.shared_anomalies == ("a",)
    assert view.prior_only is False


def test_disjoint_anomalies_count_as_independent_agreement() -> None:
    view = evidence_module.aggregate_evidence([
        kg_path_item("a", "L1"),
        symbolic_item(("b",), "L1"),
    ])
    assert view.agreement_type == "independent_agreement"
    assert view.independent_evidence_count == 2
    assert view.shared_anomalies == ()


def test_route_disagreement_is_conflict() -> None:
    view = evidence_module.aggregate_evidence([
        kg_path_item("a", "L1", strength=1.0),
        symbolic_item(("b",), "L2", strength=0.5),
    ])
    assert view.agreement_type == "conflict"
    assert view.route_labels == {"kg": "L1", "symbolic": "L2"}
    assert view.conflict_strength == pytest.approx(0.5)


def test_prior_only_items_do_not_count_as_evidence() -> None:
    view = evidence_module.aggregate_evidence([
        EvidenceItem(evidence_id="kg_prior_only", source="kg_path", supports="L2", strength=0.3, is_prior_only=True),
        EvidenceItem(
            evidence_id="symbolic_prior_only", source="symbolic_rule", supports="L2",
            strength=0.0, is_prior_only=True,
        ),
    ])
    assert view.agreement_type == "no_evidence"
    assert view.prior_only is True
    assert view.independent_evidence_count == 0


def test_chained_shared_anomaly_collapses_into_one_group() -> None:
    """(a) 与 (a,b) 共享 a，(b,c) 又共享 b，三条证据同源，不是三路互证。"""
    view = evidence_module.aggregate_evidence([
        kg_path_item("a", "L1"),
        symbolic_item(("a", "b"), "L1"),
        symbolic_item(("b", "c"), "L1"),
    ])
    assert view.independent_evidence_count == 1
    assert view.agreement_type == "same_source_agreement"


@pytest.mark.parametrize(
    ("matched_training_cases", "confidence", "selection", "expected"),
    [
        (5, 0.50, "strict", "strong"),
        (9, 0.90, "strict", "strong"),
        (5, 0.49, "strict", "moderate"),
        (3, 0.90, "strict", "moderate"),
        (2, 1.00, "strict", "low_support"),
        (10, 0.90, "minority_fallback", "low_support"),
    ],
)
def test_support_tier_boundaries(
    matched_training_cases: int, confidence: float, selection: str, expected: str
) -> None:
    rule = {"matched_training_cases": matched_training_cases, "confidence": confidence, "selection": selection}
    assert rules_module.support_tier(rule) == expected


def test_match_reports_support_tier_without_touching_model_schema() -> None:
    train = [make_case(f"case-{index}", "L1" if index % 2 else "L2", ["a", "b"]) for index in range(10)]
    engine = rules_module.SymbolicRuleEngine().fit(train)
    result = engine.match(make_case("query", "", ["a", "b"]))
    matched = [rule for rules in result["matched_rules"].values() for rule in rules]
    assert matched, "构造的训练集应至少命中一条规则"
    assert all(rule["support_tier"] in rules_module.SUPPORT_TIERS for rule in matched)
    assert sum(result["support_tier_counts"].values()) == result["matched_rule_count"]
    saved = engine.to_dict()["rule_sets"]
    assert all(
        "support_tier" not in rule for rules in saved.values() for rule in rules
    ), "support_tier 不得写进冻结的 model schema"


def test_symbolic_evidence_items_carry_rule_antecedent_as_origin() -> None:
    train = [make_case(f"case-{index}", "L1" if index % 2 else "L2", ["a", "b"]) for index in range(10)]
    engine = rules_module.SymbolicRuleEngine().fit(train)
    result = engine.match(make_case("query", "", ["a", "b"]))
    items = rules_module.evidence_items(result)
    assert items and all(item.source == "symbolic_rule" for item in items)
    assert all(set(item.origin_anomalies) <= {"a", "b"} for item in items)
    empty = engine.match(make_case("query", "", ["zzz"]))
    fallback = rules_module.evidence_items(empty)
    assert len(fallback) == 1 and fallback[0].is_prior_only is True


def test_support_audit_reports_weak_rules_per_label() -> None:
    train = [make_case(f"case-{index}", "L1" if index % 2 else "L2", ["a", "b"]) for index in range(10)]
    audit = rules_module.SymbolicRuleEngine().fit(train).support_audit()
    assert set(audit) == set(ROOT_CAUSES)
    for label in ROOT_CAUSES:
        assert audit[label]["rule_count"] == sum(audit[label]["support_tier"].values())


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (graph_result(matched={"L1": [{"all_of": ["a", "b"]}], "L2": [], "fiber": []}), "covered_pair"),
        (graph_result(matched={"L1": [{"all_of": ["a"]}], "L2": [], "fiber": []}), "covered_singleton"),
        (graph_result(paths=[{"root_cause": "L1", "anomaly_id": "a", "score": 1.0}], similarity=0.8), "covered_exemplar"),
        (graph_result(paths=[{"root_cause": "L1", "anomaly_id": "a", "score": 1.0}], similarity=0.2), "partial"),
        (graph_result(), "uncovered"),
    ],
)
def test_classify_coverage_five_states(result: Dict[str, Any], expected: str) -> None:
    report = graph_module.classify_coverage(make_case("query", "", ["a"]), result)
    assert report.state == expected
    assert report.state in graph_module.COVERAGE_STATES


def test_uncovered_case_is_reported_as_prior_only_with_full_prior_floor() -> None:
    train = [make_case(f"case-{index}", "L1" if index % 2 else "L2", ["a"]) for index in range(10)]
    graph = graph_module.AnomalyKnowledgeGraph().fit(train)
    result = graph.query(make_case("query", "", []))
    assert result["prior_only"] is True
    assert result["score_composition"]["prior_floor"] == 1.0
    assert result["scores"] == pytest.approx(graph.priors)
    items = graph_module.evidence_items(result)
    assert len(items) == 1 and items[0].is_prior_only is True


def test_covered_case_attributes_score_to_evidence_not_prior() -> None:
    train = [make_case(f"case-{index}", "L1" if index % 2 else "L2", ["a"]) for index in range(10)]
    graph = graph_module.AnomalyKnowledgeGraph().fit(train)
    composition = graph.query(make_case("query", "", ["a"]))["score_composition"]
    assert composition["prior_floor"] < 1.0
    assert composition["path_evidence"] > 0.0
    assert sum(composition.values()) == pytest.approx(1.0, abs=1e-6)


def test_retrieval_split_keeps_legacy_results_and_can_hide_labels() -> None:
    train = [make_case(f"case-{index}", "L1" if index % 2 else "L2", ["a", "b"]) for index in range(6)]
    graph = graph_module.AnomalyKnowledgeGraph().fit(train)
    query = make_case("query", "", ["a"])
    legacy = graph.retrieve(query, 3)
    moved = retrieval.retrieve(graph.train_index, graph.idf, query, 3)
    assert legacy == moved
    hidden = graph.retrieve(query, 3, hide_labels=True)
    assert all("root_cause" not in row for row in hidden)
    assert [row["similarity"] for row in hidden] == [row["similarity"] for row in legacy]
    assert [row["case_id"] for row in hidden] == [row["case_id"] for row in legacy]


def test_query_can_skip_retrieval_for_agent_tool_split() -> None:
    train = [make_case(f"case-{index}", "L1" if index % 2 else "L2", ["a"]) for index in range(6)]
    graph = graph_module.AnomalyKnowledgeGraph().fit(train)
    query = make_case("query", "", ["a"])
    assert graph.query(query, include_retrieval=False)["retrieved_cases"] == []
    assert graph.query(query)["scores"] == graph.query(query, include_retrieval=False)["scores"]
