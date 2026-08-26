from pathlib import Path

from rca_framework.knowledge import OfflineKnowledgeBundle
from scripts.build_filtered_rule_decision_graph import build, skeleton


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets/filtered_rule_temporal_2025_06_09_v1"
KNOWLEDGE = ROOT / "artifacts/filtered_rule_deterministic_knowledge_v2/knowledge_bundle.json"


def test_decision_graph_fallback_is_insufficient_not_default_l1():
    nodes, edges = skeleton()
    node_ids = {node["id"] for node in nodes}
    assert "insufficient" in node_ids
    assert any(edge["src"] == "quality" and edge["dst"] == "insufficient" for edge in edges)
    assert any(edge["src"] == "merge" and edge["dst"] == "insufficient" for edge in edges)
    assert not any(edge["src"] == "port_gate" and edge["dst"] in {"out:L1", "out:L2"} for edge in edges)


def test_metric_predicates_are_endpoint_symmetric_and_train_paths_are_auditable():
    graph = build(DATA, OfflineKnowledgeBundle.load(KNOWLEDGE))
    predicates = [node for node in graph["nodes"] if node["kind"] == "MetricPredicate"]
    assert len(predicates) == 20
    assert all("L1" not in node["id"] and "L2" not in node["id"] for node in predicates)
    assert graph["train_case_count"] == 124
    assert sum(row["support"] for row in graph["train_paths"]) == 124
    assert graph["n8_frozen"] is True
    assert "no test cases or labels" in graph["build_boundary"]
