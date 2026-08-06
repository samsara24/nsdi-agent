from rca_framework.graph import AnomalyKnowledgeGraph
from rca_framework.rules import SymbolicRuleEngine
from rca_framework.types import Anomaly, CaseEvidence, ROOT_CAUSES


def item(anomaly_id: str, noun: str = "异常") -> Anomaly:
    return Anomaly(anomaly_id, "SignalDrop", noun, "HAS_SIGNAL_DROP", "L1", "rxpower", 1.0, noun)


def training_views() -> list[CaseEvidence]:
    rows = []
    patterns = {
        "L1": ["signal_drop:L1:rxpower", "status_fault:L1:TxLOS"],
        "L2": ["signal_drop:L2:rxpower", "status_fault:L2:TxLOS"],
        "fiber": ["bidirectional_loss:fiber:optical_power", "directional_loss:L1_to_L2:optical_power"],
    }
    for label in ROOT_CAUSES:
        for index in range(3):
            rows.append(CaseEvidence(
                f"{label}-{index}", label, [item(value, value) for value in patterns[label]], 10, 10,
            ))
    return rows


def test_graph_has_root_centers_and_anomaly_only_edges() -> None:
    graph = AnomalyKnowledgeGraph().fit(training_views())
    assert {node["label"] for node in graph.nodes.values() if node["node_type"] == "RootCause"} == set(ROOT_CAUSES)
    assert graph.edges
    assert all(edge.target.startswith("anomaly:") for edge in graph.edges)
    assert all(graph.nodes[edge.target]["node_type"] != "RootCause" for edge in graph.edges)
    query = CaseEvidence("new", "", [item("bidirectional_loss:fiber:optical_power")], 10, 10)
    result = graph.query(query)
    assert result["prediction"] == "fiber"
    assert result["paths"][0]["path"][0] == "query:new"
    assert graph.feature_rules["fiber"]
    assert result["matched_feature_rules"]["fiber"]
    assert result["feature_profile_scores"]["fiber"] > result["feature_profile_scores"]["L1"]


def test_three_symbolic_rule_sets_are_disjoint() -> None:
    engine = SymbolicRuleEngine().fit(training_views(), min_count=2)
    assert all(engine.rule_sets[label] for label in ROOT_CAUSES)
    assert engine.overlap_audit()["total_overlap_count"] == 0
    owners = {}
    for label, rules in engine.rule_sets.items():
        for rule in rules:
            assert rule.all_of not in owners
            owners[rule.all_of] = label
    query = CaseEvidence("new", "", [item("signal_drop:L2:rxpower")], 10, 10)
    assert engine.match(query)["prediction"] == "L2"
