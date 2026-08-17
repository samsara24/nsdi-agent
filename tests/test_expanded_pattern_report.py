import json
from types import SimpleNamespace
from pathlib import Path

from scripts.analyze_expanded_rca_patterns import (
    compare_case_features,
    fit_comparison_model,
    fit_learned_predicate_model,
    graph_match,
    load_cases,
    observable_graph,
    token_criterion,
)


def test_observable_graph_does_not_treat_high_tail_context_as_fault_edge():
    graph = observable_graph(
        (
            "level:L1:txpower_mean:high_tail",
            "level:L2:rxpower_mean:high_tail",
            "level:L2:txpower_mean:high_tail",
        )
    )
    assert graph["nodes"] == ()
    assert graph["edges"] == ()
    assert graph["feature_coverage"] == 0.0
    assert len(graph["unmapped_tokens"]) == 3


def test_observable_graph_builds_typed_physical_edges_for_auditable_symptom():
    graph = observable_graph(("drop:L2:rxpower:all_lanes",))
    assert "side:L2" in graph["nodes"]
    assert "measurement:L2:rxpower" in graph["nodes"]
    assert "predicate:L2:rxpower:value_le_minus_39:all_lanes" in graph["nodes"]
    assert "symptom:no_received_light" in graph["nodes"]
    assert "physical-layer:receive_path" in graph["nodes"]
    assert "side:L2|has_measurement|measurement:L2:rxpower" in graph["edges"]
    assert "measurement:L2:rxpower|satisfies|predicate:L2:rxpower:value_le_minus_39:all_lanes" in graph["edges"]
    assert graph["feature_coverage"] == 1.0
    assert len(graph["paths"]) == 1


def test_graph_similarity_preserves_metric_and_predicate_identity():
    left = observable_graph(("drop:L2:rxpower:all_lanes",))
    right = observable_graph(("drop:L2:media_snr:all_lanes",))
    result = graph_match(left, right)
    assert result["similarity"] < 0.2
    assert result["edge_similarity"] == 0.0
    assert result["shared_predicate_paths"] == ()


def test_identical_five_layer_paths_match_exactly():
    left = observable_graph(("drop:L2:rxpower:all_lanes",))
    right = observable_graph(("drop:L2:rxpower:all_lanes",))
    result = graph_match(left, right)
    assert result["similarity"] == 1.0
    assert len(result["shared_predicate_paths"]) == 1


def test_imbalance_is_a_lane_relation_not_a_status_assertion():
    thresholds = SimpleNamespace(spread_upper={"L2:serdes_snr": 391200.0})
    graph = observable_graph(("imbalance:L2:serdes_snr",), thresholds=thresholds)
    assert "symptom:serdes_lane_quality_imbalance" in graph["nodes"]
    assert "predicate:L2:serdes_snr:healthy_lane_spread_gt_391200" in graph["nodes"]
    assert not any("status" in node for node in graph["nodes"])


def test_shared_relation_criteria_expose_fitted_thresholds():
    thresholds = SimpleNamespace(spread_upper={"L2:rxpower": 2.605})
    feature_model = SimpleNamespace(level_edges={"L2:rxpower_mean": (0.228125, 1.57875)})
    assert "2.605" in token_criterion("imbalance:L2:rxpower", thresholds, feature_model)
    assert "0.228125" in token_criterion("level:L2:rxpower_mean:low_tail", thresholds, feature_model)


def test_current_train_accepts_only_stable_supervised_ranges():
    root = Path(__file__).resolve().parents[1]
    cases = load_cases(root / "datasets/organized_rca_v2_stratified_60_40_seed42")[:126]
    model = fit_learned_predicate_model(cases)
    assert model["candidate_count"] == 16
    assert model["accepted_count"] == 2
    assert set(model["accepted"]) == {"level:L2:media_snr_min", "spread:L2:rxpower"}
    assert model["accepted"]["level:L2:media_snr_min"]["threshold"] == 23.805
    assert model["accepted"]["level:L2:media_snr_min"]["salient_branch"] == "le"
    assert model["accepted"]["spread:L2:rxpower"]["threshold"] == 2.165
    assert model["accepted"]["spread:L2:rxpower"]["salient_branch"] == "gt"


def test_case_difference_ranking_uses_train_scale_and_is_sorted():
    root = Path(__file__).resolve().parents[1]
    cases = load_cases(root / "datasets/organized_rca_v2_stratified_60_40_seed42")[:126]
    result = compare_case_features(cases[0], cases[1], fit_comparison_model(cases))
    ranked = result["ranked"]
    assert ranked
    assert [item["score"] for item in ranked] == sorted(
        (item["score"] for item in ranked), reverse=True
    )
    assert sum(result["counts"].values()) == len(ranked)
    assert result["largest"] == ranked[0]


def test_generated_report_is_expert_label_review_workbench():
    root = Path(__file__).resolve().parents[1]
    output = root / "experiments/20260816_expanded-pattern-conflict"
    html = (output / "expanded_rca_pattern_analysis.html").read_text()
    annotations = json.loads((output / "expert_annotation_template.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    contract = json.loads((output / "data_contract.json").read_text())
    assert "RCA 相似 Case 标签审核" in html
    assert "最终远端实验效果" in html
    assert "物理边界与证据优先级" in html
    assert "SerDes ≤1" in html
    assert "IDF(共享 typed edge)" in html
    assert 'id="review-search"' in html
    assert 'id="export-annotations"' in html
    assert html.count('class="expert-review"') == summary["pattern_count"]
    assert html.count('data-annotation-field="decision"') == summary["pattern_count"]
    assert "项细微差异（默认不突出）" in html
    assert html.index('id="review-list"') < html.index('class="methodology"')
    assert len(annotations["annotations"]) == summary["pattern_count"]
    assert contract["schema_version"] == "expanded-expert-clean-v1"
    assert contract["train_size"] == 122
    assert contract["test_size"] == 341
    assert contract["excluded_case_count"] == 6
    assert contract["reviewed_test_case_count"] == 42
    assert contract["unreviewed_test_case_count"] == 299
