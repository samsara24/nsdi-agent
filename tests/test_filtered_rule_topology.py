from pathlib import Path

import pytest

from rca_framework.data import cases_by_manifest_split, manifest_splits
from rca_framework.evidence_graph import EvidenceGraph, match
from rca_framework.evidence_pack import build_packs
from rca_framework.features.dictionary import FILTERED_RULE_DICTIONARY
from rca_framework.features.extractor import CaseFeatures
from scripts.run_filtered_rule_temporal_experiment import (
    FORMAL_GUIDED_JSON,
    FORMAL_MAX_ATTEMPTS,
    FORMAL_MAX_MODEL_LEN,
    FORMAL_MAX_NEW_TOKENS,
    _assert_retry_contract_traces,
    _parser,
)


DATA_DIR = Path(__file__).resolve().parents[1] / "datasets" / "filtered_rule_temporal_2025_06_09_v1"


def _features(case_id, tokens, source, topology):
    return CaseFeatures(
        case_id=case_id,
        tokens=tuple(tokens),
        by_family={},
        dictionary_version=FILTERED_RULE_DICTIONARY.version,
        dictionary_hash=FILTERED_RULE_DICTIONARY.content_hash(),
        source_dataset=source,
        topology_id=topology,
        lane_profile=topology,
    )


def test_manifest_exposes_one_train_and_two_source_tests():
    assert manifest_splits(DATA_DIR) == (
        "test/all_data",
        "test/rule1_channel_not_4",
        "train",
    )
    assert len(cases_by_manifest_split(DATA_DIR, "train")) == 124
    assert len(cases_by_manifest_split(DATA_DIR, "test/all_data")) == 417
    assert len(cases_by_manifest_split(DATA_DIR, "test/rule1_channel_not_4")) == 67


def test_case_contract_preserves_source_topology_and_endpoint_speeds():
    tests = cases_by_manifest_split(DATA_DIR, "test/all_data")
    rule1 = cases_by_manifest_split(DATA_DIR, "test/rule1_channel_not_4")
    all_pack = build_packs([tests[0]], source_dataset=str(DATA_DIR))[0]
    rule_pack = build_packs([rule1[0]], source_dataset=str(DATA_DIR))[0]
    assert all_pack.source_dataset == "all_data"
    assert all_pack.topology_id == "400g-200g-logical4"
    assert rule_pack.source_dataset == "rule1_channel_not_4"
    assert rule_pack.topology_id == "400g-400g-logical8"
    assert rule_pack.lane_widths["txpower"] == {"L1": 8, "L2": 8}


def test_retrieval_prefers_same_topology_before_top_k_truncation():
    same = _features("same", ("shared", "same-only", "extra"), "all_data", "topology-a")
    cross = _features("cross", ("shared", "query-only"), "rule1_channel_not_4", "topology-b")
    graph = EvidenceGraph.build(
        (same, cross),
        ("L1", "L2"),
        dictionary=FILTERED_RULE_DICTIONARY,
        source_dataset="combined",
    )
    query = _features("query", ("shared", "query-only"), "all_data", "topology-a")
    result = match(graph, query, top_k=1)
    assert result.candidates[0].case_id == "same"
    assert result.candidates[0].topology_compatible is True
    assert result.uses_cross_topology_fallback is False


def test_cross_topology_is_explicit_fallback_when_no_compatible_overlap():
    same = _features("same", ("unrelated",), "all_data", "topology-a")
    cross = _features("cross", ("shared",), "rule1_channel_not_4", "topology-b")
    graph = EvidenceGraph.build(
        (same, cross),
        ("L1", "L2"),
        dictionary=FILTERED_RULE_DICTIONARY,
        source_dataset="combined",
    )
    query = _features("query", ("shared",), "all_data", "topology-a")
    result = match(graph, query, top_k=1)
    assert result.candidates[0].case_id == "cross"
    assert result.uses_cross_topology_fallback is True


def test_formal_filtered_rule_generation_contract_retries_failures_with_long_output():
    args = _parser().parse_args(["--output-dir", "unused", "--model-path", "unused"])
    assert args.max_attempts == FORMAL_MAX_ATTEMPTS == 3
    assert args.max_new_tokens == FORMAL_MAX_NEW_TOKENS == 16384
    assert args.max_model_len == FORMAL_MAX_MODEL_LEN == 32768
    assert args.policy == "filtered-rule-three-channel-v2"
    assert FORMAL_GUIDED_JSON is True


def test_filtered_match_keeps_feature_and_graph_similarity_independent():
    history = _features(
        "history",
        ("drop:L1:rxpower:all_lanes",),
        "all_data",
        "topology-a",
    )
    graph = EvidenceGraph.build(
        (history,),
        ("L1",),
        dictionary=FILTERED_RULE_DICTIONARY,
        source_dataset="combined",
    )
    query = _features(
        "query",
        ("drop:L1:rxpower:single_lane",),
        "all_data",
        "topology-a",
    )
    result = match(graph, query, top_k=0)
    candidate = result.candidates[0]
    assert candidate.feature_similarity == 0.0
    assert 0.0 < candidate.graph_similarity < 1.0
    assert candidate.shared_graph_edges
    assert result.max_feature_similarity == 0.0
    assert result.max_graph_similarity == candidate.graph_similarity


def test_retry_trace_gate_accepts_one_to_three_attempts_and_rejects_invalid_counts():
    _assert_retry_contract_traces(
        {"case-a": {"attempt_count": 1}, "case-b": {"attempt_count": 3}},
        expected_case_count=2,
        scope="test",
    )
    with pytest.raises(RuntimeError, match="retry contract violated"):
        _assert_retry_contract_traces(
            {"case-a": {"attempt_count": 4}},
            expected_case_count=1,
            scope="test",
        )
    with pytest.raises(RuntimeError, match="expected one trace per case"):
        _assert_retry_contract_traces({}, expected_case_count=1, scope="test")
