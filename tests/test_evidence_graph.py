"""T4 测试：证据图索引（M2）与 Top-N 检索（M3）。

锁定三件事：
1. 证据图不在检索路径上泄漏标签，且版本号能唯一定位一份快照。
2. 检索内核与 legacy IDF 加权 Jaccard 语义一致，且结果确定性可复现。
3. 缺失 / 多余 / 冲突证据的拆分正确——这三样是 N5b 与 N6 的直接输入。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rca_framework.anomaly import fit_thresholds
from rca_framework.data import load_cases
from rca_framework.evidence_graph import EvidenceGraph, GraphCase, match, match_many
from rca_framework.evidence_graph.match import find_conflicts, weighted_jaccard
from rca_framework.evidence_pack import build_packs, labels_of
from rca_framework.features.extractor import extract_features, fit_feature_model


DATA_DIR = Path("datasets/organized_rca_v2_stratified_60_40_seed42")
TRAIN_SIZE = 126

#: T4 标定产物指纹。改动特征字典、FeatureModel 或训练切分都会让它变，
#: 变了就说明历史匹配结果与已归档的实验产物不可比。
GRAPH_CONTENT_HASH = "5e10b5b25d559777"


@pytest.fixture(scope="module")
def built():
    cases = load_cases(DATA_DIR)
    train, test = cases[:TRAIN_SIZE], cases[TRAIN_SIZE:]
    thresholds = fit_thresholds(train)
    train_packs, test_packs = build_packs(train), build_packs(test)
    model = fit_feature_model(train_packs)
    train_features = [extract_features(pack, thresholds, model) for pack in train_packs]
    test_features = [extract_features(pack, thresholds, model) for pack in test_packs]
    graph = EvidenceGraph.build(train_features, labels_of(train), feature_model=model)
    return graph, train_features, test_features, labels_of(train), labels_of(test)


# --- M2 存储与索引 ---------------------------------------------------------

def test_graph_version_is_frozen(built):
    graph, *_ = built
    assert len(graph) == TRAIN_SIZE
    assert graph.content_hash() == GRAPH_CONTENT_HASH
    assert graph.version == f"evidence-graph-v1:{TRAIN_SIZE}:{GRAPH_CONTENT_HASH}"
    assert graph.label_distribution() == {"L1": 35, "L2": 83, "fiber": 8}


def test_graph_reproduces_t1_purity_numbers(built):
    """证据图的纯净度必须与 T1 记录一致，否则说明索引路径改变了 signature。"""
    graph, *_ = built
    report = graph.purity_report()
    assert report["signature_group_count"] == 113
    assert report["mixed_label_group_count"] == 3
    assert report["mixed_label_case_count"] == 10
    assert report["mixed_label_case_ratio"] == pytest.approx(0.079365, abs=1e-6)
    assert report["singleton_group_count"] == 104
    assert report["empty_signature_case_count"] == 2


def test_token_index_and_signature_lookup_do_not_expose_labels(built):
    graph, *_ = built
    index = graph.token_index()
    assert len(index) == 40
    payload = json.dumps({"index": index, "signature": list(graph.signature_of(graph.cases[0].case_id))})
    for label in ("L1", "L2", "fiber"):
        # token 里本来就含 L1/L2 侧名，所以只断言索引结构里没有 label 字段。
        assert '"label"' not in payload
    # 取标签必须是一次显式调用。
    assert graph.label_of(graph.cases[0].case_id) in {"L1", "L2", "fiber"}


def test_graph_rejects_unknown_labels(built):
    graph, train_features, *_ = built
    with pytest.raises(ValueError, match="unsupported labels"):
        EvidenceGraph.build(train_features[:3], ["L1", "L2", "unknown"])
    with pytest.raises(ValueError, match="same length"):
        EvidenceGraph.build(train_features[:3], ["L1"])


def test_extend_returns_a_new_immutable_snapshot(built):
    graph, *_ = built
    added = GraphCase(case_id="feedback-1", label="fiber", tokens=("drop:L1:rxpower:all_lanes",))
    grown = graph.extend([added])
    assert len(grown) == TRAIN_SIZE + 1
    assert len(graph) == TRAIN_SIZE  # 原图不被就地修改
    assert grown.version != graph.version
    assert grown.idf != graph.idf  # 追加 case 必须重算 IDF
    with pytest.raises(ValueError, match="already in graph"):
        grown.extend([added])


def test_graph_round_trips_through_json(built):
    graph, *_ = built
    restored = EvidenceGraph.from_dict(json.loads(json.dumps(graph.to_dict(), ensure_ascii=False)))
    assert restored.version == graph.version
    assert restored.idf == graph.idf
    assert restored.cases == graph.cases


# --- M3 检索 ---------------------------------------------------------------

def test_similarity_kernel_matches_legacy_semantics():
    idf = {"a": 2.0, "b": 1.0, "c": 3.0}
    assert weighted_jaccard({"a", "b"}, {"a", "b"}, idf) == 1.0
    assert weighted_jaccard({"a"}, {"c"}, idf) == 0.0
    # 交集 a(2.0) / 并集 a+b+c(6.0)
    assert weighted_jaccard({"a", "b"}, {"a", "c"}, idf) == pytest.approx(2.0 / 6.0)


def test_empty_signature_never_matches_another_empty_signature():
    """两个零证据 case 的相似度必须是 0，不能是 1。

    否则「什么都没测到」的 case 会互相 100% 命中并填满 N5a，
    而它们恰恰是最该走人工介入的那批。
    """
    assert weighted_jaccard(set(), set(), {}) == 0.0


def test_retrieval_is_deterministic_and_sorted(built):
    graph, _, test_features, _, _ = built
    first = match(graph, test_features[0], top_k=5)
    second = match(graph, test_features[0], top_k=5)
    assert first.to_dict() == second.to_dict()
    scores = [item.similarity for item in first.candidates]
    assert scores == sorted(scores, reverse=True)
    assert len(first.candidates) == 5
    assert first.graph_version == graph.version


def test_leave_one_out_excludes_the_query_itself(built):
    graph, train_features, *_ = built
    query = train_features[0]
    with_self = match(graph, query, top_k=1)
    without_self = match(graph, query, top_k=1, exclude_case_ids=(query.case_id,))
    assert with_self.candidates[0].case_id == query.case_id
    assert with_self.max_similarity == 1.0
    assert without_self.candidates[0].case_id != query.case_id


def test_hide_labels_blanks_out_every_candidate(built):
    graph, _, test_features, _, _ = built
    result = match(graph, test_features[0], top_k=5, hide_labels=True)
    assert all(item.label is None for item in result.candidates)
    assert result.tie_labels == ()


def test_evidence_is_split_into_shared_missing_and_extra(built):
    graph, _, test_features, _, _ = built
    for features in test_features[:40]:
        result = match(graph, features, top_k=3)
        query = set(features.tokens)
        for candidate in result.candidates:
            history = set(graph.signature_of(candidate.case_id))
            assert set(candidate.shared_evidence) == query & history
            assert set(candidate.missing_evidence) == history - query
            assert set(candidate.extra_evidence) == query - history


def test_missing_evidence_is_the_intersection_over_tied_candidates(built):
    """补采清单取并列候选的交集：只让人去补每个候选都要求的证据。"""
    graph, _, test_features, _, _ = built
    for features in test_features:
        result = match(graph, features, top_k=0)
        top = result.top_candidates
        if len(top) < 2:
            continue
        expected = set(top[0].missing_evidence)
        for candidate in top[1:]:
            expected &= set(candidate.missing_evidence)
        assert set(result.missing_evidence) == expected
        return
    pytest.skip("no tied candidates in this split")


def test_conflict_detection_is_dimension_wise():
    assert find_conflicts(
        {"drop:L1:rxpower:all_lanes"}, {"drop:L1:rxpower:single_lane"}
    ) == (("drop:L1:rxpower:all_lanes", "drop:L1:rxpower:single_lane"),)
    # 同一分档不是冲突。
    assert find_conflicts({"drop:L1:rxpower:all_lanes"}, {"drop:L1:rxpower:all_lanes"}) == ()
    # 不同维度不是冲突。
    assert find_conflicts({"drop:L1:rxpower:all_lanes"}, {"drop:L2:rxpower:single_lane"}) == ()
    # 完全匹配的 case 之间不可能有冲突。
    assert find_conflicts({"level:L1:rxpower_mean:low_tail"}, {"level:L1:rxpower_mean:low_tail"}) == ()


def test_exact_match_implies_full_coverage_and_no_conflict(built):
    """sim=1.0 的定义性质：证据全覆盖、无补采清单、无冲突。"""
    graph, _, test_features, _, _ = built
    exact = [
        result for result in match_many(graph, test_features, top_k=0)
        if result.max_similarity >= 1.0
    ]
    assert len(exact) == 21
    for result in exact:
        assert result.evidence_coverage == 1.0
        assert result.missing_evidence == ()
        assert not result.has_conflict


def test_n4_distribution_reproduces_t1_numbers(built):
    """证据图路径必须复现 T1 记录的 N4 分布，否则两阶段产物不可比。"""
    graph, _, test_features, _, _ = built
    results = match_many(graph, test_features, top_k=0)
    n5a = sum(1 for item in results if item.max_similarity >= 1.0)
    n5b = sum(1 for item in results if 0.7 <= item.max_similarity < 1.0)
    n5c = sum(1 for item in results if item.max_similarity < 0.7)
    assert (n5a, n5b, n5c) == (21, 26, 38)


def test_t4_routing_calibration_numbers(built):
    """T4 标定结论：证据全覆盖比 70% 相似度更适合做 N5b 的入口条件。

    两个口径都在这里锁定，防止后续改动悄悄推翻标定结果（见 Validation.md V1）。
    """
    from collections import Counter

    from rca_framework.types import ROOT_CAUSES

    graph, train_features, test_features, train_labels, test_labels = built

    def majority(result):
        vote = Counter(result.tie_labels)
        if not vote:
            return None
        top = max(vote.values())
        return min((label for label in vote if vote[label] == top), key=ROOT_CAUSES.index)

    def exit_stats(results, actual, use_coverage):
        selected = [
            (result, label)
            for result, label in zip(results, actual)
            if result.max_similarity >= 1.0
            or (result.evidence_coverage >= 1.0 if use_coverage else result.max_similarity >= 0.7)
        ]
        return len(selected), sum(majority(result) == label for result, label in selected)

    loo = match_many(graph, train_features, top_k=0, leave_one_out=True)
    held_out = match_many(graph, test_features, top_k=0)

    assert exit_stats(loo, train_labels, use_coverage=False) == (67, 48)
    assert exit_stats(loo, train_labels, use_coverage=True) == (61, 45)
    assert exit_stats(held_out, test_labels, use_coverage=False) == (47, 32)
    assert exit_stats(held_out, test_labels, use_coverage=True) == (39, 31)
