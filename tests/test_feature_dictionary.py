"""T1 锁定测试：特征字典 v1 的声明完整性、抽取器行为与验收数字。

这里锁的是 T1 的结论，不是实现细节。任何改动只要动了 v1 家族集合、分档规则或
分位边界，`content_hash` 与下面的验收数字就会一起变，从而强制走评审而不是悄悄漂移。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rca_framework.anomaly import extract_evidence, fit_thresholds
from rca_framework.data import load_cases
from rca_framework.evidence_pack import EvidencePack
from rca_framework.features.dictionary import (
    FEATURE_DICTIONARY,
    FULL_DICTIONARY,
    V1_FAMILIES,
    FeatureDictionary,
    dictionary_for,
)
from rca_framework.features.extractor import (
    FAMILY_EXTRACTORS,
    extract_features,
    fit_feature_model,
)


DATA_DIR = Path("datasets/organized_rca_v2_stratified_60_40_seed42")
TRAIN_SIZE = 126

#: 特征字典 v1 的内容指纹。改动字典任何字段都会让它变化。
V1_CONTENT_HASH = "1b2e66ed650ce60e"


@pytest.fixture(scope="module")
def split():
    cases = load_cases(DATA_DIR)
    return cases[:TRAIN_SIZE], cases[TRAIN_SIZE:]


@pytest.fixture(scope="module")
def fitted(split):
    train, _ = split
    thresholds = fit_thresholds(train)
    model = fit_feature_model([EvidencePack.from_case(case) for case in train])
    return thresholds, model


def strip(case):
    target = dict(case)
    target.pop("label", None)
    return target


def test_v1_families_are_frozen():
    assert FEATURE_DICTIONARY.version == "feature-dictionary-v1"
    assert FEATURE_DICTIONARY.family_names() == V1_FAMILIES
    assert FEATURE_DICTIONARY.family_names() == (
        "signal_drop",
        "status_fault",
        "lane_imbalance",
        "level_tail",
    )
    assert FEATURE_DICTIONARY.content_hash() == V1_CONTENT_HASH


def test_every_family_declares_physical_semantics():
    for family in FULL_DICTIONARY.families:
        assert family.dimension, family.name
        assert family.physical_meaning, family.name
        assert family.unit, family.name
        assert family.value_domain, family.name
        assert family.extraction_rule, family.name
        assert "{" in family.token_template, family.name
        if family.status == "candidate":
            assert family.selection_note, f"candidate 家族必须写明未入选理由: {family.name}"


def test_dictionary_is_json_serializable_and_hash_is_stable():
    payload = json.dumps(FEATURE_DICTIONARY.to_dict(), ensure_ascii=False, sort_keys=True)
    restored = json.loads(payload)
    assert restored["version"] == "feature-dictionary-v1"
    assert FEATURE_DICTIONARY.content_hash() == FEATURE_DICTIONARY.content_hash()
    assert FEATURE_DICTIONARY.content_hash() != FULL_DICTIONARY.content_hash()


def test_every_declared_family_has_an_extractor():
    assert set(FULL_DICTIONARY.family_names()) == set(FAMILY_EXTRACTORS)


def test_profiles_resolve_to_declared_families():
    for profile in ("v1", "legacy_equivalent", "v1_plus_lane_direction", "all_families"):
        dictionary = dictionary_for(profile)
        assert isinstance(dictionary, FeatureDictionary)
        assert set(dictionary.family_names()) <= set(FULL_DICTIONARY.family_names())
    with pytest.raises(KeyError):
        dictionary_for("does-not-exist")


def test_tokens_match_declared_family_prefix(split, fitted):
    train, _ = split
    thresholds, model = fitted
    prefixes = {
        "signal_drop": "drop:",
        "status_fault": "status:",
        "lane_imbalance": "imbalance:",
        "level_tail": "level:",
    }
    for case in train:
        features = extract_features(EvidencePack.from_case(case), thresholds, model)
        for family, tokens in features.by_family.items():
            for token in tokens:
                assert token.startswith(prefixes[family]), (family, token)
        assert features.tokens == tuple(sorted(set(features.tokens)))
        assert features.dictionary_hash == V1_CONTENT_HASH


def test_extraction_is_deterministic_and_label_independent(split, fitted):
    _, test = split
    thresholds, model = fitted
    for case in test[:20]:
        pack = EvidencePack.from_case(case)
        first = extract_features(pack, thresholds, model)
        second = extract_features(pack, thresholds, model)
        assert first.tokens == second.tokens
        # 证据包构造时就摘掉了标签，所以带标签的 case 也只能得到同一个 signature。
        assert not pack.has_label_field()
        assert extract_features(EvidencePack.from_case(strip(case)), thresholds, model).tokens == first.tokens


def test_feature_model_is_fitted_on_train_only(split):
    train, test = split
    fitted_on_train = fit_feature_model([EvidencePack.from_case(case) for case in train])
    fitted_on_all = fit_feature_model([EvidencePack.from_case(case) for case in train + test])
    assert fitted_on_train.fitted_case_count == TRAIN_SIZE
    assert fitted_on_train.dictionary_hash == V1_CONTENT_HASH
    assert fitted_on_train.level_edges != fitted_on_all.level_edges
    restored = type(fitted_on_train).from_dict(fitted_on_train.to_dict())
    assert restored.to_dict() == fitted_on_train.to_dict()


def test_extractor_does_not_touch_legacy_anomaly_ids(split, fitted):
    """M1 抽取器与 legacy `anomaly_id` 完全解耦，58/85 回归锚点不受影响。"""
    train, test = split
    thresholds, model = fitted
    for case in test:
        before = extract_evidence(strip(case), thresholds).anomaly_ids
        features = extract_features(EvidencePack.from_case(case), thresholds, model)
        after = extract_evidence(strip(case), thresholds).anomaly_ids
        assert before == after
        assert not (before & set(features.tokens))


def test_t1_acceptance_numbers(split, fitted):
    """锁定 T1 的两条量化验收结论。数字来自 `scripts/analyze_signature_resolution.py`。"""
    from scripts.analyze_signature_resolution import analyze

    legacy = analyze(DATA_DIR, TRAIN_SIZE, "legacy")
    v1 = analyze(DATA_DIR, TRAIN_SIZE, "v1")

    # 基线：混合标签 signature 覆盖 83/126，N5a 桶内多数投票低于 L2 多数类。
    assert legacy["signature_resolution"]["mixed_label_case_count"] == 83
    assert legacy["signature_resolution"]["distinct_signature_groups"] == 40
    assert legacy["signature_resolution"]["empty_signature_case_count"] == 31
    assert legacy["routing"]["distribution"] == {"N5a": 46, "N5b": 8, "N5c": 31}
    assert legacy["routing"]["N5a"]["majority_correct"] == 28
    assert legacy["routing"]["N5a"]["top1_correct"] == 29

    # 验收 1：混合标签 signature 覆盖率从 65.87% 降到 7.94%。
    assert v1["signature_resolution"]["mixed_label_case_count"] == 10
    assert v1["signature_resolution"]["distinct_signature_groups"] == 113
    assert v1["signature_resolution"]["empty_signature_case_count"] == 2
    assert v1["signature_resolution"]["mixed_label_case_ratio"] < 0.10

    # 验收 2：N5a 桶内多数投票 16/21 = 76.19%，高于 L2 多数类 64.71%。
    assert v1["routing"]["distribution"] == {"N5a": 21, "N5b": 26, "N5c": 38}
    assert v1["routing"]["N5a"]["majority_correct"] == 16
    assert v1["routing"]["N5a"]["majority_accuracy"] > 0.6471

    # 附带事实：纯历史匹配在全测试集上的多数投票准确率为 58/85。
    total = sum(v1["routing"][branch]["majority_correct"] for branch in ("N5a", "N5b", "N5c"))
    assert total == 58
