from __future__ import annotations

import json
from pathlib import Path

from rca_framework.anomaly import fit_thresholds
from rca_framework.data import cases_by_manifest_split, load_split_manifest
from rca_framework.evidence_pack import build_packs
from rca_framework.features.dictionary import FEATURE_DICTIONARY_V2, dictionary_for
from rca_framework.features.extractor import extract_features, fit_feature_model
from rca_framework.sop import learn_sop


DATA_DIR = Path("datasets/rca_v2_l2fixed")


def test_l2fixed_manifest_is_reproducible_and_stratified():
    manifest = load_split_manifest(DATA_DIR)
    assert manifest["schema_version"] == "l2fixed-split-manifest-v1"
    assert manifest["case_count"] == 268
    assert manifest["split_counts"] == {
        "test": {"L1": 32, "L2": 67, "fiber": 8},
        "train": {"L1": 49, "L2": 100, "fiber": 12},
    }
    assert manifest["quality_summary"]["l1_metric_width_over_4_cases"] == 31
    assert manifest["quality_summary"]["l2_metric_width_over_4_cases"] == 0
    assert manifest["quality_summary"]["missing_alarm_ip_interface"] == 54


def test_manifest_split_loader_keeps_labels_separate():
    train = cases_by_manifest_split(DATA_DIR, "train")
    test = cases_by_manifest_split(DATA_DIR, "test")
    assert len(train) == 161
    assert len(test) == 107
    train_packs = build_packs(train, source_dataset=str(DATA_DIR))
    assert all(not pack.has_label_field() for pack in train_packs[:20])
    assert {case["label"] for case in train} == {"L1", "L2", "fiber"}


def test_feature_dictionary_v2_is_explicit_not_default():
    assert dictionary_for("v2") == FEATURE_DICTIONARY_V2
    assert FEATURE_DICTIONARY_V2.version == "feature-dictionary-v2"
    assert "serdes_state" in FEATURE_DICTIONARY_V2.family_names()
    assert "lane_direction" in FEATURE_DICTIONARY_V2.family_names()


def test_learned_sop_is_train_only_and_serializable():
    train = cases_by_manifest_split(DATA_DIR, "train")
    dictionary = dictionary_for("v2")
    thresholds = fit_thresholds(train)
    packs = build_packs(train, source_dataset=str(DATA_DIR))
    model = fit_feature_model(packs, dictionary=dictionary)
    features = [extract_features(pack, thresholds, model, dictionary=dictionary) for pack in packs]
    sop = learn_sop(features, [case["label"] for case in train], max_depth=3, min_leaf_size=5)
    payload = sop.to_dict()
    assert payload["version"] == "learned-sop-v1"
    assert payload["training_case_count"] == 161
    assert payload["dictionary_version"] == "feature-dictionary-v2"
    restored = json.loads(json.dumps(payload, ensure_ascii=False))
    assert restored["content_hash"] == sop.content_hash()
    prediction = sop.predict(features[0])
    assert prediction.support >= 5
    assert prediction.leaf_id
