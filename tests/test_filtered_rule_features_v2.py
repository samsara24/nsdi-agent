from __future__ import annotations

import copy
from pathlib import Path

from rca_framework.data import cases_by_manifest_split
from rca_framework.evidence_pack import EvidencePack, build_packs
from rca_framework.features.dictionary import dictionary_for
from rca_framework.features.extractor import extract_features, fit_feature_model
from rca_framework.anomaly import fit_thresholds


DATA_DIR = Path(__file__).resolve().parents[1] / "datasets/filtered_rule_temporal_2025_06_09_v1"


def _permute(value):
    if not isinstance(value, dict):
        return copy.deepcopy(value)
    keys = list(value)
    if keys and all(str(key).isdigit() for key in keys):
        ordered = sorted(keys, key=lambda key: int(str(key)))
        return {new: copy.deepcopy(value[old]) for old, new in zip(ordered, reversed(ordered))}
    return {key: _permute(item) for key, item in value.items()}


def test_filtered_rule_v2_keeps_only_root_cause_signature_families():
    dictionary = dictionary_for("filtered_rule_v2")
    assert dictionary.version == "filtered-rule-feature-dictionary-v2"
    assert dictionary.family_names() == (
        "status_fault",
        "paired_lane_state",
        "signal_drop_ratio",
        "topology_level_tail",
    )
    assert "telemetry_gap" not in dictionary.family_names()
    assert "serdes_state" not in dictionary.family_names()
    assert "lane_imbalance" not in dictionary.family_names()


def test_filtered_rule_v2_is_lane_permutation_invariant_and_width_calibrated():
    cases = cases_by_manifest_split(DATA_DIR, "train")
    dictionary = dictionary_for("filtered_rule_v2")
    packs = build_packs(cases)
    thresholds = fit_thresholds([pack.telemetry for pack in packs])
    model = fit_feature_model(packs, dictionary=dictionary)
    assert model.version == "filtered-rule-feature-model-v2"
    assert any("400g-200g-logical4" in key and "width4" in key for key in model.topology_level_edges)
    assert any("400g-400g-logical8" in key and "width8" in key for key in model.topology_level_edges)

    for case, pack in zip(cases, packs):
        before = extract_features(pack, thresholds, model, dictionary=dictionary)
        changed = _permute({key: value for key, value in case.items() if key != "label"})
        changed["case_id"] = case["case_id"]
        after = extract_features(EvidencePack.from_case(changed), thresholds, model, dictionary=dictionary)
        assert before.tokens == after.tokens, case["case_id"]
