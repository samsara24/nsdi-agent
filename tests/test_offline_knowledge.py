from __future__ import annotations

import json
from pathlib import Path

import pytest

from rca_framework.branches import handle_many
from rca_framework.data import cases_by_manifest_split, load_split_manifest
from rca_framework.evidence_graph import COVERAGE_POLICY, match_many, route
from rca_framework.knowledge import OfflineKnowledgeBundle, fit_offline_knowledge
from rca_framework.llm import ConstrainedReasoner, ScriptedBackend


DATA_DIR = Path("datasets/rca_v2_l2fixed")


@pytest.fixture(scope="module")
def offline_fixture():
    manifest = load_split_manifest(DATA_DIR)
    train_cases = cases_by_manifest_split(DATA_DIR, "train")
    test_cases = cases_by_manifest_split(DATA_DIR, "test")
    bundle, artifacts = fit_offline_knowledge(
        train_cases,
        source_dataset=str(DATA_DIR),
        split_manifest_hash=manifest["source_hash"],
        feature_profile="v2",
        policies=(COVERAGE_POLICY,),
        top_k=0,
    )
    test_packs, test_features = bundle.extract_test_features(
        test_cases,
        source_dataset=str(DATA_DIR),
    )
    return bundle, artifacts, train_cases, test_cases, test_packs, test_features


def test_l2fixed_knowledge_bundle_roundtrip_and_test_read_only(offline_fixture, tmp_path):
    bundle, artifacts, train_cases, test_cases, test_packs, test_features = offline_fixture
    assert len(train_cases) == len(bundle.training_features) == len(bundle.graph.cases) == 161
    assert len(test_cases) == len(test_packs) == len(test_features) == 107
    assert len(bundle.graph.case_diagnoses) == 161
    assert artifacts.summary["historical_vector_count"] == 161
    assert bundle.sop.training_case_count == 161
    assert bundle.feature_profile == "v2"
    assert COVERAGE_POLICY.name in bundle.branch_calibrations
    assert all(not pack.has_label_field() for pack in test_packs)
    assert not ({item.case_id for item in bundle.graph.cases} & {pack.case_id for pack in test_packs})

    results = match_many(bundle.graph, test_features, top_k=0)
    assert len(results) == 107
    assert all(result.query_case_id == pack.case_id for result, pack in zip(results, test_packs))

    path = bundle.save(tmp_path / "knowledge_bundle.json")
    restored = OfflineKnowledgeBundle.load(path)
    assert restored.content_hash() == bundle.content_hash()
    assert restored.graph.version == bundle.graph.version
    assert restored.sop.content_hash() == bundle.sop.content_hash()
    assert restored.train_case_ids == bundle.train_case_ids


def test_knowledge_bundle_detects_tampering(offline_fixture):
    bundle = offline_fixture[0]
    payload = json.loads(json.dumps(bundle.to_dict(), ensure_ascii=False))
    payload["thresholds"]["fitted_case_count"] += 1
    with pytest.raises(ValueError, match="content hash mismatch"):
        OfflineKnowledgeBundle.from_dict(payload)


def test_sop_path_is_injected_before_constrained_llm(offline_fixture):
    bundle, _, _, _, test_packs, test_features = offline_fixture
    results = match_many(bundle.graph, test_features, top_k=0)
    selected = next(
        (result, pack, features)
        for result, pack, features in zip(results, test_packs, test_features)
        if route(result, COVERAGE_POLICY).branch == "N5c" and result.query_tokens
    )
    result, pack, features = selected
    token = result.query_tokens[0]
    answer = json.dumps(
        {
            "steps": [
                {
                    "claim": "该物理证据支持 L2 侧设备根因",
                    "cited_evidence": [token],
                    "cited_constraints": [],
                    "effect": "support",
                    "target": "L2",
                }
            ],
            "verdict": "L2",
            "confidence": 0.7,
            "missing_information": [],
        },
        ensure_ascii=False,
    )
    backend = ScriptedBackend(responses=[[answer]])
    paired = handle_many(
        [result],
        [pack],
        bundle.branch_calibrations[COVERAGE_POLICY.name],
        policy=COVERAGE_POLICY,
        reasoner=ConstrainedReasoner(backend=backend, max_attempts=1),
        features=[features],
        sop_model=bundle.sop,
    )
    _, outcome = paired[0]
    assert backend.prompts_seen
    assert "训练集归纳 SOP" in backend.prompts_seen[0][0]
    assert any(link.kind == "learned_sop" for link in outcome.evidence_chain)
    assert any(link.kind == "llm_step" for link in outcome.evidence_chain)
    assert outcome.needs_llm is False
