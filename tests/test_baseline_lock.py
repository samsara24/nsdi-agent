"""确定性基线锁定测试。

这些测试是 Agent 化改造的硬门禁：任何声称 legacy 兼容的改动，都必须让 organized
60/40 deterministic 基线逐 case 复现 `artifacts/organized_rca_v2_60_40_seed42_baseline/`
中的产物。只要断言失败，就说明 legacy 行为发生了漂移，而不是测试需要更新。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from rca_framework.data import load_cases
from rca_framework.pipeline import RCAPipeline
from rca_framework.runtime import RuntimeConfig


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "datasets" / "organized_rca_v2_stratified_60_40_seed42"
BASELINE_DIR = ROOT / "artifacts" / "organized_rca_v2_60_40_seed42_baseline"
TRAIN_SIZE = 126

pytestmark = pytest.mark.skipif(
    not (DATA_DIR.exists() and BASELINE_DIR.exists()),
    reason="organized 60/40 数据集或基线 artifacts 不在本地",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def baseline_run() -> Dict[str, Any]:
    cases = load_cases(DATA_DIR)
    pipeline = RCAPipeline().fit(cases[:TRAIN_SIZE])
    evaluation = pipeline.evaluate(cases[TRAIN_SIZE:], runtime=RuntimeConfig())
    return {"pipeline": pipeline, "evaluation": evaluation, "case_count": len(cases)}


@pytest.fixture(scope="module")
def expected_predictions() -> List[Dict[str, Any]]:
    return read_json(BASELINE_DIR / "predictions.json")


def test_dataset_split_is_unchanged(baseline_run: Dict[str, Any]) -> None:
    manifest = read_json(BASELINE_DIR / "run_manifest.json")
    cases = load_cases(DATA_DIR)
    assert baseline_run["case_count"] == manifest["train_size"] + manifest["test_size"]
    assert [case["case_id"] for case in cases[:TRAIN_SIZE]] == manifest["train_case_ids"]
    assert [case["case_id"] for case in cases[TRAIN_SIZE:]] == manifest["test_case_ids"]


def test_summary_matches_baseline(baseline_run: Dict[str, Any]) -> None:
    assert baseline_run["evaluation"]["summary"] == read_json(BASELINE_DIR / "evaluation_summary.json")


def test_per_case_prediction_matches_baseline(
    baseline_run: Dict[str, Any], expected_predictions: List[Dict[str, Any]]
) -> None:
    actual = baseline_run["evaluation"]["predictions"]
    assert len(actual) == len(expected_predictions)
    for row, expected in zip(actual, expected_predictions):
        assert row["case_id"] == expected["case_id"]
        assert row["prediction"] == expected["prediction"]
        assert row["actual_label"] == expected["actual_label"]
        assert row["decision_status"] == expected["decision_status"]
        assert row["confidence"] == pytest.approx(expected["confidence"], rel=0, abs=1e-12)


def test_per_case_route_scores_match_baseline(
    baseline_run: Dict[str, Any], expected_predictions: List[Dict[str, Any]]
) -> None:
    for row, expected in zip(baseline_run["evaluation"]["predictions"], expected_predictions):
        for route in ("KG_RAG_LLM", "KG_RCA"):
            assert row[route]["scores"] == pytest.approx(expected[route]["scores"], rel=0, abs=1e-12)
        assert row["KG_RAG_LLM"]["reasoning_mode"] == expected["KG_RAG_LLM"]["reasoning_mode"]


def test_per_case_anomaly_ids_match_baseline(
    baseline_run: Dict[str, Any], expected_predictions: List[Dict[str, Any]]
) -> None:
    for row, expected in zip(baseline_run["evaluation"]["predictions"], expected_predictions):
        assert [item["anomaly_id"] for item in row["extracted_anomalies"]] == [
            item["anomaly_id"] for item in expected["extracted_anomalies"]
        ]


def as_serialized(model: Dict[str, Any]) -> Any:
    """按 JSON 往返后的形态比较，避免 tuple 与 list 的表示差异。"""
    return json.loads(json.dumps(model, ensure_ascii=False))


def test_model_artifact_stays_loadable_and_identical(baseline_run: Dict[str, Any]) -> None:
    saved = read_json(BASELINE_DIR / "model" / "model.json")
    assert as_serialized(baseline_run["pipeline"].to_dict()) == saved
    reloaded = RCAPipeline.load(BASELINE_DIR / "model")
    assert as_serialized(reloaded.to_dict()) == saved


def test_symbolic_rules_stay_mutually_exclusive(baseline_run: Dict[str, Any]) -> None:
    assert baseline_run["pipeline"].rules.overlap_audit()["total_overlap_count"] == 0


def test_model_serialization_is_hash_seed_independent(baseline_run: Dict[str, Any]) -> None:
    """`idf` 由集合迭代构建，键序必须固定，否则同一模型每次导出字节都不同。"""
    idf_keys = list(baseline_run["pipeline"].graph.idf)
    assert idf_keys == sorted(idf_keys)
