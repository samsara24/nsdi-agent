"""确定性基线锁定测试。

这些测试是 Agent 化改造的硬门禁：任何声称 legacy 兼容的改动，都必须让 organized
60/40 deterministic 基线逐 case 复现 `artifacts/organized_rca_v2_60_40_seed42_baseline/`
中的产物。只要断言失败，就说明 legacy 行为发生了漂移，而不是测试需要更新。
"""

from __future__ import annotations

import json
from collections import Counter
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
    """legacy summary 的每个键都必须逐值复现基线。

    从阶段 1 起 summary 会追加纯观测键（`observations`），因此这里按键比对而不是
    整字典比对，但基线里出现过的键一个都不许漂移，也不许消失。
    """
    expected = read_json(BASELINE_DIR / "evaluation_summary.json")
    actual = baseline_run["evaluation"]["summary"]
    assert set(expected) <= set(actual)
    for key, value in expected.items():
        assert actual[key] == value, f"legacy summary 键 {key} 发生漂移"
    assert set(actual) - set(expected) == {"observations"}


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


def assert_additive(actual: Any, expected: Any, path: str, added: List[str]) -> None:
    """递归断言 `actual` 相对 `expected` 只新增键，不改值、不删键。"""
    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"{path} 的类型发生变化"
        missing = set(expected) - set(actual)
        assert not missing, f"{path} 丢失了 legacy 键 {sorted(missing)}"
        added.extend(f"{path}.{key}" for key in set(actual) - set(expected))
        for key in expected:
            assert_additive(actual[key], expected[key], f"{path}.{key}", added)
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), f"{path} 的长度发生变化"
        for index, (left, right) in enumerate(zip(actual, expected)):
            assert_additive(left, right, f"{path}[{index}]", added)
    else:
        assert actual == expected, f"{path} 的值发生漂移: {actual!r} != {expected!r}"


def test_predictions_only_add_observation_fields(
    baseline_run: Dict[str, Any], expected_predictions: List[Dict[str, Any]]
) -> None:
    """阶段 1 起的核心门禁：新增观测字段只能是附加的。

    逐层比对整份 `predictions.json`，任何 legacy 值改变或键消失都会失败。
    新增键会被收集起来并断言其集合，避免无意中往产物里塞字段。
    """
    added: List[str] = []
    assert_additive(as_serialized(baseline_run["evaluation"]["predictions"]), expected_predictions, "predictions", added)
    assert {item.split(".")[-1] for item in added} == {"observation", "support_tier", "support_tier_counts"}


def test_prediction_record_only_gains_observation_key(
    baseline_run: Dict[str, Any], expected_predictions: List[Dict[str, Any]]
) -> None:
    actual_keys = set(baseline_run["evaluation"]["predictions"][0])
    baseline_keys = set(expected_predictions[0])
    assert baseline_keys <= actual_keys
    assert actual_keys - baseline_keys == {"observation"}


def test_observation_numbers_match_stage1_measurement(baseline_run: Dict[str, Any]) -> None:
    """锁定阶段 1 实测的观测数字。

    与 legacy 断言的区别：这些数字允许随观测口径演进而改变，但必须是有意改变，
    并同步更新 `Progress.md`。它们目前是"双路架构是否真的提供两路证据"的唯一证据。
    """
    observations = baseline_run["evaluation"]["summary"]["observations"]
    assert observations["coverage_state"] == {
        "covered_pair": 47,
        "covered_singleton": 5,
        "covered_exemplar": 10,
        "partial": 1,
        "uncovered": 22,
    }
    assert observations["prior_only_cases"] == 22
    assert observations["agreement_type"] == {
        "independent_agreement": 4,
        "same_source_agreement": 58,
        "conflict": 1,
        "no_evidence": 22,
    }
    assert observations["rule_support_audit"]["fiber"] == {
        "rule_count": 28,
        "support_tier": {"strong": 0, "moderate": 0, "low_support": 28},
        "selection": {"strict": 28},
    }


def test_legacy_agreement_is_mostly_same_source(baseline_run: Dict[str, Any]) -> None:
    """82 条 legacy `agreement` 里真正的独立互证只有 2 条。

    legacy 把 `agreement` 解释为"两条独立推理链结论一致"，这条测试固定住反例：
    绝大多数是同源一致，还有 22 条根本没有 case 特异证据。
    """
    rows = [
        row for row in baseline_run["evaluation"]["predictions"]
        if row["decision_status"] == "agreement"
    ]
    assert len(rows) == 82
    kinds = Counter(row["observation"]["evidence"]["agreement_type"] for row in rows)
    assert kinds == {"same_source_agreement": 58, "no_evidence": 22, "independent_agreement": 2}


def test_prior_only_cases_all_fall_back_to_majority_class(baseline_run: Dict[str, Any]) -> None:
    rows = [
        row for row in baseline_run["evaluation"]["predictions"]
        if row["observation"]["coverage"]["prior_only"]
    ]
    assert len(rows) == 22
    assert {row["prediction"] for row in rows} == {"L2"}
    assert all(row["observation"]["score_composition"]["prior_floor"] == 1.0 for row in rows)


def test_model_serialization_is_hash_seed_independent(baseline_run: Dict[str, Any]) -> None:
    """`idf` 由集合迭代构建，键序必须固定，否则同一模型每次导出字节都不同。"""
    idf_keys = list(baseline_run["pipeline"].graph.idf)
    assert idf_keys == sorted(idf_keys)
