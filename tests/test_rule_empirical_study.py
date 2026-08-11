import importlib.util
import json
import sys
from pathlib import Path

from rca_framework.branches.general import DiagnosisRequest
from rca_framework.llm.empirical import (
    build_empirical_prompt,
    empirical_prompt_hash,
)
from rca_framework.llm.protocol import DiagnosisResponse


def load_script():
    spec = importlib.util.spec_from_file_location(
        "rule_empirical", Path("scripts/run_rule_empirical_study.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def request():
    return DiagnosisRequest(
        case_id="case_query",
        evidence_tokens=("status:L1:RxLOS",),
        missing_fields=("L1.host_snr",),
        telemetry_status="partial_telemetry",
        candidate_root_causes=("L1", "L2", "fiber"),
        exclusions=(),
        constraint_ids=("C14_host_snr_mostly_missing",),
        nearest_similarity=0.0,
        branch="empirical",
        routing_reason="test",
        historical_case_ids=("case_history_must_not_appear",),
        historical_label_distribution=(("L2", 9),),
    )


def test_empirical_prompts_isolate_rules_and_never_include_history():
    evidence_only = build_empirical_prompt(request(), include_rules=False)
    rules = build_empirical_prompt(request(), include_rules=True)

    assert "status:L1:RxLOS" in evidence_only and "status:L1:RxLOS" in rules
    assert "C14_host_snr_mostly_missing" not in evidence_only
    assert "C14_host_snr_mostly_missing" in rules
    assert "case_history_must_not_appear" not in evidence_only
    assert "case_history_must_not_appear" not in rules
    assert '"L2": 9' not in evidence_only and '"L2": 9' not in rules
    assert empirical_prompt_hash(include_rules=False) != empirical_prompt_hash(include_rules=True)


def test_paired_comparison_counts_improvements_and_worsening():
    module = load_script()
    left = [
        DiagnosisResponse(verdict="L1"),
        DiagnosisResponse(verdict="L2"),
        DiagnosisResponse(verdict=None),
    ]
    right = [
        DiagnosisResponse(verdict="L1"),
        DiagnosisResponse(verdict="fiber"),
        DiagnosisResponse(verdict="fiber"),
    ]
    report = module.paired_comparison(left, right, ("L1", "L2", "fiber"))
    assert report["both_correct"] == 1
    assert report["improved_cases"] == 1
    assert report["worsened_cases"] == 1
    assert report["net_correct_change"] == 0


def test_none_backend_writes_complete_study_artifacts(tmp_path, monkeypatch):
    module = load_script()
    output = tmp_path / "study"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_rule_empirical_study.py",
            "--backend",
            "none",
            "--output-dir",
            str(output),
        ],
    )
    module.main()

    assert {path.name for path in output.iterdir()} == {
        "run_manifest.json",
        "summary.json",
        "outcomes.json",
        "traces.json",
    }
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["experimental_control"]["history_in_prompt"] is False
    assert manifest["experimental_control"]["checker_reuses_rules_prompt_first_pass"] is True
    assert manifest["experimental_control"]["retrieval_top_k_for_stratification_only"] == 0
    assert manifest["versions"]["sop"] == "not_used_in_isolated_rule_study"

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert set(summary["arms"]) == {
        "evidence_only",
        "rules_prompt",
        "rules_prompt_checker",
    }
    for arm in summary["arms"].values():
        assert arm["overall"]["case_count"] == 85
        assert arm["by_route"]["N5c"]["case_count"] == 46

    outcomes = json.loads((output / "outcomes.json").read_text(encoding="utf-8"))
    assert len(outcomes) == 85
    assert all(set(row["arms"]) == set(summary["arms"]) for row in outcomes)
