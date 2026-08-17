import importlib.util
import json
import sys
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_formal_evaluation_writes_reproducible_artifacts(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("evaluate_routing", Path("scripts/evaluate_routing.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    output_dir = tmp_path / "run"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_routing.py",
            "--llm-backend",
            "none",
            "--policies",
            "coverage-v2",
            "--output-dir",
            str(output_dir),
        ],
    )
    module.main()

    assert {path.name for path in output_dir.iterdir()} == {
        "summary.json",
        "run_manifest.json",
        "outcomes.json",
        "traces.json",
    }
    manifest = read_json(output_dir / "run_manifest.json")
    assert manifest["versions"]["evidence_graph"].startswith("evidence-graph-v1:")
    assert manifest["versions"]["feature_dictionary_hash"]
    assert manifest["versions"]["constraint_library_hash"]
    assert manifest["versions"]["prompt_template_hash"]
    assert manifest["decision"]["final_lower_bound"] == 0.5
    assert manifest["data"]["train_size"] == 126
    assert manifest["data"]["test_size"] == 85

    summary = read_json(output_dir / "summary.json")["policies"]["coverage-v2"]
    assert summary["routing"]["counts"] == {"N5a": 20, "N5b": 17, "N5c": 46, "N6": 2}
    assert summary["n5a"]["mixed_signature_cases"] == 1
    assert summary["final_decisions"]["answered"] == 25
    assert summary["final_decisions"]["class_metrics"]["fiber"]["support"] == 6
    assert summary["forced_class_metrics"]["fiber"]["support"] == 6
    assert summary["threshold_sweep"]
    assert summary["selective_risk_curve"]

    outcomes = read_json(output_dir / "outcomes.json")["coverage-v2"]
    assert len(outcomes) == 85
    assert all({"match", "routing", "branch_outcome", "final_decision"} <= set(row) for row in outcomes)
    assert read_json(output_dir / "traces.json") == {"coverage-v2": {}}

