from copy import deepcopy

from rca_framework.fusion import fuse_results
from rca_framework.pipeline import RCAPipeline
from rca_framework.types import CaseEvidence


def simple_case(index: int, label: str, side: str) -> dict:
    normal = {"0": 1.0, "1": 1.1, "2": 1.2, "3": 1.1}
    down = {"0": -40.0, "1": 1.1, "2": 1.2, "3": 1.1}
    return {
        "case_id": f"case-{index}",
        "label": label,
        "rxpower": {"L1": down if side == "L1" else normal, "L2": down if side == "L2" else normal},
        "txpower": {"L1": normal, "L2": normal},
        "media_snr": {"L1": normal, "L2": normal},
        "host_snr": {"L1": normal, "L2": normal},
        "serdes_snr": {"L1": {"0": 600000}, "L2": {"0": 600000}},
        "TxLOS": {"L1": "Normal", "L2": "Normal"},
        "TxLOL": {"L1": "Normal", "L2": "Normal"},
        "RxLOS": {"L1": "Abnormal" if side == "L1" else "Normal", "L2": "Abnormal" if side == "L2" else "Normal"},
        "RxLOL": {"L1": "Normal", "L2": "Normal"},
    }


def test_target_label_cannot_change_inference() -> None:
    train = [simple_case(i, "L1" if i % 2 == 0 else "L2", "L1" if i % 2 == 0 else "L2") for i in range(12)]
    pipeline = RCAPipeline().fit(train)
    target = simple_case(100, "L1", "L1")
    changed = deepcopy(target)
    changed["label"] = "fiber"
    first, second = pipeline.infer(target), pipeline.infer(changed)
    assert first["prediction"] == second["prediction"]
    assert first["KG_RAG_LLM"]["scores"] == second["KG_RAG_LLM"]["scores"]
    assert first["KG_RCA"]["scores"] == second["KG_RCA"]["scores"]


def test_pipeline_close_releases_and_forgets_cached_reasoners() -> None:
    class DummyReasoner:
        closed = False

        def close(self) -> None:
            self.closed = True

    pipeline = RCAPipeline()
    reasoner = DummyReasoner()
    pipeline._reasoners[object()] = reasoner
    pipeline.close()
    assert reasoner.closed is True
    assert pipeline._reasoners == {}


def test_fusion_marks_close_conflict_for_review() -> None:
    case = CaseEvidence("new", "", [], 5, 10)
    first = {"prediction": "L1", "confidence": 0.4, "scores": {"L1": 0.45, "L2": 0.4, "fiber": 0.15}}
    second = {"prediction": "L2", "confidence": 0.4, "scores": {"L1": 0.4, "L2": 0.45, "fiber": 0.15}, "matched_rules": {}}
    result = fuse_results(case, first, second)
    assert result["prediction"] in {"L1", "L2", "fiber"}
    assert result["decision_status"] == "manual_review_recommended"


def test_agreement_does_not_repeat_support_as_conflicting_evidence() -> None:
    case = CaseEvidence("new", "", [], 5, 10)
    paths = [
        {"root_cause": "L1", "anomaly_id": "a", "score": 0.8},
        {"root_cause": "L2", "anomaly_id": "b", "score": 0.2},
    ]
    first = {
        "prediction": "L1", "confidence": 0.8,
        "scores": {"L1": 0.8, "L2": 0.2, "fiber": 0.0},
        "graph_paths": paths,
    }
    second = {
        "prediction": "L1", "confidence": 0.8,
        "scores": {"L1": 0.8, "L2": 0.2, "fiber": 0.0},
        "matched_rules": {"L1": [], "L2": [], "fiber": []},
    }
    result = fuse_results(case, first, second)
    assert result["decision_status"] == "agreement"
    assert [item["detail"]["anomaly_id"] for item in result["supporting_evidence"]] == ["a"]
    assert [item["detail"]["anomaly_id"] for item in result["conflicting_evidence"]] == ["b"]
