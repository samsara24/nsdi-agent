from __future__ import annotations

from pathlib import Path

from rca_framework.html_report import render_experiment_html


POLICY = "coverage-v2"


def _inputs():
    summary = {
        "graph_version": "evidence-graph-v2:test",
        "policies": {
            POLICY: {
                "routing": {"counts": {"N5c": 2, "N6": 1}},
                "branches": {
                    "N5c": {
                        "n": 2,
                        "answered": 1,
                        "correct": 0,
                        "precision_when_answered": 0.0,
                        "needs_llm": 1,
                        "needs_human": 0,
                    },
                    "N6": {
                        "n": 1,
                        "answered": 0,
                        "correct": 0,
                        "precision_when_answered": None,
                        "needs_llm": 0,
                        "needs_human": 1,
                    },
                },
                "final_decisions": {
                    "answered": 1,
                    "correct": 0,
                    "coverage": 1 / 3,
                    "precision_when_answered": 0.0,
                    "actions": {"final": 1, "request_evidence": 1, "human_review": 1},
                    "class_metrics": {
                        "L1": {
                            "support": 1,
                            "predicted": 0,
                            "true_positive": 0,
                            "precision": 0.0,
                            "recall": 0.0,
                            "f1": 0.0,
                        },
                        "L2": {
                            "support": 1,
                            "predicted": 1,
                            "true_positive": 0,
                            "precision": 0.0,
                            "recall": 0.0,
                            "f1": 0.0,
                        },
                        "fiber": {
                            "support": 1,
                            "predicted": 0,
                            "true_positive": 0,
                            "precision": 0.0,
                            "recall": 0.0,
                            "f1": 0.0,
                        },
                    },
                },
            }
        },
    }
    manifest = {
        "schema_version": "agentic-rca-run-manifest-v1",
        "created_at_utc": "2026-08-11T00:00:00Z",
        "data": {"data_dir": "dataset<&>", "train_size": 10, "test_size": 3},
        "versions": {
            "evidence_graph": "graph-v2",
            "feature_dictionary": "feature-v2",
            "constraint_library": "constraints-v3",
            "sop": "learned-sop-v1",
            "prompt_template": "prompt-v3",
            "decision_policy": "decision-v1",
        },
    }
    outcomes = {
        POLICY: [
            {
                "case_id": "case-wrong",
                "actual": "L1",
                "match": {
                    "max_similarity": 0.75,
                    "evidence_coverage": 0.8,
                    "retrieved_candidate_count": 1,
                    "candidates": [
                        {
                            "case_id": "history-1",
                            "label": "L2",
                            "similarity": 0.75,
                            "evidence_coverage": 0.8,
                            "overlap": ["status:L1:RxLOS"],
                            "missing_evidence": ["host_snr"],
                            "conflicting_evidence": [],
                        }
                    ],
                },
                "routing": {"branch": "N5c", "reason": "low similarity"},
                "branch_outcome": {
                    "branch": "N5c",
                    "verdict": "L2",
                    "confidence": 0.8,
                    "confidence_lower_bound": 0.6,
                    "calibration_group": "llm:N5c",
                    "calibration_support": 20,
                    "missing_evidence": ["host_snr"],
                    "evidence_chain": [
                        {
                            "kind": "physical_constraint",
                            "statement": "C8 supports medium <unsafe>",
                            "tokens": ["status:L1:RxLOS"],
                            "source": "constraint-library-v3",
                        },
                        {
                            "kind": "learned_sop",
                            "statement": "leaf predicts L2",
                            "tokens": ["status:L1:RxLOS"],
                            "source": "learned-sop-v1",
                        },
                    ],
                },
                "final_decision": {
                    "action": "final",
                    "verdict": "L2",
                    "proposed_verdict": "L2",
                    "confidence": 0.8,
                    "confidence_lower_bound": 0.6,
                    "calibration_group": "llm:N5c",
                    "calibration_support": 20,
                    "reason": "passes <script>alert('m9')</script>",
                    "requested_evidence": [],
                },
                "features": {
                    "tokens": ["status:L1:RxLOS", "<script>alert('token')</script>"],
                    "by_family": {"status_fault": ["status:L1:RxLOS"]},
                    "telemetry_status": "full_telemetry",
                },
                "evidence_pack": {
                    "telemetry_status": "full_telemetry",
                    "field_states": {"L1.host_snr": "missing"},
                },
                "sop_prediction": {
                    "leaf_id": "leaf-2",
                    "prediction": "L2",
                    "support": 15,
                    "wilson_lower_bound": 0.55,
                },
                "diagnosis_graph": {
                    "case_id": "case-wrong",
                    "sop_version": "learned-sop-v1",
                    "constraint_library_version": "constraint-library-v3",
                    "nodes": [
                        {
                            "id": "feature:1",
                            "type": "FeatureToken",
                            "attrs": {"token": "status:L1:RxLOS"},
                        }
                    ],
                    "edges": [
                        {"src": "feature:1", "dst": "outcome:L2", "type": "supports", "attrs": {}}
                    ],
                },
                "report": {"caveats": ["<b>untrusted</b>"]},
                "trace_id": "case-wrong",
            },
            {
                "case_id": "case-no-trace",
                "actual": "fiber",
                "match": {},
                "routing": {"branch": "N5c", "reason": "missing evidence"},
                "branch_outcome": {
                    "branch": "N5c",
                    "verdict": None,
                    "missing_evidence": ["L2.host_snr"],
                },
                "final_decision": {
                    "action": "request_evidence",
                    "verdict": None,
                    "proposed_verdict": None,
                    "reason": "需要补采",
                    "requested_evidence": ["L2.host_snr"],
                },
                "report": {},
                "trace_id": None,
            },
            {
                "case_id": "case-telemetry",
                "actual": "L2",
                "routing": {"branch": "N6", "reason": "optical blackout"},
                "branch_outcome": {"branch": "N6", "verdict": None},
                "final_decision": {
                    "action": "human_review",
                    "verdict": None,
                    "reason": "遥测失效，转人工",
                },
                "features": {"telemetry_status": "no_telemetry", "tokens": []},
                "trace_id": None,
            },
        ]
    }
    traces = {
        POLICY: {
            "case-wrong": {
                "case_id": "case-wrong",
                "backend": "scripted",
                "prompt_version": "prompt-v3",
                "constraint_library_version": "constraint-library-v3",
                "attempt_count": 1,
                "rewrote": False,
                "attempts": [
                    {
                        "index": 0,
                        "prompt": "diagnose <case>",
                        "raw_output": '{"verdict":"L2","note":"<raw>"}',
                        "parsed": True,
                        "check": {
                            "violations": [
                                {
                                    "kind": "forbidden_claim",
                                    "severity": "warning",
                                    "message": "unsafe <claim>",
                                }
                            ]
                        },
                    }
                ],
                "accepted": {"verdict": "L2"},
                "abstain_reason": "",
            }
        }
    }
    return summary, manifest, outcomes, traces


def test_render_writes_index_and_one_page_per_case(tmp_path):
    result = render_experiment_html(tmp_path, *_inputs(), training_summary={"cases": 10})

    assert result["schema_version"] == "rca-experiment-html-v1"
    assert result["case_count"] == 3
    assert result["file_count"] == 4
    assert result["policies"] == {POLICY: 3}
    assert {path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*.html")} == {
        "index.html",
        "cases/coverage-v2-case-wrong.html",
        "cases/coverage-v2-case-no-trace.html",
        "cases/coverage-v2-case-telemetry.html",
    }


def test_index_groups_cases_and_escapes_untrusted_values(tmp_path):
    render_experiment_html(tmp_path, *_inputs())
    index = (tmp_path / "index.html").read_text(encoding="utf-8")
    case = (tmp_path / "cases/coverage-v2-case-wrong.html").read_text(encoding="utf-8")

    assert "实验深度分析" in index
    assert "候选质量（M9 降级前）" in index
    assert "SOP 对照" in index
    assert "LLM 校验失败构成" in index
    assert "LLM 高频失败原因" in index
    assert "分类指标" in index
    assert "路由分布" in index
    assert "模型答错" in index
    assert "弃权/补采" in index
    assert "遥测不足" in index
    assert "dataset&lt;&amp;&gt;" in index
    assert "<script>alert('m9')</script>" not in case
    assert "&lt;script&gt;alert(&#x27;m9&#x27;)&lt;/script&gt;" in case
    assert "<script>alert('token')</script>" not in case
    assert "&lt;b&gt;untrusted&lt;/b&gt;" in case


def test_case_page_contains_deep_audit_sections_and_handles_missing_trace(tmp_path):
    render_experiment_html(tmp_path, *_inputs())
    detailed = (tmp_path / "cases/coverage-v2-case-wrong.html").read_text(encoding="utf-8")
    no_trace = (tmp_path / "cases/coverage-v2-case-no-trace.html").read_text(encoding="utf-8")

    for heading in (
        "结论、真值与路由",
        "特征 Token",
        "历史候选",
        "SOP 预测",
        "物理约束与证据链",
        "M9 决策原因",
        "缺失证据",
        "LLM 逐轮推理",
        "Prompt",
        "Raw 输出",
        "校验违规",
        "诊断图",
    ):
        assert heading in detailed
    assert "forbidden_claim" in detailed
    assert "无 trace：该 case 未调用 LLM" in no_trace
