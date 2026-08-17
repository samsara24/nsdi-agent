from __future__ import annotations

from rca_framework.branches.base import BranchOutcome, EvidenceLink
from rca_framework.decision import FinalDecision
from rca_framework.evidence_graph import EVIDENCE_GRAPH_V2_SCHEMA, EvidenceGraph, GraphCase
from rca_framework.evidence_pack import EvidencePack
from rca_framework.features.extractor import CaseFeatures
from rca_framework.feedback import apply_confirmed_feedback, build_case_diagnosis, feedback_manifest
from rca_framework.report import build_report, render_markdown


def _outcome() -> BranchOutcome:
    return BranchOutcome(
        case_id="case-x",
        branch="N5c",
        verdict="L2",
        confidence=0.8,
        confidence_lower_bound=0.6,
        calibration_group="sop:leaf",
        calibration_support=12,
        evidence_chain=(
            EvidenceLink(kind="learned_sop", statement="leaf predicts L2", tokens=("present:a",), source="learned-sop-v1"),
            EvidenceLink(kind="constraint_exclusion", statement="fiber excluded", tokens=("present:a",), source="P5_tx_down_excludes_medium"),
        ),
    )


def _decision() -> FinalDecision:
    return FinalDecision(
        case_id="case-x",
        branch="N5c",
        action="final",
        verdict="L2",
        proposed_verdict="L2",
        confidence=0.8,
        confidence_lower_bound=0.6,
        calibration_group="sop:leaf",
        calibration_support=12,
        reason="passes gate",
    )


def test_report_renders_evidence_chain():
    report = build_report(_outcome(), _decision())
    payload = report.to_dict()
    assert payload["schema_version"] == "rca-report-v1"
    assert payload["verdict"] == "L2"
    assert payload["evidence_chain"][0]["kind"] == "learned_sop"
    markdown = render_markdown(report)
    assert "RCA 报告" in markdown
    assert "leaf predicts L2" in markdown


def test_feedback_adds_case_diagnosis_graph():
    pack = EvidencePack.from_case({"case_id": "case-x", "rxpower": {"L1": {"0": -40.0}}})
    features = CaseFeatures(
        case_id="case-x",
        tokens=("drop:L1:rxpower:single_lane",),
        by_family={"signal_drop": ("drop:L1:rxpower:single_lane",)},
        dictionary_version="feature-dictionary-v2",
        dictionary_hash="hash",
        telemetry_status=pack.telemetry_status,
    )
    diagnosis = build_case_diagnosis(
        pack,
        features,
        _outcome(),
        _decision(),
        sop_version="learned-sop-v1",
        constraint_library_version="constraint-library-v5",
    )
    node_types = {node.node_type for node in diagnosis.nodes}
    edge_types = {edge.edge_type for edge in diagnosis.edges}
    assert "ConstraintCheck" in node_types
    assert "precedes" in edge_types
    assert "checked_by" in edge_types
    graph = EvidenceGraph(cases=(GraphCase("case-x", "L2", features.tokens),), dictionary_hash="hash")
    updated = apply_confirmed_feedback(graph, [diagnosis], confirmed_by="operator-a")
    assert updated.schema_version == EVIDENCE_GRAPH_V2_SCHEMA
    assert updated.case_diagnoses[0].confirmed_by == "operator-a"
    manifest = feedback_manifest(updated)
    assert manifest["diagnosis_count"] == 1
