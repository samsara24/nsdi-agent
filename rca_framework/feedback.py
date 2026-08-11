"""N8 evidence-graph feedback helpers."""

from __future__ import annotations

from typing import Any, Dict, Sequence

from .branches.base import BranchOutcome
from .decision import FinalDecision
from .evidence_graph.store import (
    CaseDiagnosis,
    DiagnosisEdge,
    DiagnosisNode,
    EvidenceGraph,
)
from .evidence_pack import EvidencePack
from .features.extractor import CaseFeatures


FEEDBACK_SCHEMA = "rca-feedback-v1"


def build_case_diagnosis(
    pack: EvidencePack,
    features: CaseFeatures,
    outcome: BranchOutcome,
    decision: FinalDecision,
    *,
    sop_version: str,
    constraint_library_version: str,
    confirmed_by: str = "",
    confirmed_label: str = "",
) -> CaseDiagnosis:
    outcome_attrs = decision.to_dict()
    if confirmed_label:
        outcome_attrs["confirmed_label"] = confirmed_label
        outcome_attrs["prediction_matches_confirmation"] = decision.verdict == confirmed_label
    nodes = [
        DiagnosisNode("case", "Case", {"case_id": pack.case_id, "telemetry_status": pack.telemetry_status}),
        DiagnosisNode("outcome", "Outcome", outcome_attrs),
    ]
    edges = [DiagnosisEdge("case", "outcome", "concludes")]
    for index, token in enumerate(features.tokens):
        node_id = f"feature:{index}"
        nodes.append(DiagnosisNode(node_id, "FeatureToken", {"token": token}))
        edges.append(DiagnosisEdge("case", node_id, "has_token"))
    for index, link in enumerate(outcome.evidence_chain):
        node_id = f"step:{index}"
        nodes.append(DiagnosisNode(node_id, "SOPStep", link.to_dict()))
        edges.append(DiagnosisEdge("case", node_id, "has_step"))
        edges.append(DiagnosisEdge(node_id, "outcome", "supports_decision"))
    return CaseDiagnosis(
        case_id=pack.case_id,
        sop_version=sop_version,
        constraint_library_version=constraint_library_version,
        nodes=tuple(nodes),
        edges=tuple(edges),
        confirmed_by=confirmed_by,
    )


def apply_confirmed_feedback(
    graph: EvidenceGraph,
    diagnoses: Sequence[CaseDiagnosis],
    *,
    confirmed_by: str,
) -> EvidenceGraph:
    if not confirmed_by:
        raise ValueError("confirmed_by is required for feedback graph updates")
    normalized = [
        CaseDiagnosis(
            case_id=item.case_id,
            sop_version=item.sop_version,
            constraint_library_version=item.constraint_library_version,
            nodes=item.nodes,
            edges=item.edges,
            confirmed_by=confirmed_by,
        )
        for item in diagnoses
    ]
    return graph.with_case_diagnoses(normalized)


def feedback_manifest(graph: EvidenceGraph) -> Dict[str, Any]:
    return {
        "schema_version": FEEDBACK_SCHEMA,
        "graph_version": graph.version,
        "diagnosis_count": len(graph.case_diagnoses),
        "diagnoses": [item.to_dict() for item in graph.case_diagnoses],
    }
