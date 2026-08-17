"""N8 evidence-graph feedback helpers."""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

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
    feature_node_by_token: Dict[str, str] = {}
    for index, token in enumerate(features.tokens):
        node_id = f"feature:{index}"
        feature_node_by_token[token] = node_id
        nodes.append(DiagnosisNode(node_id, "FeatureToken", {"token": token}))
        edges.append(DiagnosisEdge("case", node_id, "has_token"))
    previous_step_id = ""
    for index, link in enumerate(outcome.evidence_chain):
        node_id = f"step:{index}"
        nodes.append(DiagnosisNode(node_id, "SOPStep", link.to_dict()))
        edges.append(DiagnosisEdge("case", node_id, "has_step"))
        edges.append(DiagnosisEdge(node_id, "outcome", "supports_decision"))
        if previous_step_id:
            edges.append(DiagnosisEdge(previous_step_id, node_id, "precedes"))
        previous_step_id = node_id
        for token in link.tokens:
            feature_node = feature_node_by_token.get(token)
            if feature_node is not None:
                edges.append(DiagnosisEdge(node_id, feature_node, "uses_token"))
        for constraint_id in _constraint_ids(link.source):
            check_id = f"constraint:{index}:{constraint_id}"
            nodes.append(
                DiagnosisNode(
                    check_id,
                    "ConstraintCheck",
                    {
                        "constraint_id": constraint_id,
                        "step_id": node_id,
                        "evidence_link_kind": link.kind,
                    },
                )
            )
            edges.append(DiagnosisEdge(node_id, check_id, "checked_by"))
            edges.append(DiagnosisEdge(check_id, "outcome", "constrains_decision"))
    return CaseDiagnosis(
        case_id=pack.case_id,
        sop_version=sop_version,
        constraint_library_version=constraint_library_version,
        nodes=tuple(nodes),
        edges=tuple(edges),
        confirmed_by=confirmed_by,
    )


def _constraint_ids(source: str) -> Tuple[str, ...]:
    ids = []
    for item in source.split("|"):
        item = item.strip()
        if item.startswith(("C", "P", "M")) and "_" in item:
            ids.append(item)
    return tuple(ids)


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
