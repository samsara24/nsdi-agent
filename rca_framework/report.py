"""N7 report rendering for RCA v2 outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

from .branches.base import BranchOutcome
from .decision import FinalDecision
from .evidence_graph.store import CaseDiagnosis


REPORT_SCHEMA = "rca-report-v1"


@dataclass(frozen=True)
class RCAReport:
    case_id: str
    action: str
    verdict: str | None
    proposed_verdict: str | None
    confidence: float
    confidence_lower_bound: float
    branch: str
    evidence_chain: Sequence[Dict[str, Any]]
    requested_evidence: Sequence[str]
    caveats: Sequence[str]
    diagnosis_graph: Dict[str, Any] | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": REPORT_SCHEMA,
            "case_id": self.case_id,
            "action": self.action,
            "verdict": self.verdict,
            "proposed_verdict": self.proposed_verdict,
            "confidence": self.confidence,
            "confidence_lower_bound": self.confidence_lower_bound,
            "branch": self.branch,
            "evidence_chain": list(self.evidence_chain),
            "requested_evidence": list(self.requested_evidence),
            "caveats": list(self.caveats),
            "diagnosis_graph": self.diagnosis_graph,
        }


def build_report(
    outcome: BranchOutcome,
    decision: FinalDecision,
    *,
    diagnosis: CaseDiagnosis | None = None,
) -> RCAReport:
    return RCAReport(
        case_id=outcome.case_id,
        action=decision.action,
        verdict=decision.verdict,
        proposed_verdict=decision.proposed_verdict,
        confidence=decision.confidence,
        confidence_lower_bound=decision.confidence_lower_bound,
        branch=outcome.branch,
        evidence_chain=[item.to_dict() for item in outcome.evidence_chain],
        requested_evidence=decision.requested_evidence,
        caveats=outcome.caveats,
        diagnosis_graph=diagnosis.to_dict() if diagnosis is not None else None,
    )


def render_markdown(report: RCAReport) -> str:
    verdict = report.verdict or "未形成最终结论"
    lines = [
        f"# RCA 报告：{report.case_id}",
        "",
        f"- 出口动作：`{report.action}`",
        f"- 最终根因：`{verdict}`",
        f"- 候选根因：`{report.proposed_verdict or '无'}`",
        f"- 分支来源：`{report.branch}`",
        f"- 置信度：{report.confidence:.2%}（Wilson 下界 {report.confidence_lower_bound:.2%}）",
        "",
        "## 证据链",
    ]
    for index, item in enumerate(report.evidence_chain, start=1):
        lines.append(f"{index}. [{item.get('kind')}] {item.get('statement')}")
    if report.requested_evidence:
        lines += ["", "## 需要补采", *[f"- `{item}`" for item in report.requested_evidence]]
    if report.caveats:
        lines += ["", "## 注意事项", *[f"- {item}" for item in report.caveats]]
    return "\n".join(lines) + "\n"
