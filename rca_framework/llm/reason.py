"""M8 受约束推理循环：生成 -> 逐步校验 -> 不合规则重写 -> 强制产出低置信结论。

这是 T6 验收「LLM 每步输出可被约束校验；不合规可回退或重写」的落点。

循环的终止条件服务于 N6 阈值标定：每个 case 都必须产出 L1/L2/fiber 三分类候选。
fatal 仍会触发重写；最后仍未通过时，把最后一次可解析结论降为极低置信候选，
完全解析失败时再用历史多数/训练先验兜底，并在 trace 中单列 fallback_source。

每一轮的 prompt、原始输出、校验报告都记入 `ReasoningTrace`，
这就是画板要求的「逐步推理日志」，可以直接进报告与论文附录。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..constraints.checker import CheckReport, Violation, check_evidence, check_response
from ..constraints.library import CONSTRAINT_LIBRARY, ConstraintLibrary
from ..evidence_pack import EvidencePack
from ..types import ROOT_CAUSES
from .backend import Backend, NoneBackend
from .prompts import PROMPT_TEMPLATE_VERSION, build_prompt, prompt_template_version_for
from .protocol import ConfidenceBreakdown, DiagnosisResponse, parse_response


#: 校验器零违约时给 physical_compliance 的分数，对应 rubric 的「推理链与约束高度一致」。
_CLEAN_COMPLIANCE_SCORE = 0.9
#: verdict 与推理链矛盾时 reasoning_completeness 的上限，对应 rubric 的「单步弱判断」。
_MISMATCH_REASONING_CAP = 0.3
_FIBER_FIELD_EVIDENCE_CONTRACT = "M6_fiber_not_identifiable_without_field_evidence"
#: fiber 被降级且推理链没给出端点倾向时的默认动作。M5 允许群体统计决定默认动作，
#: 但不允许把它写成 support 步骤的证据。
_MAJORITY_CLASS_DEFAULT = "L2"


@dataclass(frozen=True)
class Attempt:
    """一轮生成的完整记录。"""

    index: int
    prompt: str
    raw_output: str
    parsed: bool
    check: CheckReport

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "prompt": self.prompt,
            "raw_output": self.raw_output,
            "parsed": self.parsed,
            "check": self.check.to_dict(),
        }


@dataclass(frozen=True)
class ReasoningTrace:
    """逐步推理日志。`accepted` 为 None 表示最终弃权。"""

    case_id: str
    attempts: Tuple[Attempt, ...] = ()
    accepted: Optional[DiagnosisResponse] = None
    evidence_check: Optional[CheckReport] = None
    backend_name: str = ""
    prompt_version: str = PROMPT_TEMPLATE_VERSION
    constraint_library_version: str = ""
    abstain_reason: str = ""
    degradation_reason: str = ""
    fallback_source: str = ""

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)

    @property
    def rewrote(self) -> bool:
        return self.attempt_count > 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "backend": self.backend_name,
            "prompt_version": self.prompt_version,
            "constraint_library_version": self.constraint_library_version,
            "attempt_count": self.attempt_count,
            "rewrote": self.rewrote,
            "attempts": [item.to_dict() for item in self.attempts],
            "accepted": self.accepted.to_dict() if self.accepted else None,
            "abstain_reason": self.abstain_reason,
            "degradation_reason": self.degradation_reason or self.abstain_reason,
            "fallback_source": self.fallback_source,
            "evidence_check": self.evidence_check.to_dict() if self.evidence_check else None,
        }


@dataclass
class ConstrainedReasoner:
    """在物理约束内做推理，并强制每一步可校验。"""

    backend: Backend = field(default_factory=NoneBackend)
    library: ConstraintLibrary = CONSTRAINT_LIBRARY
    max_attempts: int = 3

    def reason(self, request: Any, pack: EvidencePack) -> ReasoningTrace:
        return self.reason_many([request], [pack])[0]

    def reason_many(
        self,
        requests: Sequence[Any],
        packs: Sequence[EvidencePack],
    ) -> List[ReasoningTrace]:
        """批量推理。重写只针对**尚未通过**的 case 重新发一批，已通过的不再消耗算力。"""
        if len(requests) != len(packs):
            raise ValueError("requests and packs must be the same length")

        evidence_checks = [check_evidence(pack, library=self.library) for pack in packs]
        attempts: List[List[Attempt]] = [[] for _ in requests]
        accepted: List[Optional[DiagnosisResponse]] = [None for _ in requests]
        last_responses: List[Optional[DiagnosisResponse]] = [None for _ in requests]
        last_reports: List[Optional[CheckReport]] = [None for _ in requests]
        feedback: List[str] = ["" for _ in requests]
        pending = list(range(len(requests)))

        for round_index in range(max(1, self.max_attempts)):
            if not pending:
                break
            prompts = [
                build_prompt(
                    requests[i],
                    library=self.library,
                    retry_feedback=_round_feedback(feedback[i], round_index, self.max_attempts),
                )
                for i in pending
            ]
            outputs = self.backend.generate(prompts)
            still_pending: List[int] = []
            for position, i in enumerate(pending):
                raw = outputs[position] if position < len(outputs) else ""
                response = parse_response(raw)
                if response is None:
                    report = CheckReport(violations=(
                        Violation(
                            kind="unsupported_step", severity="fatal",
                            message="输出不是符合 schema 的 JSON，无法逐步校验",
                            detail=raw[:200],
                        ),
                    ))
                else:
                    report = check_response(
                        response,
                        packs[i],
                        requests[i].evidence_tokens,
                        allowed_root_causes=requests[i].candidate_root_causes,
                        library=self.library,
                    )
                    last_responses[i] = response
                attempts[i].append(Attempt(
                    index=round_index, prompt=prompts[position], raw_output=raw,
                    parsed=response is not None, check=report,
                ))
                last_reports[i] = report
                if response is not None and report.ok:
                    reconciled = _apply_fiber_gate(
                        _reconcile_verdict_with_steps(response), report
                    )
                    accepted[i] = _objective_confidence(
                        _apply_compliance_penalties(reconciled, report),
                        report,
                        requests[i],
                    )
                else:
                    feedback[i] = report.feedback()
                    still_pending.append(i)
            pending = still_pending

        for i, response in enumerate(accepted):
            if response is None:
                accepted[i] = _forced_response(
                    requests[i],
                    last_responses[i],
                    last_reports[i],
                    attempts[i],
                )

        return [
            ReasoningTrace(
                case_id=requests[i].case_id,
                attempts=tuple(attempts[i]),
                accepted=accepted[i],
                evidence_check=evidence_checks[i],
                backend_name=self.backend.name,
                prompt_version=prompt_template_version_for(requests[i]),
                constraint_library_version=self.library.version,
                abstain_reason=_degradation_reason(accepted[i], attempts[i]),
                degradation_reason=_degradation_reason(accepted[i], attempts[i]),
                fallback_source=accepted[i].fallback_source if accepted[i] else "",
            )
            for i in range(len(requests))
        ]


def _round_feedback(feedback: str, round_index: int, max_attempts: int) -> str:
    # The first request has no prior answer. Never label the first/only round as
    # a rewrite merely because it is also the configured final round.
    if round_index == 0:
        return ""
    if round_index < max(1, max_attempts) - 1:
        return feedback
    final_hint = (
        "这是最后一轮，请必须输出符合 schema 的 JSON，并在 L1/L2/fiber 中三选一。"
        "如果证据不足，不要拒答，把低把握体现在 confidence_breakdown。"
    )
    return (feedback + "\n" + final_hint).strip() if feedback else final_hint


def _apply_compliance_penalties(
    response: DiagnosisResponse,
    report: CheckReport,
) -> DiagnosisResponse:
    if not report.veto:
        return response
    breakdown = response.confidence_breakdown.with_physical_cap(report.physical_compliance_cap)
    return response.with_adjustments(
        confidence_breakdown=breakdown,
        compliance_penalties=report.compliance_penalties,
    )


def _apply_fiber_gate(
    response: DiagnosisResponse,
    report: Optional[CheckReport],
) -> DiagnosisResponse:
    """缺少双向现场证据时，fiber 只能当候选，不能当结论。

    上一轮把 M6 从否决降级成扣分之后，fiber 变成了证据不足时的默认出口：
    107 例里判了 23 次而真值只有 8 条。这里恢复准入门槛，但保留 fiber 作为
    带低合规分的候选记录在 `compliance_penalties` 里，仍然可被人工复核看到。
    """
    if response.verdict != "fiber" or report is None:
        return response
    if not any(item.constraint_id == _FIBER_FIELD_EVIDENCE_CONTRACT for item in report.veto):
        return response
    votes = response.step_votes()
    endpoint_votes = {name: votes.get(name, 0) for name in ("L1", "L2") if name in votes}
    if endpoint_votes:
        best = max(endpoint_votes.values())
        winners = sorted(name for name, count in endpoint_votes.items() if count == best)
        endpoint = winners[0] if len(winners) == 1 else _MAJORITY_CLASS_DEFAULT
    else:
        endpoint = _MAJORITY_CLASS_DEFAULT
    penalties = list(response.compliance_penalties)
    penalties.append({
        "constraint_id": _FIBER_FIELD_EVIDENCE_CONTRACT,
        "kind": "fiber_downgraded_to_candidate",
        "message": (
            f"缺少同 lane 双向对称丢失证据，fiber 降为候选，结论改判 {endpoint}；"
            f"需补采 OTDR / 端面镜检 / 双向功率标定后复核"
        ),
        "physical_compliance_cap": 0.3,
        "step_index": None,
    })
    return response.with_adjustments(
        verdict=endpoint,
        compliance_penalties=penalties,
    )


def _objective_confidence(
    response: DiagnosisResponse,
    report: Optional[CheckReport],
    request: Any,
) -> DiagnosisResponse:
    """把能客观算出的置信度维度改写为代码侧结果。

    模型自评这两维几乎没有判别力：它倾向于给 rubric 的中间锚点。
    `physical_compliance` 由校验器的违约情况决定，`history_similarity`
    直接取检索到的最高相似度。
    """
    if report is None:
        compliance = response.confidence_breakdown.physical_compliance
    elif report.fatal:
        compliance = 0.0
    else:
        compliance = min(_CLEAN_COMPLIANCE_SCORE, report.physical_compliance_cap)
    similarity = getattr(request, "nearest_similarity", None)
    breakdown = response.confidence_breakdown.with_overrides(
        physical_compliance=compliance,
        history_similarity=None if similarity is None else float(similarity),
    )
    return response.with_adjustments(confidence_breakdown=breakdown)


def _reconcile_verdict_with_steps(response: DiagnosisResponse) -> DiagnosisResponse:
    """结论必须站在自己的推理链上。

    实测有约五分之一的回答，`verdict` 与 steps 的 support 汇总指向不同的根因，
    而这些 case 里推理链的正确率明显高于 verdict。此处以推理链为准改写结论，
    并折减 `reasoning_completeness`，让门禁能看到这次自相矛盾。
    """
    majority = response.step_majority()
    if majority is None or majority == response.verdict:
        return response
    penalties = list(response.compliance_penalties)
    penalties.append({
        "constraint_id": "",
        "kind": "verdict_step_mismatch",
        "message": (
            f"verdict={response.verdict} 与推理链汇总 {majority} 不一致，"
            f"已按推理链改写结论"
        ),
        "physical_compliance_cap": 1.0,
        "step_index": None,
    })
    breakdown = response.confidence_breakdown.with_overrides(
        reasoning_completeness=min(
            response.confidence_breakdown.reasoning_completeness,
            _MISMATCH_REASONING_CAP,
        ),
    )
    return response.with_adjustments(
        verdict=majority,
        confidence_breakdown=breakdown,
        compliance_penalties=penalties,
    )


def _forced_response(
    request: Any,
    last_response: Optional[DiagnosisResponse],
    last_report: Optional[CheckReport],
    attempts: Sequence[Attempt],
) -> DiagnosisResponse:
    if last_response is not None:
        penalties = list(last_report.compliance_penalties if last_report is not None else ())
        penalties.extend(
            {
                "constraint_id": item.constraint_id,
                "kind": item.kind,
                "message": item.message,
                "physical_compliance_cap": 0.0,
                "step_index": item.step_index,
            }
            for item in (last_report.fatal if last_report is not None else ())
        )
        reconciled = _apply_fiber_gate(
            _reconcile_verdict_with_steps(last_response), last_report
        )
        similarity = getattr(request, "nearest_similarity", None)
        breakdown = reconciled.confidence_breakdown.with_overrides(
            physical_compliance=0.0,
            history_similarity=None if similarity is None else float(similarity),
        )
        return reconciled.with_adjustments(
            confidence_breakdown=breakdown,
            forced=True,
            fallback_source="last_parsed_after_fatal",
            compliance_penalties=penalties,
        )
    fallback = _fallback_verdict(request)
    reason = _degradation_reason(None, attempts)
    return DiagnosisResponse(
        steps=(),
        verdict=fallback,
        confidence=0.0,
        confidence_breakdown=ConfidenceBreakdown(),
        self_reported_confidence=0.0,
        missing_information=("LLM 未产出可解析结构化输出，需要人工复核原始日志",),
        raw_output=attempts[-1].raw_output if attempts else "",
        forced=True,
        fallback_source="parse_failure",
        compliance_penalties=(
            {
                "constraint_id": "",
                "kind": "unsupported_step",
                "message": reason,
                "physical_compliance_cap": 0.0,
                "step_index": None,
            },
        ),
    )


def _fallback_verdict(request: Any) -> str:
    distribution = dict(getattr(request, "historical_label_distribution", ()) or ())
    if distribution:
        return max(
            distribution.items(),
            key=lambda item: (
                item[1],
                -ROOT_CAUSES.index(item[0]) if item[0] in ROOT_CAUSES else -len(ROOT_CAUSES),
            ),
        )[0]
    candidates = tuple(getattr(request, "candidate_root_causes", ()) or ())
    if candidates:
        return candidates[0]
    return "L2"


def _degradation_reason(accepted: Optional[DiagnosisResponse], attempts: Sequence[Attempt]) -> str:
    if accepted is not None:
        if accepted.fallback_source:
            return f"强制低置信兜底：{accepted.fallback_source}"
        if accepted.compliance_penalties:
            return "结论触发物理/量测 veto，已折减 physical_compliance"
        return ""
    if not attempts:
        return "未进行任何生成"
    last = attempts[-1]
    if not last.parsed:
        return f"{len(attempts)} 次生成均未产出可校验的结构化输出"
    return (
        f"{len(attempts)} 次生成均未通过物理约束校验，最后一次的问题："
        + "；".join(item.message for item in (last.check.fatal + last.check.veto))
    )
