"""M8 受约束推理循环：生成 -> 逐步校验 -> 不合规则重写 -> 仍不合规则弃权。

这是 T6 验收「LLM 每步输出可被约束校验；不合规可回退或重写」的落点。

循环的终止条件刻意设计成**弃权而不是接受**：重写 `max_attempts` 次仍不合规时，
返回一个 `verdict=None` 的结果并附上全部违规记录，而不是退而求其次接受最后一次输出。
理由是被判为 fatal 的违规里，最常见的是引用了不存在的证据——
一个建立在虚构证据上的结论，比没有结论更有害。

每一轮的 prompt、原始输出、校验报告都记入 `ReasoningTrace`，
这就是画板要求的「逐步推理日志」，可以直接进报告与论文附录。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..constraints.checker import CheckReport, Violation, check_evidence, check_response
from ..constraints.library import CONSTRAINT_LIBRARY, ConstraintLibrary
from ..evidence_pack import EvidencePack
from .backend import Backend, NoneBackend
from .prompts import PROMPT_TEMPLATE_VERSION, build_prompt
from .protocol import DiagnosisResponse, parse_response


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
            "evidence_check": self.evidence_check.to_dict() if self.evidence_check else None,
        }


@dataclass
class ConstrainedReasoner:
    """在物理约束内做推理，并强制每一步可校验。"""

    backend: Backend = field(default_factory=NoneBackend)
    library: ConstraintLibrary = CONSTRAINT_LIBRARY
    max_attempts: int = 2

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
        feedback: List[str] = ["" for _ in requests]
        pending = list(range(len(requests)))

        for round_index in range(max(1, self.max_attempts)):
            if not pending:
                break
            prompts = [
                build_prompt(requests[i], library=self.library, retry_feedback=feedback[i])
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
                attempts[i].append(Attempt(
                    index=round_index, prompt=prompts[position], raw_output=raw,
                    parsed=response is not None, check=report,
                ))
                if response is not None and report.ok:
                    accepted[i] = response
                else:
                    feedback[i] = report.feedback()
                    still_pending.append(i)
            pending = still_pending

        return [
            ReasoningTrace(
                case_id=requests[i].case_id,
                attempts=tuple(attempts[i]),
                accepted=accepted[i],
                evidence_check=evidence_checks[i],
                backend_name=self.backend.name,
                constraint_library_version=self.library.version,
                abstain_reason=_abstain_reason(accepted[i], attempts[i]),
            )
            for i in range(len(requests))
        ]


def _abstain_reason(accepted: Optional[DiagnosisResponse], attempts: Sequence[Attempt]) -> str:
    if accepted is not None:
        return "" if accepted.verdict is not None else "模型主动弃权：证据不足以支撑任何根因"
    if not attempts:
        return "未进行任何生成"
    last = attempts[-1]
    if not last.parsed:
        return f"{len(attempts)} 次生成均未产出可校验的结构化输出"
    return (
        f"{len(attempts)} 次生成均未通过物理约束校验，最后一次的问题："
        + "；".join(item.message for item in last.check.fatal)
    )
