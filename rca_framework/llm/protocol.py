"""M8 输出协议：把 LLM 的自由文本约束成可逐步校验的结构。

设计要点是**让每一步推理都可被单独校验**，而不是只校验最终结论。
legacy 的输出结构（`prediction` + 一段 `reasoning` 自由文本）做不到这一点：
文本里说了什么无法机械判定，因此约束校验器无从下手。

这里把推理拆成 `ReasoningStep` 序列，每一步必须声明：

- `claim`：这一步断言了什么。
- `cited_evidence`：这一步用到了哪些**证据 token**。必须是证据包里实际存在的 token，
  校验器会逐条比对。这是防幻觉最有效的一招——模型可以编造措辞，但编不出一个
  不在证据包里的 token 而不被发现。
- `cited_constraints`：这一步依据了哪几条物理约束。同样会比对约束库。
- `effect`：这一步对候选根因的作用（`support` / `exclude` / `neutral`）及作用对象。

`effect` 是结构化的，因此「这一步排除了 fiber」这种断言可以直接与约束库里的
排除条件对照，而不需要理解自然语言。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..types import ROOT_CAUSES


EFFECTS: Tuple[str, ...] = ("support", "exclude", "neutral")
CONFIDENCE_DIMENSIONS: Tuple[str, ...] = (
    "evidence_completeness",
    "physical_compliance",
    "reasoning_completeness",
    "history_similarity",
)
DEFAULT_CONFIDENCE_WEIGHTS: Dict[str, float] = {
    "evidence_completeness": 0.30,
    "physical_compliance": 0.30,
    "reasoning_completeness": 0.20,
    "history_similarity": 0.20,
}

#: 供 vLLM guided decoding 使用。字段与 `DiagnosisResponse` 一一对应。
DIAGNOSIS_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "sop_step_id": {"type": "string"},
                    "cited_predicates": {"type": "array", "items": {"type": "string"}},
                    "claim": {"type": "string"},
                    "cited_evidence": {"type": "array", "items": {"type": "string"}},
                    "cited_constraints": {"type": "array", "items": {"type": "string"}},
                    "effect": {"type": "string", "enum": list(EFFECTS)},
                    "target": {"type": "string", "enum": list(ROOT_CAUSES) + [""]},
                },
                # Optional for existing l2fixed/legacy-compatible callers.  The expanded
                # runner requires and validates it after parsing.
                "required": ["claim", "cited_evidence", "cited_constraints", "effect", "target"],
                "additionalProperties": False,
            },
        },
        "verdict": {"type": "string", "enum": list(ROOT_CAUSES)},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "confidence_breakdown": {
            "type": "object",
            "properties": {
                name: {"type": "number", "minimum": 0.0, "maximum": 1.0}
                for name in CONFIDENCE_DIMENSIONS
            },
            "required": list(CONFIDENCE_DIMENSIONS),
            "additionalProperties": False,
        },
        "missing_information": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["steps", "verdict", "confidence", "confidence_breakdown", "missing_information"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class ReasoningStep:
    claim: str
    sop_step_id: str = ""
    cited_predicates: Tuple[str, ...] = ()
    cited_evidence: Tuple[str, ...] = ()
    cited_constraints: Tuple[str, ...] = ()
    effect: str = "neutral"
    target: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sop_step_id": self.sop_step_id,
            "cited_predicates": list(self.cited_predicates),
            "claim": self.claim,
            "cited_evidence": list(self.cited_evidence),
            "cited_constraints": list(self.cited_constraints),
            "effect": self.effect,
            "target": self.target,
        }


def _clamp_score(value: Any) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


@dataclass(frozen=True)
class ConfidenceBreakdown:
    evidence_completeness: float = 0.0
    physical_compliance: float = 0.0
    reasoning_completeness: float = 0.0
    history_similarity: float = 0.0

    @classmethod
    def from_mapping(cls, value: Any) -> "ConfidenceBreakdown":
        if not isinstance(value, dict):
            return cls()
        return cls(
            evidence_completeness=_clamp_score(value.get("evidence_completeness", 0.0)),
            physical_compliance=_clamp_score(value.get("physical_compliance", 0.0)),
            reasoning_completeness=_clamp_score(value.get("reasoning_completeness", 0.0)),
            history_similarity=_clamp_score(value.get("history_similarity", 0.0)),
        )

    def weighted_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        selected = weights or DEFAULT_CONFIDENCE_WEIGHTS
        total_weight = sum(max(0.0, float(selected.get(name, 0.0))) for name in CONFIDENCE_DIMENSIONS)
        if total_weight <= 0.0:
            return 0.0
        weighted = sum(
            getattr(self, name) * max(0.0, float(selected.get(name, 0.0)))
            for name in CONFIDENCE_DIMENSIONS
        )
        return round(weighted / total_weight, 6)

    def with_physical_cap(self, cap: float) -> "ConfidenceBreakdown":
        return ConfidenceBreakdown(
            evidence_completeness=self.evidence_completeness,
            physical_compliance=min(self.physical_compliance, _clamp_score(cap)),
            reasoning_completeness=self.reasoning_completeness,
            history_similarity=self.history_similarity,
        )

    def with_overrides(self, **values: Optional[float]) -> "ConfidenceBreakdown":
        """用代码侧客观算得的分数覆盖模型自评的维度。

        模型自评会大量塌陷到 rubric 的中间锚点，`physical_compliance` 与
        `history_similarity` 都能由校验器和检索相似度直接算出，不必让模型猜。
        """
        resolved = {
            name: (
                getattr(self, name)
                if values.get(name) is None
                else _clamp_score(values[name])
            )
            for name in CONFIDENCE_DIMENSIONS
        }
        return ConfidenceBreakdown(**resolved)

    def to_dict(self) -> Dict[str, float]:
        return {name: getattr(self, name) for name in CONFIDENCE_DIMENSIONS}


@dataclass(frozen=True)
class DiagnosisResponse:
    """一次推理的结构化结果。verdict 必须是 L1/L2/fiber 三选一。"""

    steps: Tuple[ReasoningStep, ...] = ()
    verdict: Optional[str] = None
    confidence: float = 0.0
    confidence_breakdown: ConfidenceBreakdown = field(default_factory=ConfidenceBreakdown)
    self_reported_confidence: float = 0.0
    missing_information: Tuple[str, ...] = ()
    raw_output: str = ""
    forced: bool = False
    fallback_source: str = ""
    compliance_penalties: Tuple[Dict[str, Any], ...] = ()

    def with_adjustments(
        self,
        *,
        confidence_breakdown: Optional[ConfidenceBreakdown] = None,
        forced: Optional[bool] = None,
        fallback_source: Optional[str] = None,
        compliance_penalties: Optional[Sequence[Dict[str, Any]]] = None,
        verdict: Optional[str] = None,
    ) -> "DiagnosisResponse":
        breakdown = confidence_breakdown or self.confidence_breakdown
        return DiagnosisResponse(
            steps=self.steps,
            verdict=self.verdict if verdict is None else verdict,
            confidence=breakdown.weighted_score(),
            confidence_breakdown=breakdown,
            self_reported_confidence=self.self_reported_confidence,
            missing_information=self.missing_information,
            raw_output=self.raw_output,
            forced=self.forced if forced is None else forced,
            fallback_source=self.fallback_source if fallback_source is None else fallback_source,
            compliance_penalties=(
                self.compliance_penalties
                if compliance_penalties is None
                else tuple(dict(item) for item in compliance_penalties)
            ),
        )

    @property
    def cited_evidence(self) -> Tuple[str, ...]:
        seen: List[str] = []
        for step in self.steps:
            for token in step.cited_evidence:
                if token not in seen:
                    seen.append(token)
        return tuple(seen)

    @property
    def cited_constraints(self) -> Tuple[str, ...]:
        seen: List[str] = []
        for step in self.steps:
            for item in step.cited_constraints:
                if item not in seen:
                    seen.append(item)
        return tuple(seen)

    def excluded_root_causes(self) -> Tuple[str, ...]:
        return tuple(
            sorted({step.target for step in self.steps if step.effect == "exclude" and step.target})
        )

    def step_votes(self) -> Dict[str, int]:
        """把推理步骤折算成对每个根因的净票数：support 加一票，exclude 减一票。"""
        votes: Dict[str, int] = {}
        for step in self.steps:
            if step.target not in ROOT_CAUSES:
                continue
            if step.effect == "support":
                votes[step.target] = votes.get(step.target, 0) + 1
            elif step.effect == "exclude":
                votes[step.target] = votes.get(step.target, 0) - 1
        return votes

    def step_majority(self) -> Optional[str]:
        """推理链自身指向的根因。并列或无有效步骤时返回 None，表示不足以推翻 verdict。"""
        votes = self.step_votes()
        if not votes:
            return None
        best = max(votes.values())
        winners = [name for name, count in votes.items() if count == best]
        return winners[0] if len(winners) == 1 else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps": [step.to_dict() for step in self.steps],
            "verdict": self.verdict,
            "confidence": self.confidence,
            "confidence_breakdown": self.confidence_breakdown.to_dict(),
            "self_reported_confidence": self.self_reported_confidence,
            "missing_information": list(self.missing_information),
            "forced": self.forced,
            "fallback_source": self.fallback_source,
            "compliance_penalties": [dict(item) for item in self.compliance_penalties],
        }


#: 推理型模型（如 DeepSeek-R1 系列）会先输出思考段再给答案。
#: 思考段里通常带花括号和半成品 JSON，必须先切掉，否则会解析到草稿而不是结论。
THINK_CLOSE_PATTERN = re.compile(r"</think\s*>", flags=re.IGNORECASE)


def strip_reasoning_prefix(text: str) -> str:
    """去掉思考段，只保留最后一个 `</think>` 之后的内容。

    没有闭合标签时原样返回：可能是非推理模型，也可能是思考被 max_tokens 截断。
    后者会因为拿不到完整 JSON 而被判为不合规，这正是期望行为——
    截断的输出不该被当成结论。
    """
    matches = list(THINK_CLOSE_PATTERN.finditer(text or ""))
    return text[matches[-1].end():] if matches else (text or "")


def extract_json_object(text: str) -> Optional[str]:
    """扫描出最后一个括号配对完整的顶层 JSON 对象。

    不能用 `re.search(r"\\{.*\\}", ..., re.S)`：那个正则是贪婪的，会从第一个左括号
    一路吃到最后一个右括号。模型在正式答案之前若写了任何带花括号的说明，
    抓出来的就是一段跨越两个对象的无效文本。这里改成扫描配对，并且取**最后一个**，
    因为模型的最终答案总在最后。
    """
    source = strip_reasoning_prefix(text)
    candidates: List[str] = []
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for index, char in enumerate(source):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    candidates.append(source[start:index + 1])
    return candidates[-1] if candidates else None


def parse_response(text: str) -> Optional[DiagnosisResponse]:
    """严格解析。解析不出来就返回 `None`，由调用方按「不合规」处理。

    刻意不做容错补全：如果模型没有按 schema 输出，那它的推理过程本来也无法校验，
    此时最安全的处置是重写或弃权，而不是猜它想说什么。
    """
    payload = extract_json_object(text)
    if payload is None:
        return None
    try:
        value = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        return None

    raw_steps = value.get("steps")
    if not isinstance(raw_steps, list):
        return None
    steps: List[ReasoningStep] = []
    for item in raw_steps:
        if not isinstance(item, dict):
            return None
        effect = str(item.get("effect", "neutral"))
        if effect not in EFFECTS:
            return None
        target = str(item.get("target", "") or "")
        if target and target not in ROOT_CAUSES:
            return None
        steps.append(
            ReasoningStep(
                sop_step_id=str(item.get("sop_step_id", "")),
                cited_predicates=tuple(str(token) for token in item.get("cited_predicates", []) or ()),
                claim=str(item.get("claim", "")),
                cited_evidence=tuple(str(token) for token in item.get("cited_evidence", []) or ()),
                cited_constraints=tuple(str(token) for token in item.get("cited_constraints", []) or ()),
                effect=effect,
                target=target,
            )
        )

    verdict_raw = str(value.get("verdict", "")).strip()
    if verdict_raw not in tuple(ROOT_CAUSES):
        return None
    self_reported_confidence = _clamp_score(value.get("confidence", 0.0))
    breakdown = ConfidenceBreakdown.from_mapping(value.get("confidence_breakdown"))
    confidence = breakdown.weighted_score()

    return DiagnosisResponse(
        steps=tuple(steps),
        verdict=verdict_raw,
        confidence=confidence,
        confidence_breakdown=breakdown,
        self_reported_confidence=self_reported_confidence,
        missing_information=tuple(str(item) for item in value.get("missing_information", []) or ()),
        raw_output=text,
    )
