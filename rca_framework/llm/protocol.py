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

#: 供 vLLM guided decoding 使用。字段与 `DiagnosisResponse` 一一对应。
DIAGNOSIS_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "cited_evidence": {"type": "array", "items": {"type": "string"}},
                    "cited_constraints": {"type": "array", "items": {"type": "string"}},
                    "effect": {"type": "string", "enum": list(EFFECTS)},
                    "target": {"type": "string", "enum": list(ROOT_CAUSES) + [""]},
                },
                "required": ["claim", "cited_evidence", "cited_constraints", "effect", "target"],
                "additionalProperties": False,
            },
        },
        "verdict": {"type": "string", "enum": list(ROOT_CAUSES) + ["abstain"]},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "missing_information": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["steps", "verdict", "confidence", "missing_information"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class ReasoningStep:
    claim: str
    cited_evidence: Tuple[str, ...] = ()
    cited_constraints: Tuple[str, ...] = ()
    effect: str = "neutral"
    target: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim": self.claim,
            "cited_evidence": list(self.cited_evidence),
            "cited_constraints": list(self.cited_constraints),
            "effect": self.effect,
            "target": self.target,
        }


@dataclass(frozen=True)
class DiagnosisResponse:
    """一次 N5c 推理的结构化结果。`verdict=None` 表示模型主动弃权。"""

    steps: Tuple[ReasoningStep, ...] = ()
    verdict: Optional[str] = None
    confidence: float = 0.0
    missing_information: Tuple[str, ...] = ()
    raw_output: str = ""

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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps": [step.to_dict() for step in self.steps],
            "verdict": self.verdict,
            "confidence": self.confidence,
            "missing_information": list(self.missing_information),
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
                claim=str(item.get("claim", "")),
                cited_evidence=tuple(str(token) for token in item.get("cited_evidence", []) or ()),
                cited_constraints=tuple(str(token) for token in item.get("cited_constraints", []) or ()),
                effect=effect,
                target=target,
            )
        )

    verdict_raw = str(value.get("verdict", "abstain")).strip()
    if verdict_raw not in tuple(ROOT_CAUSES) + ("abstain",):
        return None
    try:
        confidence = min(1.0, max(0.0, float(value.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0

    return DiagnosisResponse(
        steps=tuple(steps),
        verdict=None if verdict_raw == "abstain" else verdict_raw,
        confidence=confidence,
        missing_information=tuple(str(item) for item in value.get("missing_information", []) or ()),
        raw_output=text,
    )
