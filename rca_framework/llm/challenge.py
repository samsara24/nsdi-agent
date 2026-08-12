"""迭代 4：把 LLM 从定界器改成**质疑器**。

改岗位的理由是测出来的，不是设计偏好。迭代 3 在 test 上做了配对比较：
LLM 给出结论的 48 条里，专家规则对 38 条、LLM 对 29 条，
LLM 赢过规则的只有 1 条（McNemar p=0.0117）。让一个比规则显著更差的部件去出结论，
无论怎么调 prompt 都不会有净收益。

但同一批 trace 也显示模型**读得懂证据**：v7 的方向表被正确使用，
典型回答第一步就把「L1 侧收光异常」正确地翻译成「根因在 L2」。
读得懂证据、但判不准根因——这两件事同时成立时，合适的岗位是质疑而不是定界：

    专家规则出 verdict，LLM 只回答「这条规则的前提在本 case 上是否不成立」。

这个岗位的关键性质是**它不要求 LLM 比规则更准**。规则在 test 上错 23.4%，
只要被 LLM 标为可疑的那批 case 里错误率显著高于 23.4%，
把有限的人工预算投到这批 case 上就是划算的。考核指标因此是质疑命中率，
不是准确率。

必须打败的基线不是随机，而是**按 train 组可靠性查表**：
train 上最差的两组（`single:serdes_snr` 57.9%、`port_status_gate` 50.0%）
在 test 上共 32 条、错 8 条，命中率 25.0%，相对 23.4% 的错误率几乎没有增益。
组可靠性在两个切分之间不稳定，所以查表基线很弱——但它必须被显式超过，
否则「LLM 会质疑」就只是把一张查不准的表换了个昂贵的实现。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..types import ROOT_CAUSES
from .protocol import extract_json_object

CHALLENGE_PROMPT_VERSION = "rca-challenge-v1"

ASSESSMENTS: Tuple[str, ...] = ("agree", "challenge")

#: 可被质疑的前提。做成枚举而不是自由文本，是为了让质疑本身可统计、可校验：
#: 自由文本的「我觉得不太对」无法判定它质疑的是哪一环，也就无法验证它是否说中。
PREMISES: Tuple[str, ...] = (
    "anomaly_is_real",
    "direction_table_applies",
    "other_side_overlooked",
    "scope_too_narrow",
    "no_evidence_fallback",
)

PREMISE_TEXT: Dict[str, str] = {
    "anomaly_is_real": (
        "触发规则的读数是真异常。若该读数其实来自遥测缺失、断光哨兵或单次采样抖动，"
        "这条前提就不成立。"
    ),
    "direction_table_applies": (
        "方向表适用于这个观测。接收类（rxpower、media_snr）指向对端、"
        "发送与电口类（txpower、host_snr、serdes_snr）指向本端——"
        "若本 case 的观测同时具备两类特征，方向就不是单一的。"
    ),
    "other_side_overlooked": (
        "另一端没有同等或更强的异常。规则按优先级取最小值裁决，"
        "若另一端存在被优先级压掉但同样严重的异常，定界可能反了。"
    ),
    "scope_too_narrow": (
        "异常的范围足以支撑整端定界。若只有单个 lane 异常而其余 lane 正常，"
        "更可能是该 lane 的局部问题而不是整端故障。"
    ),
    "no_evidence_fallback": (
        "规则给出的是有证据支撑的判断。若规则走的是「两端都无明显异常，兜底报本端」，"
        "那么这个结论没有任何本 case 的证据支撑。"
    ),
}

CHALLENGE_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "assessment": {"type": "string", "enum": list(ASSESSMENTS)},
        "premise_at_risk": {"type": "string", "enum": list(PREMISES) + [""]},
        "cited_evidence": {"type": "array", "items": {"type": "string"}},
        "alternative_verdict": {"type": "string", "enum": list(ROOT_CAUSES) + [""]},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "evidence_to_collect": {"type": "array", "items": {"type": "string"}},
        "explanation": {"type": "string"},
    },
    "required": [
        "assessment",
        "premise_at_risk",
        "cited_evidence",
        "alternative_verdict",
        "confidence",
        "evidence_to_collect",
        "explanation",
    ],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class ChallengeResponse:
    assessment: str = "agree"
    premise_at_risk: str = ""
    cited_evidence: Tuple[str, ...] = ()
    alternative_verdict: str = ""
    confidence: float = 0.0
    evidence_to_collect: Tuple[str, ...] = ()
    explanation: str = ""
    raw_output: str = ""

    @property
    def challenges(self) -> bool:
        return self.assessment == "challenge"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "assessment": self.assessment,
            "premise_at_risk": self.premise_at_risk,
            "cited_evidence": list(self.cited_evidence),
            "alternative_verdict": self.alternative_verdict,
            "confidence": self.confidence,
            "evidence_to_collect": list(self.evidence_to_collect),
            "explanation": self.explanation,
        }


def parse_challenge(text: str) -> Optional[ChallengeResponse]:
    """严格解析。与 `parse_response` 同样刻意不做容错补全。

    质疑器的失败模式与定界器不同：解析不出来时**默认 agree**（由调用方决定），
    因为「无法解析」不构成质疑理由，而把它算成质疑会直接污染命中率。
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

    assessment = str(value.get("assessment", "")).strip()
    if assessment not in ASSESSMENTS:
        return None
    premise = str(value.get("premise_at_risk", "") or "")
    if premise and premise not in PREMISES:
        return None
    alternative = str(value.get("alternative_verdict", "") or "")
    if alternative and alternative not in ROOT_CAUSES:
        return None
    try:
        confidence = float(value.get("confidence", 0.0))
    except (TypeError, ValueError):
        return None

    def _strings(key: str) -> Tuple[str, ...]:
        raw = value.get(key)
        if not isinstance(raw, list):
            return ()
        return tuple(str(item) for item in raw if isinstance(item, (str, int, float)))

    return ChallengeResponse(
        assessment=assessment,
        premise_at_risk=premise,
        cited_evidence=_strings("cited_evidence"),
        alternative_verdict=alternative,
        confidence=max(0.0, min(1.0, confidence)),
        evidence_to_collect=_strings("evidence_to_collect"),
        explanation=str(value.get("explanation", "") or ""),
        raw_output=text,
    )


SYSTEM_PREAMBLE = """你是光链路故障定界的复核专家。

一套确定性专家规则已经对本 case 给出了定界结论。你的任务**不是**重新定界，
也不是给出你自己的判断——那件事规则做得比你好。你的任务只有一件：

    判断这条规则的前提在本 case 上是否有某一条不成立。

这个分工是测量出来的：在同一批 case 上，规则的准确率显著高于自由推理，
但规则是「无条件执行」的，它不会发现自己的前提被违反。你能读懂证据，
所以你能发现这件事。

判据：

- 默认应当是 `agree`。规则整体错误率约 23%，如果你对超过 23% 的 case 说 challenge，
  你就没有提供任何信息——那等于随机抽人工复核。
- 只有当你能**指出具体哪一条前提被具体哪一条证据推翻**时才 `challenge`。
  说不出证据的怀疑不是质疑，是噪声。
- `challenge` 时必须在 `cited_evidence` 里给出推翻前提的证据 token，
  且只能引用「可用证据」清单里的 token。
- `alternative_verdict` 可以留空。你不需要知道正确答案才能指出前提有问题——
  「这条规则在这里不适用」本身就是有用的输出。
- `evidence_to_collect` 写现场应当补采什么才能定案，这是给运维看的，要具体。
- `explanation` 用一到三句话说清楚，读者是值班工程师，不是模型。
"""


def build_challenge_prompt(
    *,
    case_id: str,
    expert_verdict: str,
    expert_group: str,
    expert_reason: str,
    evidence_tokens: Sequence[str],
    missing_fields: Sequence[str] = (),
) -> str:
    """拼装质疑 prompt。与定界 prompt 一样保持确定性：同输入必须同输出。"""
    premises = "\n".join(
        f"- `{name}`：{PREMISE_TEXT[name]}" for name in PREMISES
    )
    payload = {
        "case_id": case_id,
        "专家规则结论": expert_verdict,
        "命中的规则": expert_group,
        "规则给出的理由": expert_reason,
        "可用证据": sorted(evidence_tokens),
        "未采集字段": sorted(missing_fields),
    }
    return "\n".join(
        (
            SYSTEM_PREAMBLE,
            "",
            "可以被质疑的前提（`premise_at_risk` 只能填这几个之一）：",
            "",
            premises,
            "",
            "待复核的 case：",
            "",
            json.dumps(payload, ensure_ascii=False, indent=2),
            "",
            "只输出一个 JSON 对象，字段如下：",
            "",
            json.dumps(CHALLENGE_OUTPUT_SCHEMA["properties"], ensure_ascii=False, indent=2),
        )
    )


def challenge_metrics(
    rows: Sequence[Tuple[bool, bool]],
    *,
    baseline_error_rate: Optional[float] = None,
) -> Dict[str, Any]:
    """`rows` 是 (被质疑, 规则判错) 的序列。

    命中率是这个岗位唯一有意义的指标：被质疑的 case 里规则确实判错的比例。
    同时报召回（抓到了多少比例的错误）与 lift（相对整体错误率），
    因为只报命中率会被「只质疑一条最有把握的」刷高。
    """
    total = len(rows)
    challenged = [row for row in rows if row[0]]
    errors = [row for row in rows if row[1]]
    hits = [row for row in challenged if row[1]]
    error_rate = baseline_error_rate if baseline_error_rate is not None else (
        len(errors) / total if total else 0.0
    )
    precision = len(hits) / len(challenged) if challenged else 0.0
    return {
        "case_count": total,
        "challenged": len(challenged),
        "challenge_rate": round(len(challenged) / total, 6) if total else 0.0,
        "rule_error_count": len(errors),
        "rule_error_rate": round(error_rate, 6),
        "hits": len(hits),
        "hit_rate": round(precision, 6),
        "error_recall": round(len(hits) / len(errors), 6) if errors else 0.0,
        "lift_over_error_rate": round(precision - error_rate, 6),
    }
