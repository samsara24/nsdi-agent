"""迭代 5：LLM 的最后一个岗位——解释。

前四轮把 LLM 试过的三个岗位都测掉了：作为定界器显著劣于专家规则
（McNemar p=0.0117）；约束合规与结论正确之间测不到关联（p=0.15 / 0.70）；
作为质疑器，质疑强度与规则是否判错也测不到关联（p=0.108，方向还是负的）。
三次测量的对象、指标、失败方式都不同，结论一致：**它不能提供任何形式的判断。**

但同一批实验里它没有失败的地方同样明确：质疑器 v2 引用的 402 个 token 里
只有 3 个不存在于本 case（0.7%），107 条给出 102 种不同解释，
前提类别与命中规则严格对应。**读证据它做得很好。**

解释是唯一只需要「读」不需要「判」的岗位：结论由专家规则给出，
模型只负责把它讲成运维能核对的一段话。这也是本项目对外承诺的可解释性的落点。

因此评测指标必须换掉。准确率在这里没有意义（结论不是模型给的），
用人工打分又不可复现。这里用四项**机器可判**的可核对性：

1. `token_existence`：解释引用的每个证据 token 都真实存在于本 case。
2. `token_relevance`：至少有一个引用的 token 落在规则实际依据的 `(侧, 指标)` 上。
   这一条防的是「说了一堆真话，但没有一句与规则的依据有关」。
3. `direction_consistency`：解释声明的「症状在哪端 / 根因在哪端」与方向表一致。
4. `verdict_consistency`：解释声明的根因端与规则的结论一致。
   这一条防的是模型嘴上解释、心里改判——它没有改判的资格。

四项全过才算一条合格解释。这个指标不需要标签，因此可以直接在生产上持续监控。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..constraints.semantics import token_scope
from ..types import ROOT_CAUSES

EXPLAIN_PROMPT_VERSION = "rca-explain-v1"

#: 方向表，与 `docs/EXPERT_EXPERIENCE.md` 及 prompt v7 中的表一致。
#: 值是「症状出现在本端时，根因在哪一端」。
DIRECTION_BY_METRIC: Dict[str, str] = {
    "rxpower": "far",
    "media_snr": "far",
    "txpower": "local",
    "host_snr": "local",
    "serdes_snr": "local",
}

SIDES: Tuple[str, ...] = ("L1", "L2")


def _opposite(side: str) -> str:
    return "L2" if side == "L1" else "L1"


EXPLAIN_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "key_evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "token": {"type": "string"},
                    "reads_as": {"type": "string"},
                },
                "required": ["token", "reads_as"],
                "additionalProperties": False,
            },
        },
        "symptom_side": {"type": "string", "enum": list(SIDES)},
        "root_cause_side": {"type": "string", "enum": list(ROOT_CAUSES)},
        "direction_reason": {"type": "string"},
        "next_actions": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "summary",
        "key_evidence",
        "symptom_side",
        "root_cause_side",
        "direction_reason",
        "next_actions",
    ],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Explanation:
    summary: str = ""
    key_evidence: Tuple[Tuple[str, str], ...] = ()
    symptom_side: str = ""
    root_cause_side: str = ""
    direction_reason: str = ""
    next_actions: Tuple[str, ...] = ()
    raw_output: str = ""

    @property
    def cited_tokens(self) -> Tuple[str, ...]:
        return tuple(token for token, _ in self.key_evidence)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "key_evidence": [
                {"token": token, "reads_as": reads_as} for token, reads_as in self.key_evidence
            ],
            "symptom_side": self.symptom_side,
            "root_cause_side": self.root_cause_side,
            "direction_reason": self.direction_reason,
            "next_actions": list(self.next_actions),
        }


def parse_explanation(text: str) -> Optional[Explanation]:
    from .protocol import extract_json_object

    payload = extract_json_object(text)
    if payload is None:
        return None
    try:
        value = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        return None

    symptom = str(value.get("symptom_side", "") or "")
    root = str(value.get("root_cause_side", "") or "")
    if symptom not in SIDES or root not in ROOT_CAUSES:
        return None

    raw_evidence = value.get("key_evidence")
    if not isinstance(raw_evidence, list):
        return None
    evidence: List[Tuple[str, str]] = []
    for item in raw_evidence:
        if not isinstance(item, dict):
            return None
        token = str(item.get("token", "") or "")
        if not token:
            return None
        evidence.append((token, str(item.get("reads_as", "") or "")))

    raw_actions = value.get("next_actions")
    return Explanation(
        summary=str(value.get("summary", "") or ""),
        key_evidence=tuple(evidence),
        symptom_side=symptom,
        root_cause_side=root,
        direction_reason=str(value.get("direction_reason", "") or ""),
        next_actions=(
            tuple(str(item) for item in raw_actions) if isinstance(raw_actions, list) else ()
        ),
        raw_output=text,
    )


SYSTEM_PREAMBLE = """你是光链路故障定界专家，正在为值班工程师解释一条已经完成的定界。

结论已经由一套确定性专家规则给出。**你没有改判的资格**：
你的任务是把这个结论讲清楚，不是重新判断它对不对。
如果你觉得结论可疑，也只能照实解释规则的依据，不能改写 `root_cause_side`。

要求：

1. `key_evidence` 里的每个 token 必须**逐字**来自「可用证据」清单，
   并且应当优先选择规则实际依据的那几项。`reads_as` 用一句人话说明它的含义，
   例如「L1 侧某个 lane 的接收光功率跌到工程阈值以下」。
2. `symptom_side` 是**现象出现在哪一端**，`root_cause_side` 是**根因在哪一端**。
   两者经常不同，这正是需要向工程师讲清楚的地方。
3. `direction_reason` 解释为什么症状在这一端而根因在那一端。依据是：
   接收类读数（rxpower、media_snr）测的是对端发来的光，所以指向**对端**；
   发送与电口类读数（txpower、host_snr、serdes_snr）测的是本端自己发出或接收的电信号，
   所以指向**本端**。
4. `summary` 两到四句，读者是值班工程师，不是模型。不要复述 token 名，讲现象。
5. `next_actions` 是现场可执行的动作，要具体到查哪一端的什么。
"""


def build_explain_prompt(
    *,
    case_id: str,
    expert_verdict: str,
    expert_group: str,
    expert_reason: str,
    rule_evidence: Sequence[Tuple[str, str]],
    evidence_tokens: Sequence[str],
    missing_fields: Sequence[str] = (),
) -> str:
    payload = {
        "case_id": case_id,
        "规则结论（不可更改）": expert_verdict,
        "命中的规则": expert_group,
        "规则给出的理由": expert_reason,
        "规则实际依据的异常": [
            {"侧": side, "指标": metric} for side, metric in rule_evidence
        ],
        "可用证据": sorted(evidence_tokens),
        "未采集字段": sorted(missing_fields),
    }
    return "\n".join(
        (
            SYSTEM_PREAMBLE,
            "",
            "待解释的 case：",
            "",
            json.dumps(payload, ensure_ascii=False, indent=2),
            "",
            "只输出一个 JSON 对象，字段如下：",
            "",
            json.dumps(EXPLAIN_OUTPUT_SCHEMA["properties"], ensure_ascii=False, indent=2),
        )
    )


def check_explanation(
    explanation: Optional[Explanation],
    *,
    available_tokens: Sequence[str],
    rule_evidence: Sequence[Tuple[str, str]],
    rule_verdict: Optional[str],
) -> Dict[str, Any]:
    """四项机器可判的可核对性检查。解析失败时四项全判不通过。"""
    if explanation is None:
        return {
            "parsed": False,
            "token_existence": False,
            "token_relevance": False,
            "direction_consistency": False,
            "verdict_consistency": False,
            "all_pass": False,
            "fabricated_tokens": [],
        }

    available = set(available_tokens)
    fabricated = [token for token in explanation.cited_tokens if token not in available]
    token_existence = bool(explanation.cited_tokens) and not fabricated

    rule_scopes = {(side, metric) for side, metric in rule_evidence}
    cited_scopes = {
        scope
        for scope in (token_scope(token) for token in explanation.cited_tokens)
        if scope is not None and scope[1] is not None
    }
    # 规则依据为空（no_anomaly 兜底）时这一条无从判定，按通过处理：
    # 那种 case 本来就没有依据可引，苛求相关性等于惩罚模型如实解释。
    token_relevance = (not rule_scopes) or bool(cited_scopes & rule_scopes)

    direction_consistency = _direction_consistent(
        explanation, rule_used_evidence=bool(rule_scopes)
    )
    verdict_consistency = explanation.root_cause_side == rule_verdict

    checks = {
        "parsed": True,
        "token_existence": token_existence,
        "token_relevance": token_relevance,
        "direction_consistency": direction_consistency,
        "verdict_consistency": verdict_consistency,
        "fabricated_tokens": fabricated,
    }
    checks["all_pass"] = all(
        checks[name]
        for name in (
            "token_existence",
            "token_relevance",
            "direction_consistency",
            "verdict_consistency",
        )
    )
    return checks


def _direction_consistent(explanation: Explanation, *, rule_used_evidence: bool = True) -> bool:
    """引用证据的指标类型是否支持声明的根因端。

    方向由**每个 token 自己的 `(侧, 指标)`** 推出，不看模型声明的 `symptom_side`。
    第一版是拿 `symptom_side` 去筛 token 的，实测 5 条不合格里有 3 条坏在这里：
    模型把「运维在哪一端发现问题」当成了 symptom_side，与它引用的读数所在侧不一致，
    于是物理叙述完全正确的解释被判错。症状端标签本身是模棱两可的，
    不该成为方向检查的支点；token 的侧别不模棱两可。

    两种情况跳过这项检查：

    - `fiber` 结论。方向表说不出光纤，它是两端仲裁的产物。
    - 规则本身没有用到任何异常（`no_anomaly` 兜底）。此时不存在方向推理，
      模型如实说明「规则在两端都无异常时兜底报本端」反而会被判违反方向表。
      与 `token_relevance` 同一个理由：不能因为模型如实解释而惩罚它。

    引用里同时含本端类与对端类指标时，两个方向都算合规——
    这种 case 本来就是混合的，逼模型二选一会把如实描述判成错误。
    """
    if explanation.root_cause_side == "fiber" or not rule_used_evidence:
        return True
    implied = set()
    for token in explanation.cited_tokens:
        scope = token_scope(token)
        if scope is None or scope[0] not in SIDES:
            continue
        direction = DIRECTION_BY_METRIC.get(scope[1] or "")
        if direction == "far":
            implied.add(_opposite(scope[0]))
        elif direction == "local":
            implied.add(scope[0])
    if not implied:
        return False
    return explanation.root_cause_side in implied


def summarize_checks(reports: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    total = len(reports)

    def rate(name: str) -> float:
        return round(sum(1 for item in reports if item[name]) / total, 6) if total else 0.0

    return {
        "case_count": total,
        "parse_rate": rate("parsed"),
        "token_existence": rate("token_existence"),
        "token_relevance": rate("token_relevance"),
        "direction_consistency": rate("direction_consistency"),
        "verdict_consistency": rate("verdict_consistency"),
        "all_pass": rate("all_pass"),
        "fabricated_token_count": sum(len(item["fabricated_tokens"]) for item in reports),
    }
