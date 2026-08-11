"""规则 empirical study 的固定 prompt。

这个模块只回答一个实验问题：在输入证据完全相同、模型与解码参数完全相同的条件下，
加入当前物理约束是否提高 RCA 判断质量。它刻意不注入历史 case、历史标签或路由结论，
避免把证据图复用能力误算成大模型的规则推理能力。
"""

from __future__ import annotations

import hashlib
import inspect
import json
from typing import Any, Dict

from ..constraints.library import CONSTRAINT_LIBRARY, ConstraintLibrary, render_prompt_block
from .prompts import OUTPUT_INSTRUCTION, ROOT_CAUSE_DEFINITIONS


EMPIRICAL_PROMPT_VERSION = "rule-empirical-study-v1"

EVIDENCE_ONLY_PREAMBLE = """你是光链路故障定界助手。请只根据给定的可用证据，
判断根因是 L1、L2、fiber，或在证据无法区分时输出 abstain。

硬性要求：
1. 只能引用「可用证据」清单中的 token，不得编造观测。
2. 本组实验不提供物理约束，因此 cited_constraints 必须为空。
3. 每一步必须引用至少一条可用证据。
4. 证据同时支持多个根因、或证据自相矛盾时，应输出 abstain。
"""

RULES_PREAMBLE = """你是光链路故障定界专家。请根据给定证据，并严格遵守当前物理约束，
判断根因是 L1、L2、fiber，或在证据无法区分时输出 abstain。

硬性要求：
1. 只能引用「可用证据」清单中的 token，不得编造观测。
2. 只能引用本 prompt 提供的物理约束编号。
3. 每一步必须至少引用一条证据或一条约束。
4. 排除类约束优先于倾向性线索；已排除的根因不得作为结论。
5. 标注为「待专家审核」的规则可以参考，但不能作为唯一定论依据。
6. 证据无法区分候选根因、或证据自相矛盾时，应输出 abstain。
"""


def _payload(request: Any) -> Dict[str, Any]:
    return {
        "任务": "光链路根因三分类规则经验研究",
        "根因定义": ROOT_CAUSE_DEFINITIONS,
        "本 case 证据": {
            "case_id": request.case_id,
            "可用证据": list(request.evidence_tokens),
            "遥测完整性": request.telemetry_status,
            "未采集字段（仅作补采参考）": list(request.missing_fields),
        },
        "已由确定性规则排除的根因": [
            {
                "根因": item.root_cause,
                "规则": item.constraint_id,
                "原因": item.reason,
            }
            for item in request.exclusions
        ],
        "可选根因": list(request.candidate_root_causes),
    }


def build_empirical_prompt(
    request: Any,
    *,
    include_rules: bool,
    library: ConstraintLibrary = CONSTRAINT_LIBRARY,
    retry_feedback: str = "",
) -> str:
    """构造 evidence-only 或 rules-prompt，二者共享证据与输出协议。"""
    sections = [RULES_PREAMBLE if include_rules else EVIDENCE_ONLY_PREAMBLE]
    if retry_feedback:
        sections.append(
            "上一次回答未通过规则校验，问题如下。请只修正这些问题，不要引入新证据：\n"
            + retry_feedback
        )
    if include_rules:
        constraints = [library.get(item) for item in request.constraint_ids]
        sections.append(
            "当前物理约束（按 排除 -> 禁止推断 -> 恒等关系 -> 倾向性 排序）：\n"
            + render_prompt_block(constraints=constraints)
        )
    sections.append(json.dumps(_payload(request), ensure_ascii=False, indent=2))
    sections.append(OUTPUT_INSTRUCTION)
    return "\n\n".join(sections)


def empirical_prompt_hash(*, include_rules: bool) -> str:
    payload = "\n".join(
        (
            EMPIRICAL_PROMPT_VERSION,
            "rules" if include_rules else "evidence-only",
            RULES_PREAMBLE if include_rules else EVIDENCE_ONLY_PREAMBLE,
            OUTPUT_INSTRUCTION,
            inspect.getsource(_payload),
            inspect.getsource(build_empirical_prompt),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
