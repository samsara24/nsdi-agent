"""Expert-authored SOP playbook for low-match RCA cases.

This module is intentionally separate from `rca_framework.expert`: the latter
is a deterministic direction/priority rule used as a baseline and semantic
guard, while this playbook is an ordered checklist injected only into the N5c
cold-start LLM prompt.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, Sequence, Tuple


EXPERT_SOP_VERSION = "expert-sop-playbook-v2"
EXPERT_SOP_SOURCE = "docs/EXPERT_EXPERIENCE.md:华为word"


@dataclass(frozen=True)
class ExpertSOPStep:
    step_id: str
    title: str
    instruction: str
    expected_output: str
    source_lines: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


EXPERT_SOP_STEPS: Tuple[ExpertSOPStep, ...] = (
    ExpertSOPStep(
        step_id="S1_collect_anomaly_level",
        title="确认异常级别",
        instruction=(
            "先区分 down 异常、指标值异常和指标离群异常。down 异常优先级最高，"
            "其次是值异常，最后才是离群异常。不要把缺失字段当成异常。"
        ),
        expected_output="每个可用指标的异常级别或 normal/unknown。",
        source_lines=("指标异常优先级：down异常0，指标值异常1，指标离群异常2",),
    ),
    ExpertSOPStep(
        step_id="S2_match_fault_pattern",
        title="按故障模式优先级定位单侧",
        instruction=(
            "对 L1、L2 两侧分别按优先级检查：txpower down、mediaSNR+serdesSNR+rxpower "
            "组合异常、hostSNR、serdesSNR、mediaSNR、rxpower、txpower 非 down 异常。"
        ),
        expected_output="每侧的最高优先级故障模式、指向端、priority 编码。",
        source_lines=(
            "故障定位模式优先级：txpower down 0；组合异常 1；hostSNR 2；serdesSNR 3；mediaSNR 4；rxpower 5；txpower 非down 6",
        ),
    ),
    ExpertSOPStep(
        step_id="S3_apply_direction_semantics",
        title="应用归因方向",
        instruction=(
            "接收类观测 rxpower/media_snr 度量对端发来的光，指向对端；"
            "发送类与电口观测 txpower/host_snr/serdes_snr 指向本端。"
        ),
        expected_output="把症状端转换为候选根因端，避免把接收侧症状归给本端。",
        source_lines=("单侧光模块定位逻辑",),
    ),
    ExpertSOPStep(
        step_id="S4_arbitrate_two_sides",
        title="两端裁决",
        instruction=(
            "若只有一端有有效异常，直接按该端 S3 的定位结果作为 verdict；"
            "若两端均有异常，比较 S2 的 priority 编码，数值更小的一侧胜出并作为 verdict；"
            "只有当两端 priority 相同、且定位结果互相冲突时，fiber 才进入候选。"
            "端点候选已经能解释现有症状时不得改判 fiber。"
        ),
        expected_output="L1/L2/fiber 候选及其裁决依据。",
        source_lines=("结合两端光模块的定位结果和优先级给出最终光链路故障定位结果",),
    ),
    ExpertSOPStep(
        step_id="S5_report_confidence_and_gaps",
        title="证据不足时降级表达",
        instruction=(
            "若现有证据无法区分候选根因，仍必须在 L1/L2/fiber 中选出最可能的一个作为 verdict，"
            "把不确定性表达为低 evidence_completeness 或低 reasoning_completeness，"
            "并在 missing_information 写明补采项；禁止输出 abstain。"
            "不得用类别先验、训练集叶节点或历史标签投票作为当前 case 的物理证据。"
        ),
        expected_output="最可能根因、对应的低置信度维度打分与补采项。",
        source_lines=("个人整体思路：置信度偏低触发降级策略，转人工介入",),
    ),
)


def expert_sop_to_dict(steps: Sequence[ExpertSOPStep] = EXPERT_SOP_STEPS) -> Dict[str, Any]:
    return {
        "version": EXPERT_SOP_VERSION,
        "source": EXPERT_SOP_SOURCE,
        "steps": [step.to_dict() for step in steps],
        "content_hash": expert_sop_hash(steps),
    }


def expert_sop_hash(steps: Sequence[ExpertSOPStep] = EXPERT_SOP_STEPS) -> str:
    payload = {
        "version": EXPERT_SOP_VERSION,
        "source": EXPERT_SOP_SOURCE,
        "steps": [step.to_dict() for step in steps],
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def render_expert_sop_prompt_block(steps: Sequence[ExpertSOPStep] = EXPERT_SOP_STEPS) -> str:
    lines = [
        f"专家排障 SOP（{EXPERT_SOP_VERSION}）：以下检查顺序与裁决规则必须逐步执行，"
        "S4 的裁决结论就是 verdict 的默认取值，偏离它必须在推理步骤中给出物理约束依据。"
        "SOP 条目本身不是遥测观测，不能写进 `cited_evidence` 或 `cited_constraints`。"
    ]
    for index, step in enumerate(steps, 1):
        lines.append(
            f"{index}. {step.step_id}｜{step.title}\n"
            f"   - 操作：{step.instruction}\n"
            f"   - 输出：{step.expected_output}"
        )
    return "\n".join(lines)
