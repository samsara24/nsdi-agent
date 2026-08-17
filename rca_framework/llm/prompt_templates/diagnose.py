"""Low-tier diagnosis prompt with layered knowledge injection."""

from __future__ import annotations

import json
from typing import Any

from ...constraints.measurement import render_measurement_prompt_block
from ...constraints.physics import render_physics_prompt_block
from ...sop.expert_sop import render_expert_sop_prompt_block
from ...types import ROOT_CAUSES
from ..confidence_rubric import CONFIDENCE_RUBRIC


DIAGNOSE_PROMPT_VERSION = "rca-diagnose-forced-v5"

ROOT_CAUSE_DEFINITIONS = {
    "L1": "400G 端口一侧的设备或端口根因",
    "L2": "200G 端口一侧的设备或端口根因",
    "fiber": "L1 与 L2 之间的光纤 / 链路介质根因",
}


def build_diagnose_prompt(request: Any, *, retry_feedback: str = "") -> str:
    payload = {
        "case_id": request.case_id,
        "branch": request.branch,
        "routing_reason": request.routing_reason,
        "root_causes": ROOT_CAUSE_DEFINITIONS,
        "available_evidence": list(request.evidence_tokens),
        "missing_fields": list(request.missing_fields),
        "telemetry_status": request.telemetry_status,
        "candidate_root_causes": list(request.candidate_root_causes),
        "deterministic_exclusions": [item.to_dict() for item in request.exclusions],
        "nearest_similarity": request.nearest_similarity,
        "historical_case_ids": list(request.historical_case_ids),
        "historical_label_distribution": dict(request.historical_label_distribution),
        "expert_sop": getattr(request, "expert_sop", None),
        "numeric_decision_tree_path": getattr(request, "decision_tree_prediction", None),
    }
    sections = [
        "你是光链路故障定界专家。当前 case 与历史证据图相似度不足，"
        "必须按专家排障 SOP 的检查顺序，在纯物理约束和量测契约内给出 L1/L2/fiber 三选一结论。\n"
        "遥测不完整不是拒答理由；证据不足必须体现为低 evidence_completeness 或低 reasoning_completeness。\n"
        "量测契约只能否决不可信推理，严禁写入 `cited_constraints` 作为 support/exclude 依据。\n"
        "数值决策树路径只是训练集统计先验，不能作为当前 case 的物理证据或最终结论来源。\n"
        "\n"
        "定界主线（按顺序执行，不要跳步）：\n"
        "1. 先用 P10 把每条接收侧症状翻译成对端：L1 侧接收异常支持 L2，L2 侧接收异常支持 L1。\n"
        "2. 再用 P13 把每条发送侧与电口症状归给本端：L1 侧本地链路异常支持 L1，L2 侧支持 L2。\n"
        "3. 按 SOP 的 S4 裁决两端：只有一端有异常就取该端；两端都有异常取 priority 数值更小的一端。\n"
        "4. `verdict` 必须等于上述裁决胜出的那一端，除非有物理约束步骤明确排除了它。\n"
        "\n"
        "关于 fiber：现有两端遥测通常无法把介质根因与端点根因唯一分开，端点根因是默认解释。"
        "只有同时具备「两端均已发光」（P4/P5）和「同一 lane 双向对称丢失」（P8）时，"
        "才允许把 fiber 写成 `verdict`。仅有单向 tx_ok_rx_down（P7）时 target 取对端端点，不取 fiber。"
        "若在缺少上述双向现场证据的情况下仍判 fiber，`physical_compliance` 必须 <= 0.3，"
        "并在 `missing_information` 中请求 OTDR、端面镜检、双向功率标定或换纤复测。",
        render_expert_sop_prompt_block(),
        render_physics_prompt_block(),
        render_measurement_prompt_block(),
        CONFIDENCE_RUBRIC,
    ]
    if retry_feedback:
        sections.append("上一次回答未通过物理约束校验，请修正以下问题：\n" + retry_feedback)
    sections.append("本 case 证据：\n" + json.dumps(payload, ensure_ascii=False, indent=2))
    sections.append(
        "只输出一个 JSON 对象，结构如下：\n"
        "{\n"
        '  "steps": [{"claim": "...", "cited_evidence": ["..."], '
        '"cited_constraints": ["..."], "effect": "support | exclude | neutral", '
        '"target": "L1 | L2 | fiber | " }],\n'
        '  "verdict": "L1 | L2 | fiber",\n'
        '  "confidence": 0.0,\n'
        '  "confidence_breakdown": {\n'
        '    "evidence_completeness": 0.0,\n'
        '    "physical_compliance": 0.0,\n'
        '    "reasoning_completeness": 0.0,\n'
        '    "history_similarity": 0.0\n'
        "  },\n"
        '  "missing_information": ["..."]\n'
        "}\n"
        f"verdict 只能取 {', '.join(ROOT_CAUSES)}，禁止输出 abstain。\n"
        "verdict 必须与 steps 自洽：把所有 effect=support 的 target 汇总，"
        "减去 effect=exclude 的 target，verdict 取得票最高的那一个。"
        "如果你想给出的 verdict 与 steps 的汇总结果不一致，先补写能支撑它的 support 步骤，"
        "不要直接输出与推理链矛盾的结论。"
    )
    return "\n\n".join(sections)
