"""Low-tier diagnosis prompt with layered knowledge injection."""

from __future__ import annotations

import json
from typing import Any

from ...constraints.measurement import MEASUREMENT_CONTRACT_LIBRARY, render_measurement_prompt_block
from ...constraints.physics import PHYSICS_LIBRARY, render_physics_prompt_block
from ...sop.expert_sop import render_expert_sop_prompt_block
from ...types import ROOT_CAUSES
from ..confidence_rubric import CONFIDENCE_RUBRIC


LEGACY_DIAGNOSE_PROMPT_VERSION = "rca-diagnose-dual-sop-v7-full-step-ids"
FILTERED_RULE_DIAGNOSE_PROMPT_VERSION = "filtered-rule-diagnose-three-channel-v3"
# Backward-compatible public constant used by legacy experiment manifests.
DIAGNOSE_PROMPT_VERSION = LEGACY_DIAGNOSE_PROMPT_VERSION

SOP_STEP_ID_SEQUENCE = (
    "Q0_validate_measurements",
    "P_apply_physical_boundaries",
    "R_expand_directional_chain",
    "L_apply_stable_learned_ranges",
    "D_select_or_request_evidence",
)

LEGACY_ROOT_CAUSE_DEFINITIONS = {
    "L1": "400G 端口一侧的设备或端口根因",
    "L2": "200G 端口一侧的设备或端口根因",
    "fiber": "L1 与 L2 之间的光纤 / 链路介质根因",
}

FILTERED_RULE_ROOT_CAUSE_DEFINITIONS = {
    "L1": "当前 case 本端的设备或端口根因",
    "L2": "当前 case 对端的设备或端口根因",
    "fiber": "L1 与 L2 之间的光纤 / 链路介质根因",
}

BRANCH_INSTRUCTIONS = {
    "N5a": (
        "完全匹配通道：历史证据链只作为可审计上下文；必须基于当前 case 的物理证据"
        "给出独立候选和置信度，不得直接复制历史标签。"
    ),
    "N5b": (
        "部分匹配通道：围绕 shared、missing、conflicting evidence 判断缺失项是否关键，"
        "再使用物理约束完成一次仲裁。"
    ),
    "N5c": (
        "低匹配通道：不得复用历史结论，按专家 SOP、纯物理约束和量测契约完成一次推理。"
    ),
}


def _matches_request(prefixes: tuple[str, ...], tokens: tuple[str, ...]) -> bool:
    return not prefixes or any(token.startswith(prefix) for token in tokens for prefix in prefixes)


def _branch_knowledge_sections(request: Any) -> list[str]:
    """Keep each channel focused while retaining physical and measurement vetoes."""
    branch = getattr(request, "branch", "N5c")
    tokens = tuple(getattr(request, "evidence_tokens", ()))
    if branch == "N5c":
        return [
            render_expert_sop_prompt_block(),
            render_physics_prompt_block(),
            render_measurement_prompt_block(),
        ]
    physics = tuple(
        item for item in PHYSICS_LIBRARY.constraints
        if _matches_request(item.applies_to_token_prefixes, tokens)
    )
    contracts = tuple(
        item for item in MEASUREMENT_CONTRACT_LIBRARY.contracts
        if _matches_request(item.applies_to_token_prefixes, tokens)
    )
    return [
        render_physics_prompt_block(constraints=physics),
        render_measurement_prompt_block(contracts=contracts),
    ]


def diagnose_prompt_version_for(profile: str = "legacy") -> str:
    if profile == "filtered_rule_v1":
        return FILTERED_RULE_DIAGNOSE_PROMPT_VERSION
    return LEGACY_DIAGNOSE_PROMPT_VERSION


def build_diagnose_prompt(
    request: Any,
    *,
    retry_feedback: str = "",
    profile: str = "legacy",
) -> str:
    filtered_rule = profile == "filtered_rule_v1"
    root_cause_definitions = (
        FILTERED_RULE_ROOT_CAUSE_DEFINITIONS
        if filtered_rule
        else LEGACY_ROOT_CAUSE_DEFINITIONS
    )
    payload = {
        "case_id": request.case_id,
        "branch": request.branch,
        "branch_instruction": BRANCH_INSTRUCTIONS.get(request.branch, BRANCH_INSTRUCTIONS["N5c"]),
        "routing_reason": request.routing_reason,
        "root_causes": root_cause_definitions,
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
        "raw_measurements_with_units_and_lane_counts": getattr(request, "raw_measurements", {}),
        "dual_similarity": {
            "S_feature": getattr(request, "feature_similarity", 0.0),
            "S_graph": getattr(request, "graph_similarity", 0.0),
        },
        "five_layer_evidence_paths": list(getattr(request, "evidence_paths", ())),
        "historical_evidence_chains": list(getattr(request, "historical_evidence_chains", ())),
        "opposing_historical_cases": list(getattr(request, "opposing_historical_cases", ())),
        "largest_feature_differences": list(getattr(request, "largest_differences", ())),
        "critical_missing_evidence": list(getattr(request, "critical_missing_evidence", ())),
        "declared_predicates": list(getattr(request, "declared_predicates", ())),
        "executed_sop_trace": list(getattr(request, "sop_trace", ())),
        "deterministic_sop_candidates": list(getattr(request, "sop_candidates", ())),
    }
    if filtered_rule:
        payload["source_and_topology"] = getattr(request, "topology_context", {})
    history_context = (
        "根据当前 case 的历史匹配分支"
        if filtered_rule
        else "当前 case 与历史证据图相似度不足"
    )
    branch_instruction = BRANCH_INSTRUCTIONS.get(request.branch, BRANCH_INSTRUCTIONS["N5c"])
    reasoning_method = (
        "按专家排障 SOP 的检查顺序"
        if request.branch == "N5c"
        else "按该通道提供的历史证据链、差异清单和当前物理证据"
    )
    sections = [
        f"你是光链路故障定界专家。{history_context}。当前为 {request.branch} 分支："
        f"{branch_instruction}\n必须{reasoning_method}，"
        "在纯物理约束和量测契约内给出 L1/L2/fiber 三选一结论。\n"
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
    ]
    sections.extend(_branch_knowledge_sections(request))
    sections.append(CONFIDENCE_RUBRIC)
    if retry_feedback:
        sections.append("上一次回答未通过物理约束校验，请修正以下问题：\n" + retry_feedback)
    sections.append("本 case 证据：\n" + json.dumps(payload, ensure_ascii=False, indent=2))
    sop_id_contract = (
        "N5c 必须使用专家 SOP 中的完整 sop_step_id，并按以下顺序组织相关步骤："
        + " → ".join(SOP_STEP_ID_SEQUENCE)
        + "；禁止使用 Q0/P/R/L/D 等缩写。"
        if request.branch == "N5c"
        else "N5a/N5b 不要求补写未执行的 SOP 步骤；如引用 sop_step_id，必须逐字使用完整 ID。"
    )
    sections.append(
        "只能使用 declared_predicates 中已有的阈值；禁止发明、移动或重新拟合阈值。"
        + sop_id_contract + "\n"
        "首个输出字符必须是 `{`，末个输出字符必须是 `}`；不要输出 `<think>`、Markdown 或 JSON 之外的文字。\n"
        "只输出一个符合下列结构的 JSON 对象：\n"
        "{\n"
        '  "steps": [{"sop_step_id": "P_apply_physical_boundaries", "cited_predicates": ["谓词 ID"], "claim": "...", "cited_evidence": ["证据 ID"], '
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
